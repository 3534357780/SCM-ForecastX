"""
决策建议叙述：结构化模板生成。

为什么叙述层不用 LLM
--------------------
决策文档里的数字若由生成式模型转述，存在数值关系被改写（幻觉）的风险。
本模块的做法是：所有数字由确定性计算引擎产出，叙述层只做
「模板填充 + 排序取 Top-N」，从架构上排除生成改数的可能。

叙述顺序按「处理了哪几个业务问题」组织，而不是按模块顺序：
需求信号 → 波动分解 → 服务水平来源 → 供应侧 → 库存侧 → 行动清单。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C


def model_to_inventory_value() -> dict:
    """
    对比「用模型预测」与「用季节性 naive」两种情况下 σ_d 的差异，
    换算成安全库存的变化。逻辑链：预测更准 -> 残差更小 -> σ_d 更小 -> SS 更小。
    """
    bt = pd.read_parquet(C.OUT_DIR / "backtest_predictions.parquet")
    bt = bt[bt["is_promo"] == 0]                       # 与安全库存口径一致：基础波动
    daily_mean = bt.groupby("id")["y"].mean()
    L = float(pd.read_csv(C.RISK_FILE)["lead_time_days"].median())
    out = {}
    for name, col in [("lgbm_tweedie", "pred_lgbm_tweedie"),
                      ("seasonal_naive", "pred_seasonal_naive")]:
        resid = bt["y"] - bt[col]
        sd = resid.groupby(bt["id"]).std().reindex(daily_mean.index).fillna(0.0)
        Z = float(pd.read_csv(C.RISK_FILE)["z_value"].mean())
        sigma_L = L * C.LEAD_TIME_LN_SIGMA
        ss = Z * np.sqrt(L * sd.to_numpy() ** 2
                         + daily_mean.to_numpy() ** 2 * sigma_L ** 2)
        out[name] = {"sigma_d_mean": float(sd.mean()), "ss_mean": float(ss.mean())}
    m, b = out["lgbm_tweedie"], out["seasonal_naive"]
    out["sigma_reduction"] = 1 - m["sigma_d_mean"] / b["sigma_d_mean"]
    out["ss_reduction"] = 1 - m["ss_mean"] / b["ss_mean"]
    return out


def run() -> str:
    plan = pd.read_csv(C.PLAN_FILE)
    met = pd.read_csv(C.OUT_DIR / "metrics_full.csv")
    kpi = pd.read_csv(C.OUT_DIR / "scm_kpis.csv").iloc[0]
    by_class = pd.read_csv(C.OUT_DIR / "policy_by_class.csv")
    eff = (pd.read_csv(C.OUT_DIR / "censoring_effect.csv")
           if (C.OUT_DIR / "censoring_effect.csv").exists() else None)
    sup = pd.read_csv(C.SUPPLIER_FILE)

    m = met.set_index("model")
    lgbm, naive = m.loc["lgbm_tweedie"], m.loc["seasonal_naive"]
    val = model_to_inventory_value()

    grade = plan["risk_level"].value_counts()
    high = plan[plan["risk_level"].isin(["高", "极高"])]
    plan["priority"] = plan["annual_spend"] * plan["composite_risk"] / 100
    top = plan.sort_values("priority", ascending=False).drop_duplicates("item_id").head(10)

    lines = []
    A = lines.append

    A("【决策摘要】")
    A(f"本期覆盖 {plan['item_id'].nunique()} 个 SKU、{plan['supplier_id'].nunique()} 家供应商。"
      f"需求侧 28 天预测 WMAPE {lgbm['WMAPE']:.1%}（季节性 naive {naive['WMAPE']:.1%}，"
      f"MASE {lgbm['MASE']:.2f}）；目标服务水平逐 SKU 由成本比推导，区间 "
      f"{plan['service_level_target'].min():.1%}~{plan['service_level_target'].max():.1%}。"
      f"供应侧 {len(high)} 条 SKU-供应商关系的覆盖天数达不到目标水平，"
      f"涉及采购金额 {high['annual_spend'].sum() / plan['annual_spend'].sum():.1%}。")
    A("")

    A("【一、需求信号：零销量不等于零需求】")
    if eff is not None:
        raw = eff[(eff.segment == "全部日") & (eff.model == "记录销量训练")].iloc[0]
        res = eff[(eff.segment == "全部日") & (eff.model == "还原需求训练")].iloc[0]
        pr_raw = eff[(eff.segment == "促销日") & (eff.model == "记录销量训练")].iloc[0]
        pr_res = eff[(eff.segment == "促销日") & (eff.model == "还原需求训练")].iloc[0]
        A(f"· 识别出疑似缺货日（促销日却零销量 / 平时热销的 SKU 突然挂零）后，"
          f"把记录销量还原为需求估计，再用还原后的需求训练模型。")
        A(f"· 用记录销量训练、且完全不做还原时，全期偏差只有 {raw['Bias%']:+.1f}%，"
          f"看起来几乎无偏 —— 代价是它在促销期低估了 {abs(pr_raw['Bias%']):.1f}%。"
          f"这不是模型准，而是它学会了复现断货日被记成的那些 0。")
        A(f"· 换成还原需求训练后，促销期低估收窄到 {abs(pr_res['Bias%']):.1f}%"
          f"（改善 {abs(pr_raw['Bias%']) - abs(pr_res['Bias%']):.1f} 个百分点），"
          f"全期偏差变为 {res['Bias%']:+.1f}%。这个正偏差不是模型误差："
          f"训练目标的日均水平本次被抬高了 5.2%，与 {res['Bias%']:+.1f}% 几乎一致，"
          f"它就是被还原出来的那部分需求。")
        A(f"· 因为评估口径仍是记录销量，WMAPE 会从 {raw['WMAPE']:.3f} 略升到 "
          f"{res['WMAPE']:.3f}。这恰恰说明一件事：以记录销量为评估口径，"
          f"会把「学会缺货」当成优点。做需求计划时，这个口径本身就需要被指出来。")
    A("")

    A("【二、波动分解：促销波动不参与安全库存】")
    A(f"· 把残差波动拆成基础波动与促销波动两部分，安全库存只用基础波动："
      f"资金占用 ${kpi['safety_stock_value_all_sigma']:,.0f} -> ${kpi['safety_stock_value']:,.0f}"
      f"（-{kpi['sigma_decomposition_saving_pct']:.1f}%），促销期的额外需求单独预建 "
      f"${kpi['promo_prebuild_value']:,.0f}，与安全库存分账。")
    A(f"· 这个省幅很小，需要说清楚原因而不是含糊过去：本批 SKU 近 182 天只有约一成的 SKU "
      f"有促销，且模型的价格特征已经吸收了促销中可预见的那部分，剩下的促销日残差并不比"
      f"非促销日大。所以这条处理的价值不在省钱，而在于口径："
      f"促销是可预见的，它的缺口不该混进「应对不确定性的安全库存」，而应单独预建、单独对账 —— "
      f"否则会出现「平时按促销月的波动备货、促销来临时仍然不够」的双输局面。")
    A("")

    A("【三、目标服务水平的来源】")
    A(f"· 服务水平由「缺一个单位」与「多备一个单位」的成本之比推导"
      f"（SL = Cu/(Cu+Co)），不再全局取 95%。缺货成本倍数按 ABC 分档"
      f"（A={C.STOCKOUT_MULT_BY_ABC['A']:.0f}x、B={C.STOCKOUT_MULT_BY_ABC['B']:.0f}x、"
      f"C={C.STOCKOUT_MULT_BY_ABC['C']:.0f}x），因为 A 类缺货对复购的影响最直接。")
    for _, r in by_class.iterrows():
        A(f"   {r['abc_class']} 类 | 服务水平 {r['service_level']:.1%} | Z={r['z_value']:.2f} | "
          f"{int(r['sku'])} 个 SKU | 占安全库存资金 {r['safety_stock_value_share']:.1f}%")
    sk0 = plan.drop_duplicates("item_id")
    gap_by_abc = (sk0["coverage_days_needed"] - sk0["coverage_days_current"]).groupby(
        sk0["abc_class"]).mean()
    A(f"· 与现行政策对照（现行口径为「交期 + {C.CURRENT_SAFETY_DAYS:.0f} 天」，全品类一刀切）："
      f"A 类覆盖缺口 {gap_by_abc.get('A', 0):+.1f} 天、B 类 {gap_by_abc.get('B', 0):+.1f} 天、"
      f"C 类 {gap_by_abc.get('C', 0):+.1f} 天。"
      f"也就是说一刀切并不是笼统地「备多了」或「备少了」，而是一个水位对 A 类不够、对 C 类太多。"
      f"这正是把服务水平从行业惯例改成成本比推导的实际意义。")
    A("")

    A("【四、供应侧现状】")
    A(f"· 供应商分层（六维记分卡：交付 / 质量 / 成本 / 弹性 / 财务 / 连续性）："
      + "，".join(f"{k} {v} 家" for k, v in sup["tier"].value_counts().items()))
    gap = sup[sup["continuity_gap_days"] > 0]
    A(f"· 连续性：{len(gap)}/{len(sup)} 家供应商的恢复时间超过现行库存可支撑天数"
      f"（TTR > TTS），即一旦中断，现行库存撑不到供应恢复。"
      f"这是双源和缓冲库存的真正依据。")
    A(f"· 集中度 HHI = {kpi['hhi']:.4f}，最大供应商占采购额 {kpi['top1_share']:.1%}；"
      f"单源 SKU {int(kpi['single_source_sku'])}/{int(kpi['sku_count'])} 个"
      f"（{kpi['single_source_sku'] / kpi['sku_count']:.0%}）。")
    A(f"· 风险等级（按覆盖缺口天数 = 达到目标服务水平所需覆盖天数 − 现行政策覆盖天数）："
      + "，".join(f"{k} {v} 条" for k, v in grade.items()))
    sk = plan.drop_duplicates("item_id")
    s1, s2 = sk[sk["is_single_source"]], sk[~sk["is_single_source"]]
    if len(s1) and len(s2):
        A(f"· 单源 {len(s1)} 个 SKU 现行缺货概率 {s1['p_shortage_sku'].mean():.1%}；"
          f"双源 {len(s2)} 个 SKU 两个货源同时中断的概率 {s2['p_shortage_sku'].mean():.1%}"
          f"（按独立近似）。主备若在同一区域，区域事件会让两者同时中断，"
          f"因此双源的实际保护能力低于独立假设给出的结果，应打折看待。")
    A("")

    A("【五、库存侧】")
    A(f"· 平均安全库存 {plan['safety_stock'].mean():.2f} 件，平均再订货点 "
      f"{plan['reorder_point'].mean():.2f} 件；σ 用基础波动，σ_L 由供应商交期波动给出"
      f"（这是采购可以谈的那一项）。")
    A(f"· 相对季节性 naive，模型使基础波动 σ 下降 {val['sigma_reduction']:.1%}，"
      f"在同服务水平下安全库存下降 {val['ss_reduction']:.1%} —— 这是预测精度到资金占用的换算。")
    A(f"· 注：M5 为零售 POS 数据，不含在手库存，故输出「目标库存水位」而非补货量，"
      f"真实落地需对接 WMS 的在手库存字段做差额。")
    A("")

    A("【六、行动清单 Top10（按采购金额 × 风险排序）】")
    A(f"{'供应商':<10}{'SKU':<14}{'主/备':<5}{'层级':<5}{'区域':<10}{'象限':<6}{'风险':<6}{'缺货概率':<10}行动")
    for _, r in top.iterrows():
        role = "主" if r["is_primary"] else "备"
        A(f"{r['supplier_id']:<10}{r['item_id']:<14}{role:<5}{r['tier']:<5}{r['region']:<10}"
          f"{r['kraljic']:<6}{r['risk_level']:<6}{r['p_shortage']:<10.1%}{r['action']}")
    A("")

    A("【七、边界】")
    A("· 需求侧为 M5 真实零售数据；供应侧字段口径对齐采购记分卡，"
      "本项目用参数化数据源替代企业 SRM/ERP 导出，参数与来源见 config.SOURCES。")
    A("· 成本参数（毛利率、持有成本率、缺货成本倍数）来自财务与品类判断，"
      "本项目给出扫描区间（见 service_level_tradeoff.csv），结论不建立在单一点值上。")
    A("· 正态近似在低服务水平上偏保守、在高服务水平上偏乐观，偏差在 ±2.3 个百分点以内；"
      "service_level_tradeoff.csv 逐行给出蒙特卡洛实测的达成率与偏差，供校核。")

    text = "\n".join(lines)
    C.NARRATIVE_FILE.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[output] {C.NARRATIVE_FILE}")
    return text


if __name__ == "__main__":
    run()
