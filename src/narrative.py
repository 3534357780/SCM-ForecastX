"""
决策建议叙述：结构化模板生成。

为什么叙述层不用 LLM
--------------------
决策文档里的数字若由生成式模型转述，存在数值关系被改写（幻觉）的风险。
本模块的做法是：所有数字由确定性计算引擎产出，叙述层只做
「模板填充 + 排序取 Top-N」，从架构上排除生成改数的可能。

本模块还包含「把预测精度翻译成业务价值」的换算：模型指标本身没有商业意义，
换算成安全库存与资金占用才能进入管理层的决策语言。
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
    换算成安全库存与资金占用的变化。

    逻辑链：预测更准 -> 残差更小 -> σ_d 更小 -> SS = Z·sqrt(L·σ_d²+d̄²·σ_L²) 更小
    """
    bt = pd.read_parquet(C.OUT_DIR / "backtest_predictions.parquet")
    daily_mean = bt.groupby("id")["y"].mean()
    # 用实际供应商交期中位数，而不是硬编码常数
    L = float(pd.read_csv(C.RISK_FILE)["lead_time_days"].median())
    out = {}
    for name, col in [("lgbm_tweedie", "pred_lgbm_tweedie"),
                      ("seasonal_naive", "pred_seasonal_naive")]:
        resid = bt["y"] - bt[col]
        sd = resid.groupby(bt["id"]).std().reindex(daily_mean.index).fillna(0.0)
        sigma_L = L * C.LEAD_TIME_LN_SIGMA
        Z = C.Z_TABLE[C.SERVICE_LEVEL]
        ss = Z * np.sqrt(L * sd.to_numpy() ** 2 + daily_mean.to_numpy() ** 2 * sigma_L ** 2)
        out[name] = {"sigma_d_mean": float(sd.mean()), "ss_mean": float(ss.mean())}
    m, b = out["lgbm_tweedie"], out["seasonal_naive"]
    out["sigma_reduction"] = 1 - m["sigma_d_mean"] / b["sigma_d_mean"]
    out["ss_reduction"] = 1 - m["ss_mean"] / b["ss_mean"]
    return out


def run() -> str:
    plan = pd.read_csv(C.PLAN_FILE)
    met = pd.read_csv(C.OUT_DIR / "metrics_full.csv")
    kpi = pd.read_csv(C.OUT_DIR / "scm_kpis.csv").iloc[0]

    m = met.set_index("model")
    lgbm, naive = m.loc["lgbm_tweedie"], m.loc["seasonal_naive"]
    val = model_to_inventory_value()

    grade = plan["risk_level"].value_counts()
    high = plan[plan["risk_level"].isin(["高", "极高"])]

    # 行动清单：按「采购金额 × 风险」的综合优先级排序，去重到 SKU 层面
    plan["priority"] = plan["annual_spend"] * plan["composite_risk"] / 100
    top = (plan.sort_values("priority", ascending=False)
               .drop_duplicates("item_id")
               .head(10))

    lines = []
    A = lines.append

    A("【决策摘要】")
    A(f"本期覆盖 {plan['item_id'].nunique()} 个 SKU、{plan['supplier_id'].nunique()} 家供应商。"
      f"需求侧 28 天预测的 WMAPE 为 {lgbm['WMAPE']:.1%}（季节性 naive 为 {naive['WMAPE']:.1%}，"
      f"MASE {lgbm['MASE']:.2f}）。供应侧有 {grade.get('高', 0) + grade.get('极高', 0)} 条 SKU-供应商关系"
      f"处于高风险及以上，涉及采购金额占比 "
      f"{high['annual_spend'].sum() / plan['annual_spend'].sum():.1%}。"
      f"建议本期优先处理以下 Top10 事项，其余按分类策略常规运转。")
    A("")

    A("【需求侧现状】")
    A(f"· 模型 28 天期 WMAPE {lgbm['WMAPE']:.1%}，Bias {lgbm['Bias%']:+.1f}%"
      f"（{'系统性高估' if lgbm['Bias%'] > 0 else '系统性低估'}，需在计划环节做偏差修正）")
    A(f"· 相对季节性 naive 改善 {(1 - lgbm['WMAPE'] / naive['WMAPE']):.1%}（MASE {lgbm['MASE']:.2f}<1 表示确有价值）")
    xyz = plan.groupby("item_id")["xyz_class"].first().value_counts()
    A(f"· 需求波动分档：X 类 {xyz.get('X', 0)} 个、Y 类 {xyz.get('Y', 0)} 个、Z 类 {xyz.get('Z', 0)} 个；"
      f"Z 类 SKU 建议走低频评审 + 高安全库存，不适合按日滚动预测")
    A("")

    A("【供应侧现状】")
    A(f"· 供应商集中度 HHI = {kpi['hhi']:.3f}"
      f"（{'偏高，存在单点失效风险' if kpi['hhi'] > 0.15 else '尚可'}），最大供应商占采购额 {kpi['top1_share']:.1%}")
    A(f"· 风险等级分布：" + "，".join(f"{k} {v} 条" for k, v in grade.items()))
    A(f"· 平均断供概率 {kpi['avg_stockout_prob']:.2%}"
      f"（按现行「交期 + {C.CURRENT_SAFETY_DAYS:.0f} 天安全库存」口径测算），"
      f"单源 SKU 共 {plan['is_single_source'].sum()} 条")
    A("")

    A("【库存侧现状】")
    A(f"· 在 {C.SERVICE_LEVEL:.0%} 目标周期服务水平下（Z = {C.Z_TABLE[C.SERVICE_LEVEL]}），"
      f"平均安全库存 {plan['safety_stock'].mean():.2f} 件，平均再订货点 {plan['reorder_point'].mean():.2f} 件")
    A(f"· 注：M5 为零售 POS 数据，不含在手库存，故本项目输出「目标库存水位」而非补货量；"
      f"真实落地需对接 WMS 的在手库存字段做差额换算")
    A("")

    A("【行动清单 Top10（按采购金额 × 风险排序）】")
    A(f"{'供应商':<10}{'SKU':<14}{'区域':<10}{'象限':<6}{'风险':<6}{'断供概率':<10}行动")
    for _, r in top.iterrows():
        A(f"{r['supplier_id']:<10}{r['item_id']:<14}{r['region']:<10}{r['kraljic']:<6}"
          f"{r['risk_level']:<6}{r['p_stockout']:<10.1%}{r['action']}")
    A("")

    A("【预期影响（把预测精度翻译成业务价值）】")
    A(f"· 相比季节性 naive，模型使平均预测残差标准差 σ_d 下降 {val['sigma_reduction']:.1%}，"
      f"在同服务水平下安全库存下降 {val['ss_reduction']:.1%}")
    A(f"· 也就是说：需求预测的精度提升不是「模型指标好看」，而是直接对应"
      f"{val['ss_reduction']:.1%} 的安全库存资金释放与同等缺货率下的库存周转改善")
    A(f"· 反向验证：若把目标服务水平从 90% 提到 95%，Z 从 1.282 升到 1.645，"
      f"安全库存将上升 {(1.645 / 1.282 - 1):.1%}——保供与成本的权衡必须显式做，而不是拍脑袋")

    text = "\n".join(lines)
    C.NARRATIVE_FILE.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[output] {C.NARRATIVE_FILE}")
    return text


if __name__ == "__main__":
    run()
