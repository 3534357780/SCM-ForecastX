"""
零售补货计划决策看板（Streamlit）。

启动：
    streamlit run dashboard/app.py

看板结构
--------
① KPI 卡：预测精度 / 相对 baseline 提升 / 现行政策缺货概率 / 目标服务水平
② 需求侧：单 SKU 的「实际 vs 预测」时序 + 误差随期数变化
③ 供应侧：Kraljic 四象限散点 + ABC-XYZ 矩阵
④ 需求信号与服务水平的处理结果（缺货还原效果、成本比 -> 服务水平权衡）
⑤ 风险预警：优先处理清单 + 行动 SOP
⑥ 决策建议：结构化叙述（模板生成，数字可追溯到计算引擎）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C

st.set_page_config(page_title="零售补货计划看板", layout="wide")

RED, GREEN, BLUE, GREY = "#d62728", "#2ca02c", "#1f77b4", "#8c8c8c"


@st.cache_data
def load():
    d = {}
    d["metrics"] = pd.read_csv(C.OUT_DIR / "metrics_full.csv")
    d["kpi"] = pd.read_csv(C.OUT_DIR / "scm_kpis.csv").iloc[0]
    d["plan"] = pd.read_csv(C.PLAN_FILE)
    d["pred"] = pd.read_parquet(C.OUT_DIR / "backtest_predictions.parquet")
    d["by_h"] = pd.read_csv(C.OUT_DIR / "backtest_by_horizon.csv")
    d["narrative"] = C.NARRATIVE_FILE.read_text(encoding="utf-8")

    def _opt(name):
        p = C.OUT_DIR / name
        return pd.read_csv(p) if p.exists() else None

    d["eff"] = _opt("censoring_effect.csv")
    d["tradeoff"] = _opt("service_level_tradeoff.csv")
    d["by_class"] = _opt("policy_by_class.csv")
    d["scorecard"] = _opt(C.SUPPLIER_FILE.name)
    return d


def main():
    st.title("零售补货策略：需求还原与库存决策")
    st.caption(f"数据源：M5 Walmart 真实零售数据（{C.STATE} / {C.STORE} / {C.DEPT}）"
               f"＋ 采购记分卡口径的供应侧输入层")

    if not (C.METRICS_FILE.exists()):
        st.warning("尚未生成结果，请先运行：`python run_all.py`")
        return

    d = load()
    m = d["metrics"].set_index("model")
    lgbm, naive = m.loc["lgbm_tweedie"], m.loc["seasonal_naive"]

    # ① KPI 卡
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("28 天期 WMAPE", f"{lgbm['WMAPE']:.1%}",
              f"{(lgbm['WMAPE'] - naive['WMAPE']) * 100:+.1f}pt vs 季节性 naive",
              delta_color="inverse")
    c2.metric("MASE", f"{lgbm['MASE']:.2f}", "< 1 才算赢过朴素方法")
    c3.metric("现行政策缺货概率", f"{d['kpi']['avg_shortage_prob_current']:.1%}",
              "按「交期 + 7 天」口径测算")
    c4.metric("平均目标服务水平", f"{d['kpi']['avg_service_level_target']:.1%}",
              "A 89% / B 84% / C 73%（按 ABC 自动分档）",
              delta_color="off")
    st.divider()

    left, right = st.columns([3, 2])

    # ② 需求侧：单 SKU 时序
    with left:
        st.subheader("需求预测 vs 实际（可按 SKU 下钻）")
        plan = d["plan"]
        skus = sorted(d["pred"]["id"].unique())
        default = plan.sort_values("annual_spend", ascending=False)["item_id"].iloc[0]
        pick = st.selectbox("选择 SKU", skus,
                            index=skus.index([s for s in skus if s.startswith(default)][0])
                            if any(s.startswith(default) for s in skus) else 0)
        sub = d["pred"][d["pred"]["id"] == pick].sort_values("date")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sub["date"], y=sub["y"], name="实际销量",
                                 mode="lines", line=dict(color="#333333", width=2)))
        fig.add_trace(go.Scatter(x=sub["date"], y=sub["pred_lgbm_tweedie"], name="模型预测",
                                 mode="lines", line=dict(color=BLUE, width=1.6)))
        fig.add_trace(go.Scatter(x=sub["date"], y=sub["pred_seasonal_naive"], name="季节性 naive",
                                 mode="lines", line=dict(color=GREY, width=1, dash="dot")))
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", y=1.1), plot_bgcolor="white")
        st.plotly_chart(fig, width="stretch")

    # ③ 供应侧：Kraljic
    with right:
        st.subheader("Kraljic 采购矩阵")
        pl = plan.copy()
        pl["log_spend"] = np.log10(pl["annual_spend"].clip(lower=1))
        figk = px.scatter(pl, x="log_spend", y="composite_risk", color="kraljic",
                          size="reorder_point", hover_name="item_id",
                          color_discrete_map={"战略": RED, "瓶颈": "#ff7f0e",
                                              "杠杆": BLUE, "常规": GREEN},
                          labels={"log_spend": "采购金额（log10）", "composite_risk": "综合供应风险"})
        figk.add_hline(y=pl["composite_risk"].median(), line_dash="dot", line_color=GREY)
        figk.add_vline(x=pl["log_spend"].median(), line_dash="dot", line_color=GREY)
        figk.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white")
        st.plotly_chart(figk, width="stretch")

    st.divider()
    a, b = st.columns([1, 1])

    # ④ ABC-XYZ
    with a:
        st.subheader("ABC × XYZ 库存策略矩阵")
        pl = plan.copy()
        pl["abc"] = pl["abc_class"]
        pl["xyz"] = pl["xyz_class"]
        grid = pl.pivot_table(index="abc", columns="xyz", values="item_id",
                              aggfunc="nunique", fill_value=0)
        figz = px.imshow(grid, text_auto=True, color_continuous_scale="Reds",
                         labels=dict(x="XYZ（需求波动）", y="ABC（金额）", color="SKU 数"))
        figz.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(figz, width="stretch")
        st.caption("AX 类走自动化补货；AZ/CZ 类才是真正需要人工干预的长尾。")

    # ⑤ 误差随期数
    with b:
        st.subheader("误差随预测期数变化")
        bh = d["by_h"].groupby("h")[["lgbm_tweedie", "seasonal_naive", "ma28"]].mean().reset_index()
        fige = go.Figure()
        for col, name, color in [("lgbm_tweedie", "模型", BLUE),
                                 ("seasonal_naive", "季节性 naive", GREY),
                                 ("ma28", "28 日均值", "#ff7f0e")]:
            fige.add_trace(go.Scatter(x=bh["h"], y=bh[col], name=name, line=dict(color=color)))
        fige.update_layout(height=300, xaxis_title="预测期数 h（天）", yaxis_title="WMAPE",
                           margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                           legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fige, width="stretch")

    st.divider()
    st.subheader("需求信号处理与服务水平的来源")

    s1, s2 = st.columns([1, 1])
    with s1:
        st.markdown("**零销量不等于零需求** —— 同一模型结构，只换训练目标")
        if d["eff"] is not None:
            piv = (d["eff"].pivot(index="segment", columns="model",
                                  values=["WMAPE", "Bias%"])
                   .reindex(["全部日", "非促销日", "促销日"]))
            piv.columns = [f"{a} · {b}" for a, b in piv.columns]
            st.dataframe(piv.style.format("{:+.3f}"), width="stretch")
            st.caption("Bias% 越接近 0 表示系统性偏差越小；促销日与断货日的改善最明显。")
        else:
            st.info("未找到 censoring_effect.csv")

    with s2:
        st.markdown("**目标服务水平由成本比推导**（SL = Cu / (Cu+Co)）")
        if d["tradeoff"] is not None:
            t = d["tradeoff"][["stockout_multiple", "implied_service_level",
                               "total_safety_stock_value", "achieved_service_level_mc"]]
            t = t.rename(columns={"stockout_multiple": "缺货成本倍数",
                                  "implied_service_level": "目标服务水平",
                                  "total_safety_stock_value": "安全库存资金",
                                  "achieved_service_level_mc": "MC 实测达成"})
            st.dataframe(t.style.format({"缺货成本倍数": "{:.1f}x",
                                         "目标服务水平": "{:.1%}",
                                         "安全库存资金": "${:,.0f}",
                                         "MC 实测达成": "{:.1%}"}),
                         width="stretch")
            st.caption("把缺货看得越贵，目标服务水平与库存资金越高；最后一列是"
                       "蒙特卡洛实测达成率，与目标的差额即正态近似的偏差。")
        else:
            st.info("未找到 service_level_tradeoff.csv")

    st.divider()
    st.subheader("一刀切的服务水平在 A 类欠备、在 C 类过备")
    if d["by_class"] is not None:
        bc = d["by_class"][["abc_class", "sku", "service_level", "target_shortage",
                            "shortage_prob_current", "coverage_gap_days",
                            "safety_stock_value_share"]].copy()
        bc = bc.rename(columns={"abc_class": "ABC", "sku": "SKU 数",
                                "service_level": "目标服务水平",
                                "target_shortage": "目标缺货率",
                                "shortage_prob_current": "现行政策缺货概率",
                                "coverage_gap_days": "覆盖缺口(天)",
                                "safety_stock_value_share": "占安全库存资金"})
        st.dataframe(bc.style.format({"目标服务水平": "{:.1%}", "目标缺货率": "{:.1%}",
                                      "现行政策缺货概率": "{:.1%}", "覆盖缺口(天)": "{:+.1f}",
                                      "占安全库存资金": "{:.1f}%"}),
                     width="stretch", hide_index=True)
        st.caption("覆盖缺口 = 达到目标服务水平所需覆盖天数 − 现行「交期 + 7 天」的覆盖天数。"
                   "正数表示现行水位对这类 SKU 不够，负数表示备多了 —— "
                   "一刀切的问题不是总量，是结构。")
    else:
        st.info("未找到 policy_by_class.csv")

    st.divider()
    st.subheader("供应商记分卡（六维）")
    st.caption("供应商评价用多维记分卡，不是单一的「断供概率」。权重见 config.SCORECARD_WEIGHTS。")
    if d["scorecard"] is not None:
        sc = d["scorecard"]
        t1, t2 = st.columns([1, 3])
        with t1:
            cnt = sc["tier"].value_counts().reindex(["优选", "合格", "观察", "淘汰"]).fillna(0)
            st.dataframe(cnt.rename("家数").to_frame(), width="stretch")
        with t2:
            cols = ["supplier_id", "region", "tier", "supplier_score", "D", "Q", "C",
                    "F", "Fin", "Cont", "otd_rate", "dppm", "lead_time_days",
                    "ttr_days", "tts_days", "continuity_gap_days"]
            show = sc.sort_values("supplier_score", ascending=False).head(15)[cols].rename(
                columns={"supplier_id": "供应商", "region": "区域", "tier": "层级",
                         "supplier_score": "综合分", "D": "交付", "Q": "质量", "C": "成本",
                         "F": "弹性", "Fin": "财务", "Cont": "连续性", "otd_rate": "OTD",
                         "dppm": "DPPM", "lead_time_days": "交期(天)",
                         "ttr_days": "恢复(天)", "tts_days": "可支撑(天)",
                         "continuity_gap_days": "缺口(天)"})
            st.dataframe(show.style.format({"综合分": "{:.1f}", "交付": "{:.1f}", "质量": "{:.1f}",
                                            "成本": "{:.1f}", "弹性": "{:.1f}", "财务": "{:.1f}",
                                            "连续性": "{:.1f}", "OTD": "{:.1%}",
                                            "交期(天)": "{:.0f}", "恢复(天)": "{:.0f}",
                                            "可支撑(天)": "{:.0f}", "缺口(天)": "{:+.0f}"}),
                         width="stretch", height=300)
            st.caption("「缺口(天)」= 恢复时间 TTR − 现行库存可支撑天数 TTS；为正即存在覆盖缺口。")
    else:
        st.info("未找到供应商记分卡，请先运行 src.risk")

    st.divider()
    st.subheader("⚠️ 风险预警：优先处理清单")
    plan = plan.copy()
    plan["主/备"] = np.where(plan["is_primary"], "主", "备")
    hi = (plan[plan["risk_level"].isin(["高", "极高"])]
          .sort_values("annual_spend", ascending=False)
          .head(20)[["supplier_id", "item_id", "主/备", "tier", "region", "abc_class",
                     "xyz_class", "kraljic", "risk_level", "supplier_score", "p_shortage",
                     "ttr_days", "tts_days", "lead_time_days", "safety_stock",
                     "reorder_point", "action"]])
    hi = hi.rename(columns={"supplier_id": "供应商", "item_id": "SKU", "tier": "层级",
                            "region": "区域", "abc_class": "ABC", "xyz_class": "XYZ",
                            "kraljic": "象限", "risk_level": "风险",
                            "supplier_score": "记分卡", "p_shortage": "缺货概率",
                            "ttr_days": "恢复(天)", "tts_days": "可支撑(天)",
                            "lead_time_days": "交期(天)", "safety_stock": "安全库存",
                            "reorder_point": "再订货点", "action": "建议行动"})
    st.dataframe(hi.style.format({"记分卡": "{:.1f}", "缺货概率": "{:.1%}",
                                  "恢复(天)": "{:.0f}", "可支撑(天)": "{:.0f}",
                                  "交期(天)": "{:.0f}", "安全库存": "{:.1f}",
                                  "再订货点": "{:.1f}"}),
                 width="stretch", height=320)
    st.caption("「恢复(天)」为供应中断后恢复正常供给所需天数（TTR），"
               "「可支撑(天)」为现行库存可维持天数（TTS）；前者大于后者即为覆盖缺口。")

    st.divider()
    st.subheader("决策建议（结构化叙述）")
    st.code(d["narrative"], language=None)


if __name__ == "__main__":
    main()
