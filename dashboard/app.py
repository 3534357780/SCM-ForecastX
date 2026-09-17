"""
SCM-ForecastX 决策看板（Streamlit）。

启动：
    streamlit run dashboard/app.py

看板结构
--------
① KPI 卡：预测精度 / 相对 baseline 提升 / 平均断供概率 / 供应商集中度
② 需求侧：单 SKU 的「实际 vs 预测」时序 + 分层误差分布
③ 供应侧：Kraljic 四象限散点 + ABC-XYZ 矩阵
④ 风险预警：高风险 SKU-供应商清单 + 行动 SOP
⑤ 决策建议：结构化叙述（模板生成，数字可追溯到计算引擎）
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

st.set_page_config(page_title="SCM-ForecastX 供应链决策看板", layout="wide")

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
    return d


def main():
    st.title("SCM-ForecastX · 供应链需求预测与供应商风险驾驶舱")
    st.caption(f"数据源：M5 Walmart 真实零售数据（{C.STATE} / {C.STORE} / {C.DEPT}）"
               f"＋ 参数化情景模拟的供应商层")

    if not (C.METRICS_FILE.exists()):
        st.warning("尚未生成结果，请先依次运行：`python -m src.prepare` → "
                   "`python -m src.forecast` → `python -m src.risk` → "
                   "`python -m src.plan` → `python -m src.narrative`")
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
    c3.metric("平均断供概率", f"{d['kpi']['avg_stockout_prob']:.1%}", "现行备货口径")
    c4.metric("供应商集中度 HHI", f"{d['kpi']['hhi']:.3f}",
              f"最大供应商占 {d['kpi']['top1_share']:.1%}", delta_color="inverse")
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
        st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(figk, use_container_width=True)

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
        st.plotly_chart(figz, use_container_width=True)
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
        st.plotly_chart(fige, use_container_width=True)

    st.divider()
    st.subheader("⚠️ 风险预警：优先处理清单")
    hi = (plan[plan["risk_level"].isin(["高", "极高"])]
          .sort_values("annual_spend", ascending=False)
          .head(20)[["supplier_id", "item_id", "region", "abc_class", "xyz_class",
                     "kraljic", "risk_level", "qcdsm_score", "p_stockout",
                     "lead_time_days", "safety_stock", "reorder_point", "action"]])
    hi = hi.rename(columns={"supplier_id": "供应商", "item_id": "SKU", "region": "区域",
                            "abc_class": "ABC", "xyz_class": "XYZ", "kraljic": "象限",
                            "risk_level": "风险", "qcdsm_score": "QCDSM",
                            "p_stockout": "断供概率", "lead_time_days": "交期(天)",
                            "safety_stock": "安全库存", "reorder_point": "再订货点",
                            "action": "建议行动"})
    st.dataframe(hi.style.format({"QCDSM": "{:.1f}", "断供概率": "{:.1%}",
                                  "交期(天)": "{:.0f}", "安全库存": "{:.1f}",
                                  "再订货点": "{:.1f}"}),
                 use_container_width=True, height=320)

    st.divider()
    st.subheader("决策建议（结构化叙述）")
    st.code(d["narrative"], language=None)


if __name__ == "__main__":
    main()
