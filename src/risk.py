"""
供应商风险层：QCDSM 规则评分 + 交期/需求蒙特卡洛模拟。

数据边界
--------
M5 是零售 POS 数据，不含供应商信息。企业采购主数据（供应商名录、实际交期、
质量记录、价格协议）属于商业敏感数据，公开数据集不提供。因此本层采用
**参数化情景模拟**：分布参数取行业公开区间与专家先验，全部在 config.py 中
显式声明、随机种子固定，保证可复现、可辩护；输出列名对齐采购主数据字段，
可直接替换为 ERP/SRM 导出表。

建模结构
--------
供应商是**实体**，不是 SKU 的附属属性：
  ① 先生成供应商主表 —— 一家供应商只有一套 OTD / DPPM / 交期 / 财务健康度；
  ② 再生成 SKU-供应商供货关系 —— 一家供应商可服务多个 SKU（符合真实采购格局）；
  ③ 评分在**供应商层面**做（QCDSM 评价的是供应商，不是 SKU）；
  ④ 断供概率在**供货关系层面**做（同一家供应商，供货给高需求 SKU 的断供风险更高）。
若每个 SKU 各配一个独立参数的"供应商"，集中度指标（HHI）会失去意义。

方法选择：为什么是「规则评分 + 概率模拟」而不是训练分类器
--------------------------------------------------------
「供应商风险」缺乏可信的监督标签——现实中风险事件稀疏且滞后。常见做法是用
特征加权公式合成标签再训练模型，但标签与特征同源，模型的"准确率"只是在
拟合造标签的公式，没有泛化含义。本层把「风险」拆成两件互不循环的事：
  A. **评分（规则，不学习）**：QCDSM 五维加权打分 -> 透明、可解释、可调权重。
  B. **概率（模拟，不学习）**：蒙特卡洛模拟交期波动与需求波动，
     计算「在给定备货覆盖天数下的断供概率」——这是概率建模。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C


# ------------------------------------------------------------- 供应商主数据
def build_suppliers(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """生成供应商主表：一家供应商一套属性（不是每个 SKU 一套）。"""
    regions = rng.choice(list(C.REGION_PROFILE), size=n,
                         p=[0.25, 0.35, 0.25, 0.15])
    lt_base = rng.uniform(7, 12, n)                     # 本地区域基准交期
    lead = np.array([lt_base[i] * C.REGION_PROFILE[regions[i]]["lead_mult"]
                     for i in range(n)]) * rng.lognormal(0, 0.10, n)
    geo = np.array([C.REGION_PROFILE[r]["geo_risk"] for r in regions])
    return pd.DataFrame({
        "supplier_id": ["SUP-%04d" % (i + 1) for i in range(n)],
        "region": regions,
        "lead_time_days": np.round(lead, 1),
        "otd_rate": np.round(rng.beta(C.OTD_ALPHA, C.OTD_BETA, n), 4),
        "dppm": np.round(np.exp(rng.normal(C.DPPM_MU, C.DPPM_SIGMA, n))).astype(int),
        "price_index": np.round(rng.normal(1.0, C.PRICE_INDEX_SD, n), 4),
        "financial_health": np.round(rng.beta(C.FIN_ALPHA, C.FIN_BETA, n), 4),
        "capacity_flex": np.round(rng.beta(C.FLEX_ALPHA, C.FLEX_BETA, n), 4),
        "response_days": np.round(rng.gamma(C.RESP_SHAPE, C.RESP_SCALE, n), 1),
        "geo_risk": geo,
    })


def map_items_to_suppliers(panel: pd.DataFrame, suppliers: pd.DataFrame,
                           rng: np.random.Generator) -> pd.DataFrame:
    """
    SKU -> 供应商供货关系。
    采购金额越大（ABC 类越靠前）越倾向双源；供应商选择按偏斜分布抽取，
    使少数供应商服务较多 SKU —— 这样集中度指标才有意义。
    """
    item = (panel.groupby("item_id")
                 .agg(total=("sales", "sum"), cat=("cat_id", "first"))
                 .reset_index()
                 .sort_values("total", ascending=False)
                 .reset_index(drop=True))
    item["abc_class"] = np.where(
        item["total"].cumsum() / item["total"].sum() <= 0.70, "A",
        np.where(item["total"].cumsum() / item["total"].sum() <= 0.90, "B", "C"))

    n_sup = len(suppliers)
    # 偏斜的供应商吸引力权重：少数供应商承接多数 SKU
    attract = rng.dirichlet(np.full(n_sup, C.SUPPLIER_SKEW))
    sid = suppliers["supplier_id"].to_numpy()

    rows = []
    for _, r in item.iterrows():
        k = 2 if r["abc_class"] in ("A", "B") else 1        # 战略/杠杆品双源，长尾单源
        chosen = rng.choice(n_sup, size=k, replace=False, p=attract)
        for rank, j in enumerate(chosen):
            rows.append({"item_id": r["item_id"], "supplier_id": sid[j],
                         "abc_class": r["abc_class"],
                         "is_primary": rank == 0, "n_sources": k,
                         "is_single_source": k == 1,
                         "item_total_sales": float(r["total"])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- QCDSM 评分
def qcdsm(suppliers: pd.DataFrame) -> pd.DataFrame:
    """五维加权评分（供应商层面）。每一维的映射函数与业务含义见下。"""
    m = suppliers.copy()

    # Quality：缺陷率跨数量级，用对数尺度（50 DPPM = 100 分；5000 DPPM = 60 分）
    m["Q"] = np.clip(100 - 20 * np.log10(np.maximum(m["dppm"], 1) / 50), 0, 100)

    # Cost：相对同类中位价，越便宜分越高（斜率 150 分/倍）
    med = m["price_index"].median()
    m["C"] = np.clip(100 - 150 * (m["price_index"] / med - 1), 0, 100)

    # Delivery：准时交付率直接映射
    m["D"] = np.clip(m["otd_rate"] * 100, 0, 100)

    # Service：响应速度（组内归一化，反向）+ 产能弹性
    r = m["response_days"]
    resp = np.clip(100 * (1 - (r - r.min()) / max(r.max() - r.min(), 1e-6)), 0, 100)
    m["S"] = np.clip(0.5 * resp + 0.5 * m["capacity_flex"] * 100, 0, 100)

    # Management：财务健康 + 地缘风险（反向）
    m["M"] = np.clip(0.7 * m["financial_health"] * 100 + 0.3 * (100 - m["geo_risk"] * 1.6), 0, 100)

    w = C.QCDSM_WEIGHTS
    m["qcdsm_score"] = (w["Quality"] * m["Q"] + w["Cost"] * m["C"] +
                        w["Delivery"] * m["D"] + w["Service"] * m["S"] +
                        w["Management"] * m["M"])
    m["qcdsm_grade"] = np.where(m["qcdsm_score"] >= C.GRADE_THRESHOLDS["低"], "低",
                               np.where(m["qcdsm_score"] >= C.GRADE_THRESHOLDS["中"], "中", "高"))
    return m


# ------------------------------------------------------------- 蒙特卡洛
def monte_carlo_stockout(daily_mean: float, sigma_d: float, lead_mean: float,
                         coverage_days: float, n_sim: int, rng: np.random.Generator):
    """
    模拟「交期 L 天内的累计需求」是否超出备货覆盖天数。

    L  ~ 对数正态，均值 = lead_mean，形状参数 = LEAD_TIME_LN_SIGMA
    D_L = 随机游走近似：L·d̄ + sqrt(L)·σ_d·Z   （Z ~ N(0,1)）
    缺口 = max(0, D_L - coverage_days·d̄)
    """
    if daily_mean <= 0:
        return 0.0, 0.0, 0.0
    sig = C.LEAD_TIME_LN_SIGMA
    mu = np.log(max(lead_mean, 1e-6)) - 0.5 * sig ** 2      # 使 E[L] = lead_mean
    L = rng.lognormal(mu, sig, n_sim)
    Z = rng.standard_normal(n_sim)
    D = np.maximum(L * daily_mean + np.sqrt(L) * sigma_d * Z, 0.0)
    shortage = np.maximum(0.0, D - coverage_days * daily_mean)
    return (float(np.mean(shortage > 0)),
            float(np.mean(shortage)),
            float(np.quantile(shortage, 0.95)))


def coverage_for_service(daily_mean: float, sigma_d: float, lead_mean: float,
                         target_sl: float, rng: np.random.Generator) -> float:
    """二分求达到目标服务水平所需的最少备货覆盖天数。"""
    lo, hi = 0.0, max(lead_mean * 3, 30.0)
    for _ in range(30):
        mid = (lo + hi) / 2
        p, _, _ = monte_carlo_stockout(daily_mean, sigma_d, lead_mean, mid,
                                       C.N_SIM_COVERAGE, rng)
        if p > (1 - target_sl):
            lo = mid
        else:
            hi = mid
    return hi


# ------------------------------------------------------------- 主流程
def run() -> pd.DataFrame:
    panel = pd.read_parquet(C.PROC_DIR / "panel.parquet")
    rng = np.random.default_rng(C.RANDOM_SEED)

    suppliers = build_suppliers(C.N_SUPPLIERS, rng)
    suppliers = qcdsm(suppliers)
    links = map_items_to_suppliers(panel, suppliers, rng)

    # σ_d 优先取回测残差（模型实测误差），否则退化为历史波动
    sigma_map = {}
    bt = C.OUT_DIR / "backtest_predictions.parquet"
    if bt.exists():
        b = pd.read_parquet(bt)
        b["resid"] = b["y"] - b["pred_lgbm_tweedie"]
        # M5 的 id 形如 FOODS_3_090_CA_1_evaluation -> 去掉末尾 3 段即 item_id
        b["item_id"] = b["id"].str.rsplit("_", n=3).str[0]
        sigma_map = b.groupby("item_id")["resid"].std().to_dict()
        print(f"[sigma] 使用回测残差标准差（{len(sigma_map):,} 个 SKU）")
    else:
        print("[sigma] 未找到回测结果，使用历史波动（建议先跑 src.forecast）")
    item_sigma = panel.groupby("item_id")["sales"].std().to_dict()

    # 需求统计（最近 182 天）+ 真实单价（M5 的 sell_prices），用于年化采购金额
    last_t = int(panel["t"].max())
    recent = panel[panel["t"] > last_t - 182]
    dem = (recent.groupby("item_id")
                .agg(daily_mean=("sales", "mean"), daily_std=("sales", "std"),
                     price=("price", "median")).reset_index())

    df = (links.merge(suppliers, on="supplier_id", how="left")
               .merge(dem, on="item_id", how="left"))
    df["daily_mean"] = df["daily_mean"].fillna(0.0)
    df["daily_std"] = df["daily_std"].fillna(0.0)
    df["price"] = df["price"].fillna(df["price"].median() if df["price"].notna().any() else 1.0)
    # 年化采购金额 = 日均需求 × 365 × 单价（用真实价格，不是随机系数）
    df["annual_spend"] = df["daily_mean"] * 365 * df["price"]
    df["sigma_d"] = [float(sigma_map.get(i, item_sigma.get(i, 1.0)))
                     for i in df["item_id"]]
    df["sigma_d"] = np.clip(df["sigma_d"], 0.05, None)

    out = []
    for _, r in df.iterrows():
        cov_now = r["lead_time_days"] + C.CURRENT_SAFETY_DAYS
        p_stock, exp_short, var95 = monte_carlo_stockout(
            r["daily_mean"], r["sigma_d"], r["lead_time_days"],
            coverage_days=cov_now, n_sim=C.N_SIM, rng=rng)
        need = coverage_for_service(r["daily_mean"], r["sigma_d"],
                                    r["lead_time_days"], C.SERVICE_LEVEL, rng)
        out.append({**r.to_dict(), "p_stockout": p_stock,
                    "exp_shortage": exp_short, "var95_shortage": var95,
                    "coverage_days_current": cov_now,
                    "coverage_days_needed": need})
    res = pd.DataFrame(out)

    # 综合风险分：供应商评分风险（反向）与断供概率各半，仅用于排序
    score_risk = np.clip(100 - res["qcdsm_score"], 0, 100)
    res["composite_risk"] = 0.5 * score_risk + 0.5 * res["p_stockout"] * 100

    # 风险等级按「断供概率 vs 目标缺货率」判定 —— 直接回答"这能不能保住 95% 服务"
    t = 1 - C.SERVICE_LEVEL
    res["risk_level"] = pd.cut(
        res["p_stockout"],
        [-.001, t, t * 3, t * 7, 1.001],
        labels=["低", "中", "高", "极高"])

    # 供应商分层：用于采购策略选择（优选 / 合格 / 观察 / 淘汰）
    res["supplier_tier"] = pd.cut(
        res["qcdsm_score"], [0, 65, 75, 85, 101],
        labels=["淘汰", "观察", "合格", "优选"])

    suppliers.to_csv(C.SUPPLIER_FILE, index=False)
    res.to_csv(C.RISK_FILE, index=False)
    print(f"[supplier] {len(suppliers)} 家供应商 | QCDSM 分层 "
          f"{suppliers['qcdsm_grade'].value_counts().to_dict()}")
    print(f"[links]    {len(res):,} 条 SKU-供应商关系 | 断供风险 "
          f"{res['risk_level'].value_counts().to_dict()} "
          f"（目标缺货率 {(1 - C.SERVICE_LEVEL):.0%}）")
    share = suppliers.merge(links.groupby("supplier_id")["item_total_sales"].sum().reset_index(),
                            on="supplier_id")
    share["w"] = share["item_total_sales"] / share["item_total_sales"].sum()
    print(f"[集中度]  HHI = {(share['w'] ** 2).sum():.4f} | 最大供应商承接 SKU 数 "
          f"{links.groupby('supplier_id').size().max()} | 单源 SKU {links['is_single_source'].sum()}")
    print(f"[output] {C.SUPPLIER_FILE}")
    print(f"[output] {C.RISK_FILE}")
    return res


if __name__ == "__main__":
    run()
