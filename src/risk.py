"""
供应侧：供应商记分卡 + 计划侧的风险调整输出。

一个先要分清的问题：供应商风险到底用什么衡量
------------------------------------------
企业里评价供应商用的是**多维记分卡**，不是单一的「断供概率」。采购部门实际看的指标
大致固定：交付（OTD 准时交付率、OTIF 准时足量率、交期与交期波动）、质量（DPPM）、
产能弹性（急单可加量、响应天数）、财务健康、连续性（恢复时间 TTR 与可支撑时间 TTS）、
以及品类层面的单源暴露与区域集中度。

「缺货概率」不是供应商评价指标，它是**计划侧的输出** —— 由交期波动、需求波动和
现行备货政策三者共同推出来的结果。把结果当成对供应商的评价，会把供应商自身的
能力（能不能按时按量交货）和我们的政策（备了多少货）混为一谈。

所以本模块输出两张表，口径分开：
  supplier_scorecard.csv  供应商层面 —— 六维打分 + 分层，用于选择、考核与谈判
  supplier_risk.csv       供货关系层面 —— 风险调整交期、覆盖缺口、现行政策下的缺货概率，
                          用于定安全库存与再订货点

数据来源与边界
--------------
企业里这些字段来自 SRM / ERP 的采购主数据导出。本项目用参数化数据源替代该输入层：
字段名与采购记分卡口径一致，分布参数集中在 config.py 并标注来源与不确定性，
随机种子固定。接入企业数据时替换数据源，本模块以下的决策逻辑不变。

分布选择的两条硬约束
--------------------
  比例类指标（OTD、OTIF、产能弹性、财务健康）用 Beta 分布 —— 取值天然落在 0~1；
  时长类指标（交期、恢复时间）用对数正态 —— 正值且右偏，避免抽出负交期。
"""
from __future__ import annotations

import sys
from statistics import NormalDist
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C


# ------------------------------------------------------------- 成本 -> 服务水平
def service_level_from_costs(price, abc_class, k_override=None, holding_rate=None):
    """
    目标服务水平由成本不对称推导，不拍脑袋。

        SL = Cu / (Cu + Co)
        Cu = 单位毛利 × 缺货成本倍数（倍数 > 1 表示计入复购与口碑损失）
        Co = 单位成本 × 年持有成本率

    Co 取**年化**持有成本而不是按交期折算：安全库存是一年 365 天都压在库里的常备水位，
    它的持有成本就是整年的。若按「交期/365」折算，Co 会小到可以忽略（0.1% 量级），
    导出的服务水平会全部顶到 99% 以上，分档失去意义。

    A 类缺货对复购的影响最直接，倍数取 3；C 类长尾取 1。这使服务水平按 ABC 自动分档，
    不需要再人为设一套分档规则。

    k_override / holding_rate 供敏感性扫描复用同一套公式，避免扫描表与主口径不一致。
    """
    if k_override is not None:
        k = float(k_override)
    else:
        abc = np.asarray(abc_class)
        k = np.array([C.STOCKOUT_MULT_BY_ABC.get(str(a), 1.0)
                      for a in np.ravel(abc)]).reshape(abc.shape)
    hr = C.HOLDING_RATE_ANNUAL if holding_rate is None else float(holding_rate)
    cu = np.asarray(price, dtype=float) * C.GROSS_MARGIN_RATE * k
    co = np.asarray(price, dtype=float) * (1 - C.GROSS_MARGIN_RATE) * hr
    return np.clip(cu / np.maximum(cu + co, 1e-12), 0.50, 0.999)


def z_from_service_level(sl):
    """由服务水平反查正态分位。落在表内用表（便于与教材口径对照），否则精确计算。"""
    s = np.asarray(sl, dtype=float)
    nd = NormalDist()
    out = np.empty_like(s)
    for i, v in np.ndenumerate(s):
        key = round(float(v), 3)
        out[i] = C.Z_TABLE[key] if key in C.Z_TABLE else nd.inv_cdf(float(v))
    return out


# ------------------------------------------------------------- 供应商主数据
def build_suppliers(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """生成供应商主表：一家供应商一套属性（不是每个 SKU 一套）。"""
    regions = rng.choice(list(C.REGION_PROFILE), size=n, p=[0.25, 0.35, 0.25, 0.15])
    lt_base = rng.uniform(7, 12, n)                     # 本地区域基准交期
    lead = np.array([lt_base[i] * C.REGION_PROFILE[regions[i]]["lead_mult"]
                     for i in range(n)]) * rng.lognormal(0, 0.10, n)
    otd = rng.beta(C.OTD_ALPHA, C.OTD_BETA, n)
    return pd.DataFrame({
        "supplier_id": ["SUP-%04d" % (i + 1) for i in range(n)],
        "region": regions,
        "lead_time_days": np.round(lead, 1),
        "lead_time_cv": np.round(C.LEAD_TIME_LN_SIGMA, 4),
        "otd_rate": np.round(otd, 4),                              # 准时交付率
        "otif_rate": np.round(np.clip(otd * C.OTIF_OTD_RATIO, 0, 1), 4),  # 准时足量率
        "dppm": np.round(np.exp(rng.normal(C.DPPM_MU, C.DPPM_SIGMA, n))).astype(int),
        "capacity_flex": np.round(rng.beta(C.FLEX_ALPHA, C.FLEX_BETA, n), 4),
        "response_days": np.round(rng.gamma(C.RESP_SHAPE, C.RESP_SCALE, n), 1),
        "financial_health": np.round(rng.beta(C.FIN_ALPHA, C.FIN_BETA, n), 4),
        "ttr_days": np.round(rng.lognormal(C.TTR_MU, C.TTR_SIGMA, n), 1),
        "price_index": np.round(rng.normal(1.0, C.PRICE_INDEX_SD, n), 4),
        "geo_risk": np.array([C.REGION_PROFILE[r]["geo_risk"] for r in regions]),
        "region_disrupt_p": np.array(
            [C.REGION_PROFILE[r]["region_disrupt_p"] for r in regions]),
    })


def map_items_to_suppliers(panel: pd.DataFrame, suppliers: pd.DataFrame,
                           rng: np.random.Generator) -> pd.DataFrame:
    """
    SKU -> 供应商供货关系。

    A/B 类 SKU 双源并带主备份额：主供承接大部分需求，备供保持小份额持续下单。
    这是采购实务的通行做法 —— 备供若长期没有订单，产能响应和商务关系都会退化，
    真到切换时接不住。主供份额取 0.65~0.85。
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
    attract = rng.dirichlet(np.full(n_sup, C.SUPPLIER_SKEW))   # 少数供应商承接多数 SKU
    sid = suppliers["supplier_id"].to_numpy()

    rows = []
    for _, r in item.iterrows():
        k = 2 if r["abc_class"] in ("A", "B") else 1
        chosen = rng.choice(n_sup, size=k, replace=False, p=attract)
        split = 1.0 if k == 1 else rng.uniform(0.65, 0.85)
        for rank, j in enumerate(chosen):
            rows.append({"item_id": r["item_id"], "supplier_id": sid[j],
                         "abc_class": r["abc_class"],
                         "is_primary": rank == 0, "n_sources": k,
                         "is_single_source": k == 1,
                         "split_ratio": split if rank == 0 else 1.0 - split,
                         "item_total_sales": float(r["total"])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- 六维记分卡
def scorecard(suppliers: pd.DataFrame) -> pd.DataFrame:
    """
    六维打分（0~100），权重见 config.SCORECARD_WEIGHTS。每一维的映射函数与业务含义：

      Delivery    准时交付率与准时足量率（到了但数量不足同样造成缺货）
      Quality     缺陷率 DPPM，跨数量级故用对数尺度（50 DPPM 满分，5000 DPPM 记 60 分）
      Cost        相对同类中位价
      Flexibility 急单响应天数 + 产能弹性
      Financial   财务健康度
      Continuity  恢复时间 TTR 与可支撑时间 TTS 的缺口 + 区域中断概率
    """
    m = suppliers.copy()

    m["D"] = np.clip(100 * (0.6 * m["otd_rate"] + 0.4 * m["otif_rate"]), 0, 100)
    m["Q"] = np.clip(100 - 20 * np.log10(np.maximum(m["dppm"], 1) / 50), 0, 100)
    med = m["price_index"].median()
    m["C"] = np.clip(100 - 150 * (m["price_index"] / med - 1), 0, 100)

    r = m["response_days"]
    resp = np.clip(100 * (1 - (r - r.min()) / max(r.max() - r.min(), 1e-6)), 0, 100)
    m["F"] = np.clip(0.5 * resp + 0.5 * m["capacity_flex"] * 100, 0, 100)

    m["Fin"] = np.clip(m["financial_health"] * 100, 0, 100)

    # 连续性：现行政策下能撑住的天数 vs 中断后恢复需要的天数，缺口越大分越低
    m["tts_days"] = np.round(m["lead_time_days"] + C.CURRENT_SAFETY_DAYS, 1)
    m["continuity_gap_days"] = np.round(m["ttr_days"] - m["tts_days"], 1)
    gap_score = np.clip(100 * (1 - m["continuity_gap_days"] / np.maximum(m["ttr_days"], 1)),
                        0, 100)
    region_score = np.clip(100 * (1 - m["region_disrupt_p"] / 0.10), 0, 100)
    m["Cont"] = np.clip(0.6 * gap_score + 0.4 * region_score, 0, 100)

    w = C.SCORECARD_WEIGHTS
    m["supplier_score"] = (w["Delivery"] * m["D"] + w["Quality"] * m["Q"] +
                           w["Cost"] * m["C"] + w["Flexibility"] * m["F"] +
                           w["Financial"] * m["Fin"] + w["Continuity"] * m["Cont"])
    m["tier"] = np.where(m["supplier_score"] >= C.SCORE_TIERS["优选"], "优选",
                         np.where(m["supplier_score"] >= C.SCORE_TIERS["合格"], "合格",
                                  np.where(m["supplier_score"] >= C.SCORE_TIERS["观察"],
                                           "观察", "淘汰")))
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


# ------------------------------------------------------------- 需求波动分解
def demand_stats(panel: pd.DataFrame, bt_path: Path) -> pd.DataFrame:
    """
    SKU 级需求统计，含**波动分解**。

    促销期的波动是一次性、而且有促销日历可预见的；基础波动是持续存在的。
    如果两者混在一个 σ 里，等于按促销月的剧烈波动给全年备货 —— 安全库存被系统性抬高。
    这里把残差标准差拆成三列：
        sigma_d_all    全样本（旧口径，保留用于对比）
        sigma_d_base   非促销日（安全库存用这一列）
        sigma_promo    促销日（单独看，用于评估促销期的预测难度）
    同时给出促销抬升幅度与促销频率，供计划侧单独做促销预建。
    """
    last_t = int(panel["t"].max())
    recent = panel[panel["t"] > last_t - 182]

    stat = (recent.groupby("item_id")
                  .agg(daily_mean=("sales", "mean"), daily_std=("sales", "std"),
                       price=("price", "median")).reset_index())

    # 促销/基础需求水平与频率
    base = recent[recent["is_promo"] == 0]
    promo = recent[recent["is_promo"] == 1]
    bl = base.groupby("item_id")["sales_restored"].mean().rename("base_level")
    pl = promo.groupby("item_id")["sales_restored"].mean().rename("promo_level")
    pf = (recent.groupby("item_id")["is_promo"].mean().rename("promo_freq"))
    stat = stat.merge(bl, on="item_id", how="left").merge(pl, on="item_id", how="left") \
               .merge(pf, on="item_id", how="left")
    stat["base_level"] = stat["base_level"].fillna(stat["daily_mean"])
    stat["promo_level"] = stat["promo_level"].fillna(stat["base_level"])
    stat["promo_freq"] = stat["promo_freq"].fillna(0.0)
    stat["promo_uplift"] = np.clip(
        stat["promo_level"] / np.maximum(stat["base_level"], 1e-6) - 1, 0, None)

    # 残差标准差：全样本 / 非促销 / 促销
    if bt_path.exists():
        b = pd.read_parquet(bt_path)
        b["resid"] = b["y"] - b["pred_lgbm_tweedie"]
        b["item_id"] = b["id"].str.rsplit("_", n=3).str[0]
        g = b.groupby("item_id")["resid"]
        s_all = g.std().rename("sigma_d_all")
        s_base = b[b["is_promo"] == 0].groupby("item_id")["resid"].std().rename("sigma_d_base")
        s_pro = b[b["is_promo"] == 1].groupby("item_id")["resid"].std().rename("sigma_d_promo")
        n_base = b[b["is_promo"] == 0].groupby("item_id")["resid"].size().rename("n_base")
        stat = (stat.merge(s_all, on="item_id", how="left")
                    .merge(s_base, on="item_id", how="left")
                    .merge(s_pro, on="item_id", how="left")
                    .merge(n_base, on="item_id", how="left"))
        print(f"[sigma] 用回测残差分解波动（{stat['sigma_d_all'].notna().sum():,} 个 SKU）")
    else:
        stat["sigma_d_all"] = recent.groupby("item_id")["sales"].std().reindex(
            stat["item_id"]).to_numpy()
        stat["sigma_d_base"] = stat["sigma_d_all"]
        stat["sigma_d_promo"] = np.nan
        stat["n_base"] = np.inf
        print("[sigma] 未找到回测结果，退回历史波动（建议先跑 src.forecast）")

    # 非促销样本不足的 SKU 退回全样本口径，避免用 2~3 个点估出一个假 σ
    weak = stat["n_base"].fillna(0) < C.MIN_BASE_DAYS
    stat.loc[weak, "sigma_d_base"] = stat.loc[weak, "sigma_d_all"]
    stat["sigma_d_used"] = stat["sigma_d_base"].fillna(stat["sigma_d_all"]).fillna(0.0)
    stat["sigma_d_used"] = np.clip(stat["sigma_d_used"], C.SIGMA_FLOOR, None)
    stat["sigma_d_all"] = np.clip(stat["sigma_d_all"].fillna(stat["sigma_d_used"]),
                                  C.SIGMA_FLOOR, None)
    return stat


# ------------------------------------------------------------- 主流程
def run() -> pd.DataFrame:
    panel = pd.read_parquet(C.PANEL_FILE)
    rng = np.random.default_rng(C.RANDOM_SEED)

    suppliers = build_suppliers(C.N_SUPPLIERS, rng)
    suppliers = scorecard(suppliers)
    links = map_items_to_suppliers(panel, suppliers, rng)
    stat = demand_stats(panel, C.OUT_DIR / "backtest_predictions.parquet")

    df = (links.merge(suppliers, on="supplier_id", how="left")
               .merge(stat, on="item_id", how="left"))
    df["daily_mean"] = df["daily_mean"].fillna(0.0)
    df["price"] = df["price"].fillna(df["price"].median() if df["price"].notna().any() else 1.0)
    # 年化采购金额 = 日均需求 × 365 × 单价 × 该关系的份额
    df["annual_spend"] = df["daily_mean"] * 365 * df["price"] * df["split_ratio"]

    # 目标服务水平：由成本比推导，按 ABC 分档
    df["service_level_target"] = service_level_from_costs(
        df["price"].to_numpy(), df["abc_class"].to_numpy())
    df["z_value"] = z_from_service_level(df["service_level_target"])

    # 计划侧输出：现行政策下的缺货概率 + 达到目标服务水平所需的覆盖天数
    out = []
    for _, r in df.iterrows():
        cov_now = r["lead_time_days"] + C.CURRENT_SAFETY_DAYS
        p_short, exp_short, var95 = monte_carlo_stockout(
            r["daily_mean"], r["sigma_d_used"], r["lead_time_days"],
            coverage_days=cov_now, n_sim=C.N_SIM, rng=rng)
        need = coverage_for_service(r["daily_mean"], r["sigma_d_used"],
                                    r["lead_time_days"],
                                    float(r["service_level_target"]), rng)
        out.append({**r.to_dict(), "p_shortage": p_short,
                    "exp_shortage": exp_short, "var95_shortage": var95,
                    "coverage_days_current": cov_now,
                    "coverage_days_needed": need})
    res = pd.DataFrame(out)

    # 风险等级：不按缺货概率的相对倍数（那个口径下几乎所有 SKU 都落在同一档），
    # 而按**覆盖缺口天数** = 达到目标服务水平需要的覆盖天数 - 现行政策的覆盖天数。
    # 单位是天，供应链和管理层都能直接读懂：正数表示现行备货撑不到目标水平。
    res["coverage_gap_days"] = np.round(
        res["coverage_days_needed"] - res["coverage_days_current"], 1)
    res["risk_level"] = pd.cut(res["coverage_gap_days"], [-np.inf, 0, 7, 15, np.inf],
                               labels=["低", "中", "高", "极高"])
    # 综合风险分仅用于排序：供应商能力（记分卡）与我们的政策缺口各半
    res["composite_risk"] = 0.5 * np.clip(100 - res["supplier_score"], 0, 100) \
                            + 0.5 * np.clip(res["coverage_gap_days"] / 25 * 100, 0, 100)

    # 促销预建：促销期的额外需求单独备，不混进安全库存
    res["promo_prebuild_units"] = (res["base_level"] * res["promo_uplift"]
                                   * res["promo_freq"] * C.HORIZON * res["split_ratio"])

    # SKU 级联合缺货概率（双源 SKU 主备同时缺货）。独立性是近似 ——
    # 同区域供应商正相关，独立假设会低估联合概率；此处保留为已声明的边界。
    joint = res.groupby("item_id")["p_shortage"].agg(
        p_shortage_sku=lambda p: float(np.prod(p)))
    res = res.merge(joint, on="item_id", how="left")

    suppliers.to_csv(C.SUPPLIER_FILE, index=False)
    res.to_csv(C.RISK_FILE, index=False)

    print(f"[supplier] {len(suppliers)} 家供应商 | 分层 "
          f"{suppliers['tier'].value_counts().to_dict()}")
    print(f"[links]    {len(res):,} 条 SKU-供应商关系 | 风险等级（按覆盖缺口天数）"
          f"{res['risk_level'].value_counts().to_dict()}")
    print(f"[policy]   目标服务水平 {res['service_level_target'].min():.1%}"
          f"~{res['service_level_target'].max():.1%}"
          f"（由成本比推导，按 ABC 分档；原口径为全局 {0.95:.0%}）")
    sk = res.drop_duplicates("item_id")
    for a in ["A", "B", "C"]:
        s = sk[sk["abc_class"] == a]
        print(f"[缺口]     {a} 类 {len(s):>3} 个 SKU | 目标缺货率 "
              f"{(1 - s['service_level_target']).mean():>5.1%} | 现行政策缺货概率 "
              f"{s['p_shortage'].mean():>5.1%} | 覆盖缺口 "
              f"{s['coverage_gap_days'].mean():+.1f} 天"
              f"（{'欠备' if s['coverage_gap_days'].mean() > 0 else '过备'}）")
    s1, s2 = sk[sk["is_single_source"]], sk[~sk["is_single_source"]]
    print(f"[双源]     单源 {len(s1)} 个 SKU 现行缺货概率 {s1['p_shortage_sku'].mean():.1%}"
          f" | 双源 {len(s2)} 个 SKU 联合 {s2['p_shortage_sku'].mean():.1%}")
    share = suppliers.merge(links.groupby("supplier_id")["item_total_sales"].sum().reset_index(),
                            on="supplier_id")
    share["w"] = share["item_total_sales"] / share["item_total_sales"].sum()
    print(f"[集中度]  HHI = {(share['w'] ** 2).sum():.4f} | 最大供应商承接 SKU 数 "
          f"{links.groupby('supplier_id').size().max()} | 单源 SKU {links['is_single_source'].sum()}")
    gap = suppliers[suppliers["continuity_gap_days"] > 0]
    print(f"[连续性]  {len(gap)}/{len(suppliers)} 家供应商的恢复时间超过现行库存可支撑天数"
          f"（TTR > TTS），平均缺口 {suppliers['continuity_gap_days'].clip(lower=0).mean():.1f} 天")
    print(f"[output] {C.SUPPLIER_FILE}")
    print(f"[output] {C.RISK_FILE}")
    return res


if __name__ == "__main__":
    run()
