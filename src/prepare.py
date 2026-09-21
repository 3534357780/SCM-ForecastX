"""
数据准备：把 M5 的宽表销量数据转换成「序列 × 日期」的长表，对齐日历与周度价格，
并完成一件事：**区分「没有需求」与「有需求但没货」**。

为什么必须做这一步
------------------
零售的日销量为 0 有两种成因。真的没有需求，和当天没货可卖。M5 里 53% 的日销量是 0，
其中一部分是断货造成的 —— 这类零是**被截断的需求**（censored demand），需求实际存在，
只是没有被记录。把它们原样当 0 送进模型，模型会学到「这个 SKU 就是卖不动」，
预测系统性偏低；偏低的预测又让下一轮备货更保守，缺货继续发生。

处理方式（两条规则，全部只用当日及以前的信息，因此可安全用于训练输入）：
  规则 1  促销日 + 零销量 —— 打折还卖不掉，几乎只可能是货没到架
  规则 2  零销量，但该序列过去 28 天零销占比低于 25% —— 平时天天有销量的 SKU 突然挂零
判定为缺货的日，用「过去 56 天非促销、非零销日的中位数」还原。用中位数而非均值，
是为了消除系统性低估而不是放大需求。

规则的取舍由数据决定，不靠直觉（见 README §3.1 的规则有效性检验）
----------------------------------------------------------------
曾考虑再加一条「连续零销 ≥ 3 天」。但把它单独拉出来检验后发现：被这条规则判定的日子，
其后 7 天的销量只恢复到该序列常态水平的 28%，明显低于「零销但未判缺货」的 70%。
也就是说长串零更可能是 SKU 进入停售 / 衰退，而不是断货 —— 把它当断货还原，
等于把「停售」误读成「需求」。因此这条规则被舍弃，只用证据支持的两条。

同时标记促销日（价格低于过去 182 天常态价 2% 以上）。促销日单独标记的作用有二：
一是参与缺货判定，二是让下游把「促销波动」与「基础波动」分开 —— 见 src/risk.py。

产出：data/processed/panel.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C


def _find(raw: Path, *names: str) -> Path:
    """在原始目录里找一个存在的文件（兼容 Kaggle 压缩包里的嵌套目录）。"""
    for n in names:
        p = raw / n
        if p.exists():
            return p
    for p in raw.rglob("*.csv"):
        if p.name in names:
            return p
    raise FileNotFoundError(f"{raw} 下找不到 {names}，请先按 README 下载 M5 数据")


def load_sales() -> pd.DataFrame:
    """读入 M5 销量宽表并按配置裁剪范围。"""
    f = _find(C.RAW_DIR, "sales_train_evaluation.csv", "sales_train_validation.csv")
    head = pd.read_csv(f, nrows=0)
    meta = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    dcols = [c for c in head.columns if c.startswith("d_")]

    df = pd.read_csv(f, usecols=meta + dcols, dtype={c: "int32" for c in dcols})

    n0 = len(df)
    if C.STATE:
        df = df[df.state_id == C.STATE]
    if C.STORE:
        df = df[df.store_id == C.STORE]
    if C.DEPT:
        df = df[df.dept_id == C.DEPT]
    df = df.reset_index(drop=True)
    print(f"[sales] 全量 {n0:,} 条序列 -> 裁剪后 {len(df):,} 条"
          f" (state={C.STATE}, store={C.STORE}, dept={C.DEPT})")
    if df.empty:
        raise ValueError("裁剪后为空，请检查 config 里的 STATE/STORE/DEPT")
    return df, dcols


def load_calendar() -> pd.DataFrame:
    f = _find(C.RAW_DIR, "calendar.csv")
    cal = pd.read_csv(f, parse_dates=["date"])
    keep = ["d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
            "event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    for c in ["snap_CA", "snap_TX", "snap_WI"]:
        if c in cal.columns:
            keep.append(c)
    cal = cal[[c for c in keep if c in cal.columns]].copy()
    # 事件标签清洗：没有事件的行统一标为 "无"
    for c in ["event_name_1", "event_name_2"]:
        if c in cal.columns:
            cal[c] = cal[c].fillna("无")
    return cal


def load_prices() -> pd.DataFrame | None:
    try:
        f = _find(C.RAW_DIR, "sell_prices.csv")
    except FileNotFoundError:
        print("[prices] 未找到 sell_prices.csv，跳过价格特征（金额口径指标将不可用）")
        return None
    pr = pd.read_csv(f)
    print(f"[prices] {len(pr):,} 行周度价格")
    return pr


def mark_promo(panel: pd.DataFrame) -> pd.DataFrame:
    """
    促销日标记：当日价格低于「过去 28 天常态价」5% 以上。

    常态价用**过去** 28 天的中位数（先 shift(1) 再 rolling），不用当日及以后的价格 ——
    这样标记本身不含未来信息，可以安全地进训练。
    """
    g = panel.groupby("id", sort=False)["price"]
    med = g.transform(lambda s: s.shift(1).rolling(
        C.PROMO_PRICE_WIN, min_periods=C.PROMO_PRICE_MINP).median())
    panel["price_median_28"] = med
    panel["is_promo"] = (
        panel["price"].notna() & med.notna()
        & (panel["price"] < med * (1 - C.PROMO_PRICE_DROP))
    ).astype("int8")
    return panel


def detect_and_restore(panel: pd.DataFrame) -> pd.DataFrame:
    """
    缺货日识别与需求还原。两条规则的含义与取舍见模块 docstring。
    所有中间量都只依赖当日及以前的数据，因此还原后的序列可以合法用作训练目标。

    zero_run 仍会算出来，但只作为诊断列（用于检验「长串零」是否等于断货），不参与判定。
    """
    g = panel.groupby("id", sort=False)

    # 过去 28 天的零销占比（只用历史）
    panel["zero_ratio_28"] = g["sales"].transform(
        lambda s: (s == 0).astype(float).shift(1).rolling(28, min_periods=14).mean())
    # 过去 56 天「非促销且非零销」日的中位数，作为该序列的正常日销量
    normal_src = panel["sales"].where((panel["is_promo"] == 0) & (panel["sales"] > 0))
    panel["normal_level"] = normal_src.groupby(panel["id"]).transform(
        lambda s: s.shift(1).rolling(C.RESTORE_WIN, min_periods=C.RESTORE_MINP).median())

    # 连续零销天数（按序列分块累计）—— 仅用于诊断
    z = (panel["sales"] == 0).astype("int32")
    blk = (z == 0).groupby(panel["id"]).cumsum()
    panel["zero_run"] = z.groupby([panel["id"], blk]).cumsum()

    isnan = panel["normal_level"].isna()
    r1 = (panel["is_promo"] == 1) & (panel["sales"] == 0)
    r2 = (panel["sales"] == 0) & (panel["zero_ratio_28"] < C.STOCKOUT_TYPICAL_ZERO)
    panel["is_stockout"] = ((r1 | r2) & ~isnan).astype("int8")

    panel["sales_restored"] = np.where(panel["is_stockout"] == 1,
                                       panel["normal_level"], panel["sales"])
    panel["sales_restored"] = panel["sales_restored"].astype("float32")

    print(f"[信号] 促销日占比 {100 * panel['is_promo'].mean():.1f}% | "
          f"疑似缺货日占比 {100 * panel['is_stockout'].mean():.1f}%"
          f"（促销零销 {int(r1.sum()):,} 天 / 热销挂零 {int((r2 & ~r1).sum()):,} 天）")
    print(f"[信号] 还原前后日均销量变化 "
          f"{panel['sales_restored'].mean() - panel['sales'].mean():+.3f} 件"
          f"（{100 * (panel['sales_restored'].mean() / max(panel['sales'].mean(), 1e-9) - 1):+.1f}%）")
    return panel


def build_panel() -> pd.DataFrame:
    sales, dcols = load_sales()
    cal = load_calendar()

    # 宽 -> 长
    panel = sales.melt(
        id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        value_vars=dcols, var_name="d", value_name="sales",
    )
    print(f"[melt] 长表 {len(panel):,} 行")

    panel = panel.merge(cal, on="d", how="left")
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["id", "date"]).reset_index(drop=True)
    panel["t"] = panel.groupby("id").cumcount()          # 每个序列的内部时间序号

    pr = load_prices()
    if pr is not None:
        panel = panel.merge(
            pr.rename(columns={"sell_price": "price"}),
            on=["store_id", "item_id", "wm_yr_wk"], how="left",
        )
        # 价格是周粒度：组内前向填充，只补历史已有价格，绝不回填未来
        panel["price"] = panel.groupby("id")["price"].ffill()
    else:
        panel["price"] = np.nan

    panel = mark_promo(panel)
    panel = detect_and_restore(panel)

    C.PROC_DIR.mkdir(parents=True, exist_ok=True)
    out = C.PROC_DIR / "panel.parquet"
    panel.to_parquet(out, index=False)

    span = (panel["date"].min().date(), panel["date"].max().date())
    print(f"[panel] 序列 {panel['id'].nunique():,} 条 | 日期跨度 {span[0]} ~ {span[1]}"
          f" | 零销量占比 {100 * (panel['sales'] == 0).mean():.1f}%")
    print(f"[panel] 已写出 {out}")
    return panel


if __name__ == "__main__":
    build_panel()
