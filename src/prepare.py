"""
数据准备：把 M5 的宽表销量数据转换成「序列 × 日期」的长表，并对齐
日历（含节假日 / SNAP）与周度价格。

产出：data/processed/panel.parquet

设计要点
--------
1. M5 的宽表（每行一个 SKU-门店，列是 d_1..d_1941）无法直接做特征工程，
   必须先 melt 成长表；长表是所有时序特征的统一底座。
2. 日期来自 calendar.csv 的 d -> date 映射，不能靠位置推算，否则节假日
   与星期会对齐错位。
3. 价格是周粒度（wm_yr_wk），不能按天插值后再用，否则会产生"未来价格"泄漏。
   这里只在特征阶段按周回填，且默认使用「上一已知周」价格。
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
