"""
需求预测与滚动原点回测（rolling-origin backtest）。

严谨性设计
----------
1. **按时间切分，不按行号切分。** 序列若按 SKU 顺序拼接后按行号切分，
   切出的是「一部分 SKU 训练 / 另一部分 SKU 验证」，训练集与验证集的
   日期区间完全重叠，验证集根本不是「未来」。
   本项目按预测原点（forecast origin）切分：训练只用原点之前的数据。
2. **特征不含未来信息。** 所有滞后与滚动统计都以原点为右端点；季节性滞后
   取「原点及以前最后一个同星期值」，而不是 t-7（t-7 在 h>7 时仍在未来）。
3. **强制 baseline 对照。** 没有 baseline 的预测项目无法证明模型有价值。
   固定跑 7 个 baseline：末值外推、季节性 naive、28 日均值、drift、
   Holt、Croston、SBA（后两个是间歇性需求的经典方法）。
4. **多指标而不是单一 MAE。** WMAPE 规避零销量的分母问题；MASE 以季节性
   naive 为标尺（<1 才算赢过朴素方法）；Bias% 暴露系统性高估/低估。
5. **预测区间由实测残差给出**，不用模型自报的置信度，供后续安全库存使用。

用法
----
    python -m src.forecast              # 全流程
    python -m src.forecast --no-lgbm    # 只跑 baseline（快速验证）
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

FEATURES = [
    "h",
    "s_l1", "s_l2", "s_l3", "s_l4", "s_l5", "s_l6", "s_l7",
    "rm7", "rm14", "rm28", "rs28", "zero_ratio_28", "trend_7_28",
    "wk_lag",
    "price_lag", "price_ratio",
    "dow", "is_weekend", "month", "day_of_month",
    "snap", "has_event", "event_type",
    "series_id",
]
CATEGORICAL = ["series_id", "dow", "month", "event_type"]
MODELS = ["naive_last", "seasonal_naive", "ma28",
          "drift", "holt", "croston", "sba",
          "lgbm_tweedie"]
BASELINES = ["naive_last", "seasonal_naive", "ma28",
             "drift", "holt", "croston", "sba"]


# ------------------------------------------------------------------ 数据底座
def to_matrices(panel: pd.DataFrame):
    """把长表转成矩阵，便于按时间切片做特征。"""
    ids = np.sort(panel["id"].unique())
    idx = {v: i for i, v in enumerate(ids)}

    pivot_s = panel.pivot_table(index="id", columns="t", values="sales", aggfunc="sum")
    pivot_p = panel.pivot_table(index="id", columns="t", values="price", aggfunc="last")
    pivot_s = pivot_s.reindex(index=ids)
    pivot_p = pivot_p.reindex(index=ids)
    S = pivot_s.to_numpy(dtype=np.float32)
    P = pivot_p.to_numpy(dtype=np.float32)

    cal = (panel.sort_values("t")
                .drop_duplicates("t")
                .set_index("t")[["date", "weekday", "month", "snap_CA",
                                 "event_name_1", "event_type_1"]])
    dow_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
               "Friday": 4, "Saturday": 5, "Sunday": 6}
    dow = cal["weekday"].map(dow_map).to_numpy()
    month = cal["month"].to_numpy()
    snap = (cal["snap_CA"].fillna(0).to_numpy() if "snap_CA" in cal
            else np.zeros(len(cal)))
    evt = cal["event_type_1"].fillna("无")
    evt_categories = ["无", "Cultural", "National", "Religious", "Sporting"]
    evt_code = pd.Categorical(evt, categories=evt_categories).codes
    has_event = (evt != "无").astype(int).to_numpy()
    return ids, np.arange(len(ids)), S, P, {
        "dow": dow, "month": month, "snap": snap.astype(int),
        "evt_code": evt_code, "has_event": has_event,
        "day_of_month": pd.to_datetime(cal["date"]).dt.day.to_numpy(),
        "date": pd.to_datetime(cal["date"]).to_numpy(),
    }


def build_rows(S, P, cal, series_idx, origin: int, horizon: int) -> pd.DataFrame:
    """构造「原点 origin 出发、预测未来 horizon 天」的全部特征行，无未来信息。"""
    n = S.shape[0]
    o = origin
    feats = {"h": np.full(n, horizon, dtype=np.int16)}
    for k in range(1, 8):
        feats[f"s_l{k}"] = S[:, o - k + 1]
    win7 = S[:, o - 6:o + 1]
    win14 = S[:, o - 13:o + 1]
    win28 = S[:, o - 27:o + 1]
    feats["rm7"] = win7.mean(axis=1)
    feats["rm14"] = win14.mean(axis=1)
    feats["rm28"] = win28.mean(axis=1)
    feats["rs28"] = win28.std(axis=1)
    feats["zero_ratio_28"] = (win28 == 0).mean(axis=1)
    feats["trend_7_28"] = (np.where(feats["rm28"] > 0,
                                    feats["rm7"] / np.maximum(feats["rm28"], 1e-6), 1.0)
                           - 1.0)

    # 季节性滞后：原点及以前「最后一个同星期」的值
    back = (7 - horizon % 7) % 7
    feats["wk_lag"] = S[:, o - back]

    # 价格：只用历史已知价（保守选择）
    price_o = np.nan_to_num(P[:, o], nan=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # 部分序列整段无价，忽略空切片告警
        price_m = np.nanmean(P[:, o - 27:o + 1], axis=1)
    price_m = np.nan_to_num(price_m, nan=0.0)
    feats["price_lag"] = price_o
    feats["price_ratio"] = np.where(price_m > 0, price_o / np.maximum(price_m, 1e-6), 1.0)

    # 目标日的日历特征（预测时已知，不构成泄漏）
    t = o + horizon
    feats["dow"] = cal["dow"][t]
    feats["is_weekend"] = (cal["dow"][t] >= 5).astype(np.int8)
    feats["month"] = cal["month"][t]
    feats["day_of_month"] = cal["day_of_month"][t]
    feats["snap"] = cal["snap"][t]
    feats["has_event"] = cal["has_event"][t]
    feats["event_type"] = cal["evt_code"][t]
    feats["series_id"] = series_idx

    df = pd.DataFrame(feats)
    df["y"] = S[:, t]
    df["t"] = t
    df["origin"] = o
    df["date"] = cal["date"][t]
    return df


def baselines(S, origin: int, horizon: int) -> dict:
    """基线预测值（全部只用原点及以前的信息；每条返回长度=n 的当前 horizon 预测）。

    七类基线，分四档含义：
      - 零智识组：naive_last / ma28 —— 完全靠历史统计
      - 季节组：seasonal_naive —— 同星期值
      - 趋势/平滑组：drift / holt —— 含线性外推与双指数平滑
      - 间歇组：croston / sba —— **专门为零销占比高的间歇需求设计**
    """
    o = origin
    back = (7 - horizon % 7) % 7
    out = {
        "naive_last":     np.maximum(S[:, o], 0),
        "seasonal_naive": np.maximum(S[:, o - back], 0),
        "ma28":           np.maximum(S[:, o - 27:o + 1].mean(axis=1), 0),
        "drift":          _drift(S, o, horizon)[:, horizon - 1],
        "holt":           _holt(S, o, horizon)[:, horizon - 1],
        "croston":        _croston(S, o),
        "sba":            _sba(S, o),
    }
    return out


def _drift(S, o, horizon):
    """Naive + 线性漂移：最后一点 + (h/T) × (S[o] - S[o-T+1])，T=28。
    M3/M4 竞赛标配基线，比 naive 略强。返回 (n, H) 矩阵，由 caller 切片取当前 h。"""
    T = 28
    last = S[:, o]
    first = S[:, o - T + 1]
    slope = (last - first) / T
    h_arr = np.arange(1, horizon + 1)
    pred = last[:, None] + h_arr[None, :] * slope[:, None]   # (n, H)
    return np.maximum(pred, 0)


def _holt(S, o, horizon):
    """Holt 双参数指数平滑（水平 + 趋势），α=β=0.1，固定以保稳定。
    拟合阶段只用 28 天窗口；预测阶段给出 ℓ_T + h·b_T，h=1..H。"""
    α, β = 0.1, 0.1
    win = S[:, o - 27:o + 1]
    n = win.shape[1]
    lvl = win[:, 0]
    b = win[:, 1] - win[:, 0]
    for t in range(1, n):
        new_lvl = α * win[:, t] + (1 - α) * (lvl + b)
        new_b   = β * (new_lvl - lvl) + (1 - β) * b
        lvl, b = new_lvl, new_b
    pred = lvl[:, None] + np.arange(1, horizon + 1)[None, :] * b[:, None]   # (n, H)
    return np.maximum(pred, 0)


def _croston(S, o):
    """Croston（1972）：对**非零事件**做指数平滑，预测 = 平滑后的需求量 / 平滑后的间隔。
    间歇性需求（大量零销量）的经典专用方法。"""
    α = 0.1
    win = S[:, :o + 1]
    out = np.zeros(win.shape[0])
    for i in range(win.shape[0]):
        y = win[i]
        nz = np.where(y > 0)[0]
        if len(nz) < 2:
            out[i] = y[-1]
            continue
        # 经典 Croston：z_t = y[t] 在非零时刻的取值；p_t = 间隔长度
        z = y[nz]
        p = np.diff(np.concatenate([[-1], nz]))
        # 指数平滑
        z_hat = z[0]
        p_hat = p[0]
        for t in range(1, len(z)):
            z_hat = α * z[t] + (1 - α) * z_hat
            p_hat = α * p[t] + (1 - α) * p_hat
        out[i] = z_hat / max(p_hat, 1)
    return out


def _sba(S, o):
    """Syntetos-Boylan 修正（Croston 的偏差修正版）。
    Croston 预测是有偏的（低估）；SBA 把预测值乘 (1 - α/2) 校正。
    是 M5 比赛中击败多数「朴素法」的方法之一。"""
    α = 0.1
    win = S[:, :o + 1]
    out = np.zeros(win.shape[0])
    for i in range(win.shape[0]):
        y = win[i]
        nz = np.where(y > 0)[0]
        if len(nz) < 2:
            out[i] = y[-1]
            continue
        z = y[nz]
        p = np.diff(np.concatenate([[-1], nz]))
        z_hat = z[0]
        p_hat = p[0]
        for t in range(1, len(z)):
            z_hat = α * z[t] + (1 - α) * z_hat
            p_hat = α * p[t] + (1 - α) * p_hat
        out[i] = (z_hat / max(p_hat, 1)) * (1 - α / 2)
    return out


# ------------------------------------------------------------------ 评估指标
def wmape(y, yhat) -> float:
    s = np.sum(y)
    return float(np.sum(np.abs(y - yhat)) / s) if s > 0 else np.nan


def bias_pct(y, yhat) -> float:
    s = np.sum(y)
    return float(np.sum(yhat - y) / s) if s > 0 else np.nan


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(y - yhat)))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def metrics(y, yhat) -> dict:
    return {"MAE": mae(y, yhat), "RMSE": rmse(y, yhat),
            "WMAPE": wmape(y, yhat), "Bias%": 100 * bias_pct(y, yhat)}


# ------------------------------------------------------------------ 主流程
def run(use_lgbm: bool = True) -> pd.DataFrame:
    panel = pd.read_parquet(C.PROC_DIR / "panel.parquet")
    ids, series_idx, S, P, cal = to_matrices(panel)
    n_days = S.shape[1]
    H = C.HORIZON
    print(f"[矩阵] 序列 {S.shape[0]:,} 条 × 天数 {n_days:,}")

    val_starts = [n_days - k * H - H for k in range(C.N_FOLDS)][::-1]
    print(f"[切分] 验证窗口起点 t = {val_starts}（{C.N_FOLDS} 折 × {H} 天）")

    earliest_val = min(val_starts)
    train_origins = list(range(C.MIN_HISTORY, earliest_val - H, C.TRAIN_STEP))
    print(f"[切分] 训练原点 {len(train_origins)} 个（t = {train_origins[0]} ~ {train_origins[-1]}），"
          f"训练最晚目标 t = {train_origins[-1] + H} < 验证起点 {earliest_val}")

    frames = [build_rows(S, P, cal, series_idx, o, h)
              for o in train_origins for h in range(1, H + 1)]
    train = pd.concat(frames, ignore_index=True)
    print(f"[训练集] {len(train):,} 行 × {len(FEATURES)} 特征")

    model = None
    if use_lgbm:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            objective="tweedie", variance_power=1.1,   # 计数型 + 零膨胀需求
            n_estimators=450, learning_rate=0.06, num_leaves=63,
            min_child_samples=60, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.85, reg_lambda=1.0,
            random_state=C.RANDOM_SEED, n_jobs=-1, verbose=-1,
        )
        model.fit(train[FEATURES], train["y"],
                  categorical_feature=[c for c in CATEGORICAL if c in FEATURES])

    # ---- 逐折滚动评估 ----
    fold_metrics, per_series, rows = [], [], []
    for fold, start in enumerate(val_starts):
        o = start - 1
        ys, ps, metas = [], {k: [] for k in (MODELS if model else BASELINES)}, []
        for h in range(1, H + 1):
            df = build_rows(S, P, cal, series_idx, o, h)
            y = df["y"].to_numpy()
            b = baselines(S, o, h)
            ys.append(y)
            for k in BASELINES:
                ps[k].append(b[k])
            if model is not None:
                ps["lgbm_tweedie"].append(np.maximum(model.predict(df[FEATURES]), 0))
            metas.append(df[["series_id", "date", "h", "origin", "y"]].assign(
                **{f"pred_{k}": ps[k][-1] for k in ps}))
            rows.append({"fold": fold, "h": h, **{
                k: wmape(y, ps[k][-1]) for k in ps}})
        y_all = np.concatenate(ys)
        p_all = {k: np.concatenate(v) for k, v in ps.items()}
        for k, yhat in p_all.items():
            fold_metrics.append({"fold": fold, "model": k, **metrics(y_all, yhat)})
        ps_df = pd.concat(metas, ignore_index=True)
        ps_df["id"] = np.array(ids)[ps_df["series_id"].to_numpy()]
        per_series.append(ps_df)

    detail = pd.DataFrame(rows)
    detail.to_csv(C.OUT_DIR / "backtest_by_horizon.csv", index=False)
    fm = pd.DataFrame(fold_metrics)
    fm.to_csv(C.METRICS_FILE, index=False)

    # ---- 汇总（跨折池化）----
    allps = pd.concat(per_series, ignore_index=True)
    y_all = allps["y"].to_numpy()
    stat = []
    for k in (MODELS if model else BASELINES):
        yhat = allps[f"pred_{k}"].to_numpy()
        stat.append({"model": k, **metrics(y_all, yhat)})
    stats = pd.DataFrame(stat)
    base_mae = float(stats.loc[stats.model == "seasonal_naive", "MAE"].iloc[0])
    stats["MASE"] = stats["MAE"] / base_mae        # 以季节性 naive 为标尺
    stats = stats[["model", "WMAPE", "MAE", "RMSE", "MASE", "Bias%"]]

    print(f"\n=== 验证期汇总（{C.N_FOLDS}×{H} 天，{allps['id'].nunique():,} 条序列）===")
    print(stats.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print("\n注：MASE < 1 表示优于季节性 naive；Bias% 为正表示系统性高估。")
    stats.to_csv(C.OUT_DIR / "metrics_full.csv", index=False)
    allps.to_parquet(C.OUT_DIR / "backtest_predictions.parquet", index=False)
    print(f"\n[输出] {C.OUT_DIR / 'metrics_full.csv'}")
    print(f"[输出] {C.OUT_DIR / 'backtest_predictions.parquet'}")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-lgbm", action="store_true", help="只跑 baseline")
    a = ap.parse_args()
    run(use_lgbm=not a.no_lgbm)
