import pandas as pd
import numpy as np
from scipy.stats import boxcox
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import itertools

def improved_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    epsilon = np.mean(y_true) * 0.01 if np.mean(y_true) > 0 else 0.01
    pe = np.abs((y_true - y_pred) / np.maximum(y_true, epsilon)) * 100
    return np.mean(np.minimum(pe, 500))

def inv_boxcox(y, lmbda):
    """Inverse Box–Cox transform."""
    if lmbda == 0:
        return np.exp(y)
    return np.power(y * lmbda + 1, 1.0 / lmbda)
    
    return np.power(y * lmbda + 1, 1.0 / lmbda)


def get_forecast_accuracy_description(mape):
    if mape is None:
        return "Accuracy unknown (insufficient validation data)", "warning"
    elif mape < 10:
        return "High accuracy forecast (MAPE < 10%)", "success"
    elif mape < 20:
        return "Good accuracy forecast (MAPE < 20%)", "info"
    elif mape < 30:
        return "Moderate accuracy forecast (MAPE < 30%)", "warning"
    else:
        return "Low accuracy forecast (MAPE > 30%)", "error"


# --- Forecast Function ---

def forecast_sales(
    df: pd.DataFrame,
    date_col: str = "Delivered to Client Date",
    qty_col: str = "Line Item Quantity",
    freq: str = "M",             # "M" or "W"
    transform: str = "boxcox",   # None, "log", or "boxcox"
    debug: bool = False
):
    """
    Returns (history_df, forecast_series, metrics_dict[, debug_info])
    where metrics_dict contains only 'RMSE', 'MAE', and 'MAPE'.
    """
    dbg = {}

    # 1. Prepare & resample
    ts = (
        df
        .assign(_dt=lambda d: pd.to_datetime(d[date_col], errors="coerce"))
        .dropna(subset=["_dt", qty_col])
        .set_index("_dt")[qty_col]
        .resample(freq).sum()
    )
    dbg["initial_points"] = len(ts)

    # 2. Trim outliers via IQR
    Q1, Q3 = ts.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lb, ub = max(0, Q1 - 1.5 * IQR), Q3 + 3 * IQR
    mask = ts.between(lb, ub)
    if (~mask).sum() < 0.1 * len(ts):
        ts = ts[mask]
        dbg["outliers_removed"] = int((~mask).sum())
    dbg["post_trim_points"] = len(ts)

    if len(ts) < 12:
        err = {"error": "Insufficient data after trimming"}
        return (ts.to_frame(name=qty_col), None, err, dbg) if debug else (ts.to_frame(name=qty_col), None, err)

    # 3. Transform
    if transform == "log":
        ts_t = np.log(ts + 1e-6)
    elif transform == "boxcox":
        arr, lam = boxcox(ts + 1e-6)
        dbg["boxcox_lambda"] = lam
        ts_t = pd.Series(arr, index=ts.index)
    else:
        ts_t = ts.copy()

    # 4. Deseasonalize
    period = 12 if freq == "M" else 52
    decomp = seasonal_decompose(ts_t, model="additive", period=period, extrapolate_trend="freq")
    ts_ds = (ts_t - decomp.seasonal).dropna()
    dbg["deseason_points"] = len(ts_ds)

    # 5. Train/test split
    split = int(0.8 * len(ts_ds))
    train, test = ts_ds.iloc[:split], ts_ds.iloc[split:]

    # 6. Auto SARIMA order search on small grid
    best_aic = np.inf
    best_order = None
    for p, q in itertools.product([0,1,2], repeat=2):
        for P, Q in itertools.product([0,1], repeat=2):
            try:
                mod = SARIMAX(
                    train,
                    order=(p,1,q),
                    seasonal_order=(P,1,Q,period),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                res = mod.fit(disp=False, maxiter=50)
                if res.aic < best_aic:
                    best_aic, best_order = res.aic, (p,1,q,P,1,Q,period)
            except:
                continue
    dbg["best_aic"] = best_aic
    dbg["best_order"] = best_order

    # 7. Fit with best order & evaluate
    p,d,q,P,D,Q_,per = best_order
    model = SARIMAX(
        train,
        order=(p,d,q),
        seasonal_order=(P,D,Q_,per),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res = model.fit(disp=False, maxiter=200)

    metrics = {}
    if len(test) >= 3:
        pred_t = res.get_forecast(steps=len(test)).predicted_mean

        # inverse transform + reseasonalize
        if transform == "log":
            test_inv = np.exp(test)
            pred_inv = np.exp(pred_t)
        elif transform == "boxcox":
            lam = dbg["boxcox_lambda"]
            test_inv = inv_boxcox(test, lam)
            pred_inv = inv_boxcox(pred_t, lam)
        else:
            test_inv, pred_inv = test, pred_t

        # add back seasonality
        seas = decomp.seasonal.reindex(test.index, method="nearest")
        test_inv += seas
        pred_inv += seas

        df_eval = pd.DataFrame({"y": test_inv, "ŷ": pred_inv}).dropna()
        y, ŷ = df_eval["y"], df_eval["ŷ"]
        metrics = {
            "RMSE": np.sqrt(mean_squared_error(y, ŷ)),
            "MAE": mean_absolute_error(y, ŷ),
            "MAPE": (np.abs((y - ŷ) / y) * 100).mean()
        }

    # 8. Refit full series + forecast forward
    final_mod = SARIMAX(
        ts_ds,
        order=(p,d,q),
        seasonal_order=(P,D,Q_,per),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    final_res = final_mod.fit(disp=False, maxiter=200)

    steps = 6 if freq == "W" else 3
    f_t = final_res.forecast(steps=steps)

    # inverse + reseason
    if transform == "log":
        f_inv = np.exp(f_t)
    elif transform == "boxcox":
        lam = dbg["boxcox_lambda"]
        f_inv = inv_boxcox(f_t, lam)
    else:
        f_inv = f_t

    next_idx = pd.date_range(
        start=ts.index[-1] + pd.DateOffset(**({"weeks":1} if freq=="W" else {"months":1})),
        periods=steps,
        freq=freq
    )
    forecast = pd.Series(
        f_inv.values + decomp.seasonal.reindex(next_idx, method="nearest"),
        index=next_idx
    )

    hist = ts.to_frame(name=qty_col)
    return (hist, forecast, metrics, dbg) if debug else (hist, forecast, metrics)
