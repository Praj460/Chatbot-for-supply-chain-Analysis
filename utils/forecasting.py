import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def improved_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    epsilon = np.mean(y_true) * 0.01 if np.mean(y_true) > 0 else 0.01
    percentage_errors = np.abs((y_true - y_pred) / np.maximum(y_true, epsilon)) * 100
    percentage_errors = np.minimum(percentage_errors, 500)
    return np.mean(percentage_errors)

def calculate_reliability_score(mape, r2):
    mape = min(mape, 100) if mape is not None else 100
    mape_score = max(0, 1 - (mape / 100))
    r2_score_normalized = max(0, min(1, r2)) if r2 is not None else 0
    reliability = (0.7 * mape_score + 0.3 * r2_score_normalized) * 100
    return round(reliability)

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

def get_model_quality_description(r2):
    if r2 is None:
        return "unknown"
    elif r2 > 0.7:
        return "strong"
    elif r2 > 0.5:
        return "moderate"
    elif r2 > 0.3:
        return "weak"
    else:
        return "very weak"

def get_forecast_confidence_level(data_points, reliability_score=None):
    if reliability_score is not None:
        if reliability_score >= 70:
            return "High", "green"
        elif reliability_score >= 40:
            return "Medium", "orange"
        else:
            return "Low", "red"
    if data_points < 12:
        return "Low", "red"
    elif data_points < 24:
        return "Medium", "orange"
    else:
        return "High", "green"

# --- Forecast Function ---
def forecast_sales(df, filter_col, filter_value, debug=False):
    debug_info = {}

    if filter_col not in df.columns:
        error_msg = f"Filter column '{filter_col}' not found"
        return (None, None, {"error": error_msg}, debug_info) if debug else (None, None, {"error": error_msg})

    filtered_df = df[df[filter_col] == filter_value].copy()
    debug_info['filtered_rows'] = len(filtered_df)

    if "Delivered to Client Date" not in filtered_df.columns:
        return (None, None, {"error": "Missing 'Delivered to Client Date'"}, debug_info) if debug else (None, None, {"error": "Missing 'Delivered to Client Date'"})

    filtered_df["Delivered to Client Date"] = pd.to_datetime(filtered_df["Delivered to Client Date"], errors='coerce')
    filtered_df.dropna(subset=["Delivered to Client Date"], inplace=True)
    debug_info['rows_after_date_cleaning'] = len(filtered_df)

    if "Line Item Quantity" not in filtered_df.columns:
        return (None, None, {"error": "Missing 'Line Item Quantity'"}, debug_info) if debug else (None, None, {"error": "Missing 'Line Item Quantity'"})

    Q1 = filtered_df['Line Item Quantity'].quantile(0.25)
    Q3 = filtered_df['Line Item Quantity'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(0, Q1 - 1.5 * IQR)
    upper_bound = Q3 + 3.0 * IQR

    outlier_mask = (filtered_df['Line Item Quantity'] < lower_bound) | (filtered_df['Line Item Quantity'] > upper_bound)
    if outlier_mask.sum() < 0.1 * len(filtered_df):
        filtered_df = filtered_df[~outlier_mask]
        debug_info['outliers_removed'] = int(outlier_mask.sum())

    sales_data = filtered_df.groupby("Delivered to Client Date")["Line Item Quantity"].sum()
    sales_data = sales_data.asfreq('W').ffill().bfill().fillna(0)
    debug_info['final_data_points'] = len(sales_data)

    if sales_data.empty or len(sales_data) < 5:
        return (sales_data, None, {"error": "Insufficient data for forecasting"}, debug_info) if debug else (sales_data, None, {"error": "Insufficient data"})

    try:
        adf_p = adfuller(sales_data)[1]
        is_stationary = adf_p < 0.05
        debug_info['stationarity_test'] = "Stationary" if is_stationary else "Non-stationary"
        debug_info['adf_p_value'] = adf_p
    except Exception:
        is_stationary = False

    order = (1, 1, 1)
    seasonal_order = (1, 1, 0, 12)
    model_description = "Default SARIMA(1,1,1)(1,1,0,12)"

    train_size = int(len(sales_data) * 0.8)
    if len(sales_data) - train_size < 3:
        train_size = len(sales_data) - 3
    train_size = max(min(train_size, len(sales_data) - 1), 1)

    train_data = sales_data.iloc[:train_size]
    test_data = sales_data.iloc[train_size:]

    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False, maxiter=200)

    metrics = {}
    if len(test_data) >= 3:
        forecast_test = results.get_forecast(steps=len(test_data)).predicted_mean
        forecast_test = np.maximum(forecast_test, 0)

        rmse = np.sqrt(mean_squared_error(test_data, forecast_test))
        mae = mean_absolute_error(test_data, forecast_test)
        mape = improved_mean_absolute_percentage_error(test_data, forecast_test)
        r2 = r2_score(test_data, forecast_test)

        metrics = {
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MAPE': float(mape),
            'R2': float(r2),
            'reliability_score': calculate_reliability_score(mape, r2)
        }
    else:
        metrics = {
            'RMSE': None,
            'MAE': None,
            'MAPE': None,
            'R2': None,
            'reliability_score': None,
            'note': "Test set too small to evaluate accuracy"
        }

    final_model = SARIMAX(sales_data, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    final_results = final_model.fit(disp=False, maxiter=200)
    forecast = final_results.forecast(steps=6)
    forecast = np.maximum(forecast, 0)
    forecast = pd.Series(forecast, index=pd.date_range(sales_data.index[-1] + pd.Timedelta(weeks=1), periods=6, freq='W'))

    metrics['model_params'] = {
        'order': order,
        'seasonal_order': seasonal_order,
        'description': model_description
    }

    if metrics['MAPE'] is not None:
        metrics['forecast_accuracy'] = get_forecast_accuracy_description(metrics['MAPE'])
    if metrics['R2'] is not None:
        metrics['model_quality'] = get_model_quality_description(metrics['R2'])

    confidence_level, confidence_color = get_forecast_confidence_level(len(sales_data), metrics.get('reliability_score'))
    metrics['confidence'] = { 'level': confidence_level, 'color': confidence_color }

    return (sales_data.to_frame(), forecast, metrics, debug_info) if debug else (sales_data.to_frame(), forecast, metrics)
