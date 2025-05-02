import pandas as pd
import numpy as np
import re
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.freight_utils import clean_freight_cost_column_with_id_priority

def preprocess_dataframe_for_forecast(df):
    df = df.copy()

    # Clean Freight Cost specifically
    df = clean_freight_cost_column_with_id_priority(df)

    # Numeric columns
    numeric_cols = ["Unit Price", "Weight (Kilograms)", "Line Item Quantity"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

    # Categorical columns
    categorical_cols = ["Product Group", "Country", "Vendor", "Manufacturing Site", "Shipment Mode", "Dosage Form", "Sub Classification"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Outlier treatment for important numeric columns using IQR
    def apply_iqr_capping(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return series.clip(lower=lower_bound, upper=upper_bound)

    for col in ["Unit Price", "Freight Cost (USD)", "Weight (Kilograms)"]:
        if col in df.columns:
            df[col] = apply_iqr_capping(df[col])

    return df

def prepare_timeseries_data(df, date_col="Delivered to Client Date"):
    df = preprocess_dataframe_for_forecast(df)

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, "Unit Price"])
    df = df.set_index(date_col)

    # Keep only numeric columns before resampling
    df_numeric = df.select_dtypes(include=[np.number])

    ts_df = df_numeric.resample("W").mean()[["Unit Price"]].dropna()
    return ts_df

def forecast_unit_price(ts_df, forecast_weeks):
    if len(ts_df) < 10:
        raise ValueError("Not enough data to build a reliable forecast model.")

    try:
        model = SARIMAX(ts_df, order=(1,1,1), seasonal_order=(0,1,1,52))
        results = model.fit(disp=False)

        forecast = results.forecast(steps=forecast_weeks)

        metrics = None
        if len(ts_df) >= 20:
            train = ts_df.iloc[:-4]
            test = ts_df.iloc[-4:]
            eval_model = SARIMAX(train, order=(1,1,1), seasonal_order=(0,1,1,52)).fit(disp=False)
            pred = eval_model.forecast(steps=4)

            mae = mean_absolute_error(test, pred)
            rmse = np.sqrt(mean_squared_error(test, pred))
            

            metrics = {"mae": mae, "rmse": rmse}

        return ts_df, forecast, metrics

    except Exception as e:
        raise ValueError(f"Forecasting failed: {str(e)}")
