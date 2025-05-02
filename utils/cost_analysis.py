import pandas as pd
import numpy as np
import re
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

def clean_freight_cost_column_with_id_priority(df):
    df = df.copy()

    # Create lookup dictionaries
    id_lookup = {}
    asn_lookup = {}

    if "ID" in df.columns and "Freight Cost (USD)" in df.columns:
        temp_id = df[["ID", "Freight Cost (USD)"]].dropna()
        temp_id = temp_id[temp_id["Freight Cost (USD)"].apply(lambda x: isinstance(x, (int, float, np.number)))]
        id_lookup = dict(zip(temp_id["ID"], temp_id["Freight Cost (USD)"]))

    if "ASN/DN #" in df.columns and "Freight Cost (USD)" in df.columns:
        temp_asn = df[["ASN/DN #", "Freight Cost (USD)"]].dropna()
        temp_asn["ASN/DN #"] = temp_asn["ASN/DN #"].astype(str)
        temp_asn = temp_asn[temp_asn["Freight Cost (USD)"].apply(lambda x: isinstance(x, (int, float, np.number)))]
        asn_lookup = dict(zip(temp_asn["ASN/DN #"], temp_asn["Freight Cost (USD)"]))

    def process_freight(x):
        if pd.isnull(x):
            return np.nan
        elif isinstance(x, (int, float, np.number)):
            return x
        elif isinstance(x, str):
            x_lower = x.lower()
            if "freight included" in x_lower:
                return 0
            elif "invoiced separately" in x_lower:
                return np.nan
            elif "see asn" in x_lower:
                id_match = re.search(r"id#[:\s]*(\d+)", x_lower)
                asn_match = re.search(r"(asn-\d+)", x_lower)

                # Priority 1: Try to find using ID
                if id_match:
                    id_number = int(id_match.group(1))
                    if id_number in id_lookup:
                        return id_lookup[id_number]

                # Priority 2: Try to find using ASN
                if asn_match:
                    asn_number = asn_match.group(1).upper()
                    if asn_number in asn_lookup:
                        return asn_lookup[asn_number]

                return np.nan  # if neither found
            else:
                try:
                    return float(x)
                except:
                    return np.nan
        else:
            return np.nan

    df["Freight Cost (USD)"] = df["Freight Cost (USD)"].apply(process_freight)

    # Finally fill remaining missing values
    if df["Freight Cost (USD)"].isnull().sum() > 0:
        median_value = df["Freight Cost (USD)"].median()
        df["Freight Cost (USD)"].fillna(median_value, inplace=True)

    return df

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
    categorical_cols = ["Product Group", "Country", "Vendor", "Manufacturing Site", "Shipment Mode"]
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

    ts_df = df.resample("W").mean()[["Unit Price"]].dropna()
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
            mape = np.mean(np.abs((test.values.flatten() - pred.values.flatten()) / test.values.flatten())) * 100

            metrics = {"mae": mae, "rmse": rmse, "mape": mape}

        return ts_df, forecast, metrics

    except Exception as e:
        raise ValueError(f"Forecasting failed: {str(e)}")
