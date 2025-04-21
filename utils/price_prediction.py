import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def calculate_unit_price(df, value_col, quantity_col):
    """Calculate unit price from value and quantity columns"""
    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df[quantity_col] = pd.to_numeric(df[quantity_col], errors="coerce")
    df["Unit Price (USD)"] = df.apply(
        lambda row: row[value_col] / row[quantity_col] 
        if pd.notnull(row[value_col]) and pd.notnull(row[quantity_col]) and row[quantity_col] > 0 
        else None, 
        axis=1
    )
    return df

def prepare_timeseries_data(df, date_col, product=None):
    """Prepare data for time series analysis"""
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        # Filter out rows with invalid dates
        df_ts = df.dropna(subset=[date_col, "Unit Price (USD)"])
        # Set date as index for time series operations
        df_ts = df_ts.set_index(date_col)
        
        # Filter by product if specified
        if product and "Product Group" in df.columns:
            df_ts = df_ts[df_ts["Product Group"] == product]
            
        return df_ts
    except Exception as e:
        raise Exception(f"Error preparing time series data: {str(e)}")

def predict_price_timeseries(df, product):
    """Predict future prices using time series analysis"""
    product_df = df[df["Product Group"] == product].copy()
    
    if product_df.empty:
        return None, None, None
    
    # Group by date and calculate average price
    price_data = product_df.groupby(pd.Grouper(freq='M'))[["Unit Price (USD)"]].mean()
    
    # Drop missing values and check if we have enough data
    price_data = price_data.dropna()
    if len(price_data) < 5:
        return price_data, None, None
    
    try:
        # SARIMAX model for price forecasting - with simplified parameters for robustness
        model = SARIMAX(price_data, order=(1,1,0), seasonal_order=(0,1,0,12))
        results = model.fit(disp=False)
        forecast = results.forecast(steps=6)  # Predict next 6 months
        
        # For evaluation, we can hold out the last few data points and compare with predictions
        if len(price_data) >= 5:  # Make sure we have enough data to split
            train_size = int(len(price_data) * 0.8)  # Use 80% for training
            train_data = price_data.iloc[:train_size]
            test_data = price_data.iloc[train_size:]
            
            # Fit model on training data
            model_eval = SARIMAX(train_data, order=(1,1,0), seasonal_order=(0,1,0,12))
            results_eval = model_eval.fit(disp=False)
            
            # Forecast for the test period
            test_forecast = results_eval.forecast(steps=len(test_data))
            
            # Calculate error metrics
            mae = mean_absolute_error(test_data["Unit Price (USD)"], test_forecast)
            mape = np.mean(np.abs((test_data["Unit Price (USD)"] - test_forecast) / test_data["Unit Price (USD)"]) * 100)
            rmse = np.sqrt(mean_squared_error(test_data["Unit Price (USD)"], test_forecast))
            metrics = {
                "mae": mae,
                "mape": mape,
                "rmse": rmse
            }
        else:
            metrics = None
        
        return price_data, forecast, metrics

    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")
        return price_data, None, None

def build_regression_model(df):
    """Build and train a regression model for price prediction"""
    # Drop rows with missing Unit Price
    df_model = df.dropna(subset=["Unit Price (USD)"])
    
    if len(df_model) < 10:
        return None, None, None, None, None, None, None, None
    
    # Select features to use for prediction
    categorical_features = ["Product Group", "Shipment Mode", "Country", "Vendor", "Manufacturer"]
    numerical_features = ["Line Item Quantity", "Weight (Kilograms)", "Freight Cost (USD)"]
    
    # Keep only features that exist in the dataset
    categorical_features = [f for f in categorical_features if f in df_model.columns]
    numerical_features = [f for f in numerical_features if f in df_model.columns]
    
    if not categorical_features and not numerical_features:
        return None, None, None, None, None, None, None, None
    
    # Handle missing values in features
    for feat in numerical_features:
        df_model[feat] = df_model[feat].fillna(df_model[feat].median())
    
    for feat in categorical_features:
        df_model[feat] = df_model[feat].fillna('Unknown')
    
    # Prepare features (X) and target (y)
    X = df_model[categorical_features + numerical_features]
    y = df_model["Unit Price (USD)"]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create preprocessor for categorical and numerical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ],
        remainder='passthrough'
    )
    
    # Create the model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    try:
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Get cross-validation score
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        
        return model, categorical_features, numerical_features, mae, r2, rmse, mape, cv_scores
    except Exception as e:
        raise Exception(f"Error during model training: {str(e)}")
        return None, None, None, None, None, None, None, None

def get_price_related_cols(df):
    """Get columns that might contain price information"""
    return [col for col in df.columns if any(term in col.lower() 
                            for term in ["price", "cost", "value", "amount", "usd"])]

def get_quantity_related_cols(df):
    """Get columns that might contain quantity information"""
    return [col for col in df.columns if "quantity" in col.lower()]

def get_date_related_cols(df):
    """Get columns that might contain date information"""
    return [col for col in df.columns if any(term in col.lower() 
                            for term in ["date", "time", "day", "month", "year"])]