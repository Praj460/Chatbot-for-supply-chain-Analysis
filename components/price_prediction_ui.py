import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.price_prediction import (
    calculate_unit_price, 
    predict_price_timeseries, 
    build_regression_model,
    get_price_related_cols,
    get_quantity_related_cols,
    get_date_related_cols,
    prepare_timeseries_data
)

def render_price_prediction_tab(df):
    """Render the Unit Price Prediction tab"""
    st.header("💰 Unit Price Prediction")
    
    # Check if we need to calculate Unit Price
    df = ensure_unit_price_exists(df)
    
    if "Unit Price (USD)" not in df.columns:
        st.error("Could not calculate or find Unit Price data.")
        st.stop()
    
    # Option to choose between Time Series and Regression models
    prediction_model = st.radio(
        "Select Price Prediction Model Type",
        ["Time Series Analysis", "Feature-based Regression"],
        help="Time Series predicts future prices based on historical patterns. Regression predicts prices based on product attributes."
    )
    
    if prediction_model == "Time Series Analysis":
        render_timeseries_analysis(df)
    else:
        render_regression_analysis(df)

def ensure_unit_price_exists(df):
    """Make sure Unit Price column exists or calculate it"""
    # First check if we have unit price in the dataset
    if "Unit Price (USD)" not in df.columns:
        # Try to calculate it if we have line item value and quantity
        if "Line Item Value" in df.columns and "Line Item Quantity" in df.columns:
            df = calculate_unit_price(df, "Line Item Value", "Line Item Quantity")
            st.success("Unit Price calculated from Line Item Value and Quantity")
        else:
            # Check column names and provide info about available columns
            st.error("Unit Price data not available and cannot be calculated")
            price_related_cols = get_price_related_cols(df)
            
            if price_related_cols:
                st.info(f"Found these price-related columns: {', '.join(price_related_cols)}")
                
                # Let user select columns to calculate unit price
                value_col = st.selectbox("Select column containing total value", 
                                      price_related_cols + ["None"])
                quantity_col = st.selectbox("Select column containing quantity", 
                                         get_quantity_related_cols(df) + ["None"])
                
                if value_col != "None" and quantity_col != "None" and st.button("Calculate Unit Price"):
                    df = calculate_unit_price(df, value_col, quantity_col)
                    st.success(f"Unit Price calculated from {value_col} and {quantity_col}")
                else:
                    st.stop()
            else:
                st.info("Your dataset doesn't contain obvious price-related columns. Please check your data.")
                st.stop()
    
    return df

def render_timeseries_analysis(df):
    """Render the Time Series Analysis section"""
    st.subheader("Time Series Price Prediction")
    
    # Check if date column exists
    date_cols = get_date_related_cols(df)
    
    if not date_cols:
        st.error("No date columns found. Time series analysis requires date information.")
        st.stop()
        
    # Let user select date column if multiple are found
    if len(date_cols) > 1:
        date_col = st.selectbox("Select date column", date_cols)
    else:
        date_col = date_cols[0]
        st.info(f"Using {date_col} for time series analysis")
    
    # Try to convert to datetime and prepare data
    try:
        df_ts = prepare_timeseries_data(df, date_col)
    except Exception as e:
        st.error(f"Error converting {date_col} to datetime: {str(e)}")
        st.info("Please make sure the date column is in a proper date format.")
        st.stop()
    
    # Select product for price prediction
    if "Product Group" in df.columns:
        product_for_price = st.selectbox(
            "Select Product for Price Analysis", 
            df["Product Group"].dropna().unique(),
            key="price_product"
        )
        
        if st.button("Predict Price Trends"):
            with st.spinner("Analyzing price patterns..."):
                try:
                    price_history, price_forecast, metrics = predict_price_timeseries(df_ts, product_for_price)
                    display_time_series_results(price_history, price_forecast, metrics, product_for_price)
                except Exception as e:
                    st.error(f"Error in price prediction: {str(e)}")
    else:
        st.error("Product Group column not found. Cannot perform product-specific time series analysis.")

def display_time_series_results(price_history, price_forecast, metrics, product_for_price):
    """Display the results of time series analysis"""
    if price_history is not None:
        # Calculate basic price statistics
        mean_price = price_history["Unit Price (USD)"].mean()
        min_price = price_history["Unit Price (USD)"].min()
        max_price = price_history["Unit Price (USD)"].max()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Price", f"${mean_price:.2f}")
        with col2:
            st.metric("Minimum Price", f"${min_price:.2f}")
        with col3:
            st.metric("Maximum Price", f"${max_price:.2f}")
        
        if price_forecast is not None:
            # Plot the historical prices and forecasts
            result_df = pd.concat([price_history, price_forecast.to_frame("Forecast")])
            
            # Create a Plotly chart for better visualization
            fig = px.line(
                result_df, 
                y=["Unit Price (USD)", "Forecast"] if "Forecast" in result_df.columns else ["Unit Price (USD)"],
                title=f"Price Trend Analysis for {product_for_price}"
            )
            fig.update_layout(yaxis_title="Unit Price (USD)", xaxis_title="Date")
            st.plotly_chart(fig, use_container_width=True)
            
            # Display forecast values
            st.subheader("Price Forecast")
            forecast_df = price_forecast.to_frame("Forecast Price (USD)")
            # First fix the index issue
            if isinstance(forecast_df.index, pd.DatetimeIndex):
                forecast_df.index = forecast_df.index.strftime('%B %Y')
            else:
                end_date = price_history.index[-1]
                new_dates = pd.date_range(
                    start=end_date + pd.DateOffset(months=1),
                    periods=len(forecast_df),
                    freq='M'
                )
                forecast_df.index = new_dates.strftime('%B %Y')

            # Then display the dataframe
            st.dataframe(forecast_df.round(2))
            
            # Calculate price volatility
            volatility = price_history["Unit Price (USD)"].std() / price_history["Unit Price (USD)"].mean() * 100
            st.metric("Price Volatility", f"{volatility:.2f}%", 
                     delta=None, delta_color="inverse")
            
            # Price trend analysis
            recent_trend = price_history["Unit Price (USD)"].iloc[-3:].pct_change().mean() * 100
            forecast_trend = price_forecast.pct_change().mean() * 100

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Recent Price Trend", 
                        f"{recent_trend:.2f}%", 
                        delta=f"{recent_trend:.1f}%",
                        delta_color="normal")
            with col2:
                st.metric("Forecast Price Trend", 
                        f"{forecast_trend:.2f}%", 
                        delta=f"{forecast_trend:.1f}%",
                        delta_color="normal")

            # Display metrics if available
            if metrics:
                st.subheader("Model Performance Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Absolute Error", f"${metrics['mae']:.2f}")
                with col2:
                    st.metric("Mean Absolute % Error", f"{metrics['mape']:.2f}%")
                with col3:
                    st.metric("Root Mean Squared Error", f"${metrics['rmse']:.2f}")

        else:
            st.warning(f"Not enough price data to generate a forecast for {product_for_price}.")
            
            # Still show historical data
            fig = px.line(
                price_history, 
                y="Unit Price (USD)",
                title=f"Historical Price Data for {product_for_price}"
            )
            fig.update_layout(yaxis_title="Unit Price (USD)", xaxis_title="Date")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No price data available for {product_for_price}.")

def render_regression_analysis(df):
    """Render the Feature-based Regression Analysis section"""
    st.subheader("Feature-based Price Prediction")
    
    if st.button("Build Price Prediction Model"):
        with st.spinner("Training regression model. This may take a moment..."):
            try:
                result = build_regression_model(df)
                
                if result and len(result) >= 7:
                    model, cat_features, num_features, mae, r2, rmse, mape, cv_scores = result
                    
                    if model:
                        display_regression_model_performance(model, cat_features, num_features, 
                                                           mae, r2, rmse, mape, cv_scores)
                        render_interactive_price_prediction(df, model, cat_features, num_features)
                    else:
                        st.error("Couldn't build the model. Please check if the dataset contains the necessary features.")
                else:
                    st.error("Insufficient data to build a reliable model. Please ensure you have enough data points with Unit Price information.")
            except Exception as e:
                st.error(f"Error building model: {str(e)}")
                st.info("Try with a different dataset or check if Unit Price values are valid numbers.")

def display_regression_model_performance(model, cat_features, num_features, mae, r2, rmse, mape, cv_scores):
    """Display regression model performance metrics"""
    # Display model performance in a more organized way
    st.subheader("Model Performance Metrics")
    
    # Create a metrics dashboard
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Absolute Error (MAE)", f"${mae:.2f}", 
                help="Average absolute difference between predicted and actual prices")
        st.metric("Root Mean Squared Error", f"${rmse:.2f}", 
                help="Square root of the average squared differences")
    with col2:
        st.metric("R² Score", f"{r2:.3f}",
                help="Proportion of variance explained by the model (1.0 is perfect)")
        st.metric("Mean Absolute Percentage Error", f"{mape:.2f}%", 
                help="Average percentage difference between predicted and actual prices")
    
    # Display cross-validation results
    st.subheader("Cross-Validation Results")
    cv_df = pd.DataFrame({
        'Fold': range(1, len(cv_scores) + 1),
        'R² Score': cv_scores
    })
    cv_avg = cv_scores.mean()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.bar(cv_df, x='Fold', y='R² Score', 
                    title="5-Fold Cross-Validation R² Scores")
        fig.add_hline(y=cv_avg, line_dash="dash", line_color="red",
                    annotation_text=f"Average: {cv_avg:.3f}")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("Average CV R² Score", f"{cv_avg:.3f}")
        st.metric("CV R² Standard Deviation", f"{cv_scores.std():.3f}")
    
    # Feature importance
    if hasattr(model['regressor'], 'feature_importances_'):
        # Get feature names after one-hot encoding
        try:
            feature_names = model['preprocessor'].get_feature_names_out()
            feature_importance = model['regressor'].feature_importances_
            
            # Create a DataFrame for visualization
            fi_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False).head(10)
            
            # Plot feature importance
            st.subheader("Top Factors Influencing Price")
            fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                    title="Feature Importance (Top 10)")
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not display feature importance: {str(e)}")

def render_interactive_price_prediction(df, model, cat_features, num_features):
    """Render interactive price prediction section"""
    st.subheader("Predict Unit Price")
    st.write("Select values to predict price for a specific scenario:")
    
    # Create input form for prediction
    prediction_inputs = {}
    
    # Add categorical features
    for feature in cat_features:
        unique_values = sorted(df[feature].dropna().unique())
        if len(unique_values) > 0:
            prediction_inputs[feature] = st.selectbox(f"Select {feature}", unique_values)
    
    # Add numerical features
    for feature in num_features:
        try:
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            mean_val = float(df[feature].mean())
            step = (max_val - min_val) / 100
            
            prediction_inputs[feature] = st.slider(
                f"Select {feature}", 
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=max(step, 0.01)
            )
        except Exception as e:
            st.warning(f"Could not create slider for {feature}: {str(e)}")
            prediction_inputs[feature] = 0
    
    # Create prediction dataframe
    pred_df = pd.DataFrame([prediction_inputs])
    
    # Make prediction
    calc_button_key = "calculate_prediction_button"
    if st.button("Calculate Predicted Price", key=calc_button_key):
        try:
            predicted_price = model.predict(pred_df)[0]
            
            # Find similar items in the dataset for reference
            df_with_features = df[cat_features + num_features + ["Unit Price (USD)"]].dropna()
            
            # Display prediction with confidence interval
            st.subheader("Predicted Unit Price")
            
            # Create columns for visual separation
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"<h1 style='text-align: center; color: #1E88E5;'>${predicted_price:.2f}</h1>", 
                        unsafe_allow_html=True)
            
            with col2:
                # Calculate median price for selected product group if available
                if "Product Group" in prediction_inputs:
                    product_group = prediction_inputs["Product Group"]
                    actual_median = df[df["Product Group"] == product_group]["Unit Price (USD)"].median()
                    if not pd.isna(actual_median):
                        st.metric(
                            f"Median Price for {product_group}", 
                            f"${actual_median:.2f}",
                            delta=f"{((predicted_price - actual_median) / actual_median * 100):.1f}%",
                            delta_color="normal"
                        )
            
            # Price distribution visualization
            if "Product Group" in prediction_inputs:
                product_group = prediction_inputs["Product Group"]
                price_data = df[df["Product Group"] == product_group]["Unit Price (USD)"].dropna()
                
                if len(price_data) > 5:
                    fig = px.histogram(
                        price_data, 
                        nbins=20,
                        title=f"Price Distribution for {product_group}",
                        labels={"value": "Unit Price (USD)", "count": "Frequency"}
                    )
                    fig.add_vline(x=predicted_price, line_dash="dash", line_color="red",
                                annotation_text="Prediction")
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")