import os
import streamlit as st
import pandas as pd
import numpy as np
import chardet
import plotly.express as px
import google.generativeai as genai
from dotenv import load_dotenv
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure Google AI API
if not api_key:
    st.error("GOOGLE_API_KEY is not set in the .env file")
    st.stop()

genai.configure(api_key=api_key)

# Streamlit UI
st.title("📊 AI-Powered Supply Chain Analysis with Gemini")

# -----------------------------
# 📁 Load Dataset
# -----------------------------
file_path = "/Users/prajwalanand/Documents/Gemini /SCMS_Delivery_History_Dataset_20150929.csv"  # Change this to your file path

# Detect encoding
with open(file_path, 'rb') as f:
    raw_data = f.read()
    detected_encoding = chardet.detect(raw_data)['encoding']

# Read CSV using detected encoding
@st.cache_data
def load_data():
    df = pd.read_csv(file_path, encoding=detected_encoding, encoding_errors='replace')
    
    # Convert columns to appropriate data types
    df["Weight (Kilograms)"] = pd.to_numeric(df["Weight (Kilograms)"], errors="coerce")
    df["Freight Cost (USD)"] = pd.to_numeric(df["Freight Cost (USD)"], errors="coerce")

    # Convert date columns to datetime format
    date_columns = [
        "PQ First Sent to Client Date",
        "PO Sent to Vendor Date",
        "Scheduled Delivery Date",
        "Delivered to Client Date",
        "Delivery Recorded Date",
    ]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Fill missing values in categorical columns with 'Unknown'
    categorical_cols = ["Shipment Mode", "Dosage"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Fill missing numerical values with the median
    numerical_cols = ["Line Item Insurance (USD)"]
    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            
    return df, date_columns

df, date_columns = load_data()

# Data preview
st.subheader("📜 Preview of Dataset")
st.dataframe(df.head())

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🤖 Chatbot", "📈 Demand Forecasting", "📊 Graph Generator", "💰 Unit Price Prediction"])

with tab1:
    st.header("🤖 Supply Chain Analytics Chatbot")
    
    # -----------------------------
    # 🤖 Gemini Chatbot (Using direct genai library instead of LangChain)
    # -----------------------------
    st.subheader("🔍 Ask a Question About the Dataset")
    user_query = st.text_area("Type your question here", key="query_area")

    if st.button("Generate Analysis", key="generate_btn"):
        if user_query.strip():
            try:
                with st.spinner("Generating analysis..."):
                    structured_prompt = f"""
    You are an AI specialized in supply chain and procurement analytics. 
    Given the dataset containing procurement details, order tracking, pricing, and delivery records, 
    analyze the data to provide insights for different stakeholders like suppliers, manufacturers, and clients. 

    **Dataset Overview:**
    - Columns: {list(df.columns)}
    - Summary: {df.describe().to_string()[:1000]}
    Go through the dataset thoroughly as the questions are only related to the dataset.

    **User Query:**
    {user_query}

    **Instructions:**
    - Provide a concise, data-driven response.
    - Avoid multiple messages; return a single final answer.
    - If needed, include relevant statistics or insights.
    - Do not return any code snippets.
                    """

                    # Use direct genai library instead of LangChain
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    response = model.generate_content(structured_prompt)

                    st.subheader("📊 AI Analysis")
                    st.write(response.text)

            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a question before generating an analysis.")

with tab2:
    st.header("📈 Demand Forecasting")
    
    # -----------------------------
    # Demand Forecasting Functions
    # -----------------------------
    def forecast_sales(df, filter_col, filter_value):
        filtered_df = df[df[filter_col] == filter_value]
        
        if filtered_df.empty:
            return None, None, None
        
        # Group by date and sum quantities
        sales_data = (filtered_df.groupby("Delivered to Client Date")
                    .agg({"Line Item Quantity": "sum"}))
        
        # Ensure data is not empty before forecasting
        if sales_data.empty or len(sales_data) < 5:  # Need at least 5 data points for SARIMAX
            return sales_data, None, None
        
        # Convert to weekly frequency and fill missing values
        sales_data = sales_data.asfreq('W', fill_value=0)
        
        try:
            # Split data into train and test sets for evaluation
            train_size = int(len(sales_data) * 0.8)
            train_data = sales_data[:train_size]
            test_data = sales_data[train_size:]
            
            # Fit the model on training data
            model = SARIMAX(train_data, order=(1,1,1), seasonal_order=(1,1,1,4))
            results = model.fit(disp=False)
            
            # Generate in-sample predictions for evaluation
            in_sample_pred = results.get_prediction(start=0, end=len(train_data)-1)
            in_sample_mean = in_sample_pred.predicted_mean
            
            # Generate out-of-sample predictions for test data
            if len(test_data) > 0:
                out_sample_pred = results.get_forecast(steps=len(test_data))
                out_sample_mean = out_sample_pred.predicted_mean
                
                # Calculate evaluation metrics
                metrics = {
                    'RMSE': np.sqrt(mean_squared_error(test_data, out_sample_mean)),
                    'MAE': mean_absolute_error(test_data, out_sample_mean),
                    'MAPE': mean_absolute_percentage_error(test_data, out_sample_mean) * 100,
                    'R2': r2_score(test_data, out_sample_mean)
                }
            else:
                metrics = None
            
            # Refit the model on the full dataset for final forecast
            full_model = SARIMAX(sales_data, order=(1,1,1), seasonal_order=(1,1,1,4))
            full_results = full_model.fit(disp=False)
            forecast = full_results.forecast(steps=4)  # Predict next 4 weeks
            
            return sales_data, forecast, metrics
        except Exception as e:
            st.error(f"Forecasting error: {str(e)}")
            return sales_data, None, None

    # Product Demand Forecast
    st.subheader("Product Demand Forecast")
    product_list = df["Product Group"].dropna().unique()
    selected_product = st.selectbox("Select Product", product_list)
    
    if st.button("Forecast Product Sales"):
        with st.spinner("Generating forecast..."):
            sales_data, forecast, metrics = forecast_sales(df, "Product Group", selected_product)
            if sales_data is not None:
                if forecast is not None:
                    result_df = pd.concat([sales_data, forecast.to_frame("Forecast")])
                    
                    # Create a more informative chart using Plotly
                    fig = px.line(
                        result_df, 
                        y=["Line Item Quantity", "Forecast"],
                        title=f"Demand Forecast for {selected_product}"
                    )
                    fig.update_layout(
                        yaxis_title="Quantity", 
                        xaxis_title="Date",
                        legend_title="Data Type",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display model performance metrics if available
                    if metrics:
                        st.subheader("Model Performance Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("RMSE", f"{metrics['RMSE']:.2f}")
                            st.caption("Root Mean Square Error (lower is better)")
                        
                        with col2:
                            st.metric("MAE", f"{metrics['MAE']:.2f}")
                            st.caption("Mean Absolute Error (lower is better)")
                        
                        with col3:
                            st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                            st.caption("Mean Absolute Percentage Error")
                        
                        with col4:
                            st.metric("R²", f"{metrics['R2']:.2f}")
                            st.caption("Coefficient of Determination (higher is better)")
                        
                        # Add interpretation of model quality
                        if metrics['R2'] > 0.7:
                            quality = "strong"
                        elif metrics['R2'] > 0.5:
                            quality = "moderate"
                        elif metrics['R2'] > 0.3:
                            quality = "weak"
                        else:
                            quality = "very weak"
                            
                        st.info(f"This model shows a {quality} predictive power based on historical data. " +
                               f"On average, predictions are off by {metrics['MAE']:.1f} units ({metrics['MAPE']:.1f}%).")
                    
                    # Display forecast values in a more readable format
                    st.subheader("Forecast for next 4 weeks")
                    forecast_df = forecast.to_frame("Forecasted Quantity")
                    forecast_df.index = forecast_df.index.strftime('%b %d, %Y')
                    st.dataframe(forecast_df.astype(int))
                    
                    # Add confidence level indicator
                    data_points = len(sales_data)
                    if data_points < 10:
                        confidence = "Low"
                        color = "red"
                    elif data_points < 20:
                        confidence = "Medium"
                        color = "orange"
                    else:
                        confidence = "High"
                        color = "green"
                    
                    st.markdown(f"<p>Forecast confidence: <span style='color:{color};font-weight:bold'>{confidence}</span> (based on {data_points} historical data points)</p>", unsafe_allow_html=True)
                    
                else:
                    st.warning("Not enough data to generate a forecast. Please select another product.")
            else:
                st.warning("No data available for the selected product.")
    
    # Additional forecasting options
    st.subheader("Custom Demand Forecast")
    st.write("Select any dimension to forecast demand by:")
    
    custom_column = st.selectbox(
        "Select dimension to forecast by", 
        df.select_dtypes(include=["object"]).columns.tolist(), 
        help="Choose any categorical column to forecast by"
    )
    
    if custom_column:
        custom_values = df[custom_column].dropna().unique()
        selected_value = st.selectbox(f"Select {custom_column}", custom_values)
        
        if st.button("Generate Custom Forecast"):
            with st.spinner("Generating forecast..."):
                sales_data, forecast, metrics = forecast_sales(df, custom_column, selected_value)
                if sales_data is not None:
                    if forecast is not None:
                        result_df = pd.concat([sales_data, forecast.to_frame("Forecast")])
                        
                        # Enhanced visualization
                        fig = px.line(
                            result_df, 
                            y=["Line Item Quantity", "Forecast"],
                            title=f"Demand Forecast for {custom_column}: {selected_value}"
                        )
                        fig.update_layout(
                            yaxis_title="Quantity", 
                            xaxis_title="Date",
                            legend_title="Data Type",
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display model performance metrics if available
                        if metrics:
                            st.subheader("Model Performance Metrics")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("RMSE", f"{metrics['RMSE']:.2f}")
                                st.metric("MAE", f"{metrics['MAE']:.2f}")
                            
                            with col2:
                                st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                                st.metric("R²", f"{metrics['R2']:.2f}")
                            
                            # Add visual indicator of forecast reliability
                            reliability_score = min(100, max(0, metrics['R2'] * 100))
                            st.progress(reliability_score/100)
                            st.caption(f"Forecast Reliability Score: {reliability_score:.0f}%")
                        
                        # Display forecast statistics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Average Weekly Demand", 
                                f"{int(sales_data['Line Item Quantity'].mean())}", 
                                delta=f"{int(forecast.mean() - sales_data['Line Item Quantity'].mean())}"
                            )
                        with col2:
                            growth_rate = ((forecast.mean() / sales_data['Line Item Quantity'].mean()) - 1) * 100
                            st.metric(
                                "Forecast Growth", 
                                f"{growth_rate:.1f}%",
                                delta=f"{growth_rate:.1f}%",
                                delta_color="normal"
                            )
                        
                        # Display forecast values
                        st.subheader("Detailed Forecast")
                        forecast_df = forecast.to_frame("Forecasted Quantity")
                        forecast_df.index = forecast_df.index.strftime('%b %d, %Y')
                        st.dataframe(forecast_df.astype(int))
                        
                        # Add model details expandable section
                        with st.expander("Model Details"):
                            st.write("SARIMAX Model Parameters:")
                            st.code("order=(1,1,1), seasonal_order=(1,1,1,4)")
                            st.write("Training Data Points:", len(sales_data))
                            if metrics:
                                st.write("Forecast Interpretation:")
                                if metrics['MAPE'] < 10:
                                    st.success("High accuracy forecast (MAPE < 10%)")
                                elif metrics['MAPE'] < 20:
                                    st.info("Good accuracy forecast (MAPE < 20%)")
                                elif metrics['MAPE'] < 30:
                                    st.warning("Moderate accuracy forecast (MAPE < 30%)")
                                else:
                                    st.error("Low accuracy forecast (MAPE > 30%)")
                    else:
                        st.warning(f"Not enough data to generate a forecast for {selected_value}.")
                else:
                    st.warning(f"No data available for the selected {custom_column}.")

with tab3:
    st.header("📊 Graph Generator")
    
    # -----------------------------
    # 🗓️ Timeline Slicer for Visualization Tab
    # -----------------------------
    datetime_columns = [col for col in date_columns if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col])]
    filtered_df = df.copy()  # Default to full dataset
    
    if datetime_columns:
        st.subheader("🗓️ Filter Visualizations by Timeline")
        selected_time_col = st.selectbox("Select date/time column", datetime_columns, key="time_col_viz")

        min_date = df[selected_time_col].min()
        max_date = df[selected_time_col].max()

        if pd.isnull(min_date) or pd.isnull(max_date):
            st.warning(f"The selected column '{selected_time_col}' does not contain valid dates to filter.")
        else:
            # Convert to native Python date for Streamlit compatibility
            start_date, end_date = st.slider(
                "Select date range",
                min_value=min_date.date(),
                max_value=max_date.date(),
                value=(min_date.date(), max_date.date()),
                format="YYYY-MM-DD",
                key="date_slider_viz"
            )

            # Convert start and end back to datetime for filtering
            start_datetime = pd.to_datetime(start_date)
            end_datetime = pd.to_datetime(end_date)

            filtered_df = df[(df[selected_time_col] >= start_datetime) & (df[selected_time_col] <= end_datetime)]
            st.success(f"Filtered data from {start_date} to {end_date} ({len(filtered_df)} records)")
    
    # -----------------------------
    # 📈 Plotly Visualizations
    # -----------------------------
    chart_type = st.selectbox("Choose a chart type", 
                            ["Line", "Bar", "Histogram", "Scatter", "Pie", "Box", "Heatmap"],
                            help="Select the type of chart you want to create")
    
    # Dynamic options based on chart type
    all_columns = filtered_df.columns.tolist()
    numeric_columns = filtered_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = filtered_df.select_dtypes(include=["object"]).columns.tolist()
    
    # Different options based on chart type with explanations
    if chart_type in ["Line", "Bar", "Scatter"]:
        x_col = st.selectbox("Select X-axis column", all_columns, key="x_axis")
        y_col = st.selectbox("Select Y-axis column", numeric_columns, key="y_axis")
        color_by = st.selectbox(
            "Color by (optional)", 
            ["None"] + categorical_columns, 
            key="color",
            help="Colors data points by the selected category. For example, selecting 'Shipment Mode' will use different colors for different shipping methods."
        )
        
    elif chart_type == "Histogram":
        x_col = st.selectbox("Select column for histogram", all_columns, key="hist_x")
        y_col = None
        bins = st.slider("Number of bins", 5, 100, 20)
        color_by = st.selectbox(
            "Color by (optional)", 
            ["None"] + categorical_columns, 
            key="hist_color",
            help="Splits the histogram by categories, each category getting its own color."
        )
        
    elif chart_type == "Pie":
        values_col = st.selectbox("Select values column", numeric_columns, key="pie_values")
        names_col = st.selectbox("Select names column", categorical_columns, key="pie_names")
        x_col = names_col
        y_col = values_col
        color_by = "None"
        
    elif chart_type == "Box":
        x_col = st.selectbox("Select category column (X-axis)", categorical_columns, key="box_x")
        y_col = st.selectbox("Select value column (Y-axis)", numeric_columns, key="box_y")
        color_by = st.selectbox(
            "Color by (optional)", 
            ["None"] + categorical_columns, 
            key="box_color",
            help="Colors boxes by the selected category, allowing comparison across multiple dimensions."
        )
        
    elif chart_type == "Heatmap":
        x_col = st.selectbox("Select X-axis column", categorical_columns, key="heat_x")
        y_col = st.selectbox("Select Y-axis column", categorical_columns, key="heat_y")
        values_col = st.selectbox("Select values column", numeric_columns, key="heat_values")
        color_by = values_col
    
    # Add title option
    chart_title = st.text_input("Chart Title", "Supply Chain Analysis")
    
    if st.button("Generate Chart", key="gen_chart"):
        try:
            with st.spinner("Creating visualization..."):
                # Choose appropriate Plotly chart based on selection
                if chart_type == "Line":
                    color_param = None if color_by == "None" else color_by
                    fig = px.line(filtered_df, x=x_col, y=y_col, color=color_param, title=chart_title)
                    
                elif chart_type == "Bar":
                    color_param = None if color_by == "None" else color_by
                    fig = px.bar(filtered_df, x=x_col, y=y_col, color=color_param, title=chart_title)
                    
                elif chart_type == "Histogram":
                    color_param = None if color_by == "None" else color_by
                    fig = px.histogram(filtered_df, x=x_col, color=color_param, nbins=bins, title=chart_title)
                    
                elif chart_type == "Scatter":
                    color_param = None if color_by == "None" else color_by
                    fig = px.scatter(filtered_df, x=x_col, y=y_col, color=color_param, title=chart_title)
                    
                elif chart_type == "Pie":
                    fig = px.pie(filtered_df, values=y_col, names=x_col, title=chart_title)
                    
                elif chart_type == "Box":
                    color_param = None if color_by == "None" else color_by
                    fig = px.box(filtered_df, x=x_col, y=y_col, color=color_param, title=chart_title)
                    
                elif chart_type == "Heatmap":
                    # For heatmap, need to pivot the data
                    pivot_table = filtered_df.pivot_table(index=y_col, columns=x_col, values=values_col, aggfunc='mean')
                    fig = px.imshow(pivot_table, title=chart_title, labels=dict(color=values_col))

                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Option to download chart (simplified, using HTML)
                st.markdown("**Note:** To save this chart, use the camera icon in the chart's toolbar.")
                
        except Exception as e:
            st.error(f"Error generating chart: {str(e)}")
            st.info("Try selecting different columns or chart type")
    
# Add this as a new tab in your application

with tab4:
    st.header("💰 Unit Price Prediction")
    
    # -----------------------------
    # Unit Price Analysis & Prediction
    # -----------------------------
    
    # First check if we have unit price in the dataset
    if "Unit Price (USD)" not in df.columns:
        # Try to calculate it if we have line item value and quantity
        if "Line Item Value" in df.columns and "Line Item Quantity" in df.columns:
            # Create a copy of the dataframe to avoid SettingWithCopyWarning
            df = df.copy()
            # Convert columns to numeric to ensure division works properly
            df["Line Item Value"] = pd.to_numeric(df["Line Item Value"], errors="coerce")
            df["Line Item Quantity"] = pd.to_numeric(df["Line Item Quantity"], errors="coerce")
            # Calculate Unit Price and handle division by zero
            df["Unit Price (USD)"] = df.apply(
                lambda row: row["Line Item Value"] / row["Line Item Quantity"] 
                if row["Line Item Quantity"] > 0 else None, 
                axis=1
            )
            st.success("Unit Price calculated from Line Item Value and Quantity")
        else:
            # Check column names and provide info about available columns
            st.error("Unit Price data not available and cannot be calculated")
            st.write("Available columns that might contain price information:")
            price_related_cols = [col for col in df.columns if any(term in col.lower() 
                                 for term in ["price", "cost", "value", "amount", "usd"])]
            if price_related_cols:
                st.info(f"Found these price-related columns: {', '.join(price_related_cols)}")
                
                # Let user select columns to calculate unit price
                value_col = st.selectbox("Select column containing total value", 
                                      price_related_cols + ["None"])
                quantity_col = st.selectbox("Select column containing quantity", 
                                         [col for col in df.columns if "quantity" in col.lower()] + ["None"])
                
                if value_col != "None" and quantity_col != "None" and st.button("Calculate Unit Price"):
                    df = df.copy()
                    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
                    df[quantity_col] = pd.to_numeric(df[quantity_col], errors="coerce")
                    df["Unit Price (USD)"] = df.apply(
                        lambda row: row[value_col] / row[quantity_col] 
                        if pd.notnull(row[value_col]) and pd.notnull(row[quantity_col]) and row[quantity_col] > 0 
                        else None, 
                        axis=1
                    )
                    st.success(f"Unit Price calculated from {value_col} and {quantity_col}")
                else:
                    st.stop()
            else:
                st.info("Your dataset doesn't contain obvious price-related columns. Please check your data.")
                st.stop()
    
    # Option to choose between Time Series and Regression models
    prediction_model = st.radio(
        "Select Price Prediction Model Type",
        ["Time Series Analysis", "Feature-based Regression"],
        help="Time Series predicts future prices based on historical patterns. Regression predicts prices based on product attributes."
    )
    
    if prediction_model == "Time Series Analysis":
        # TIME SERIES APPROACH
        st.subheader("Time Series Price Prediction")
        
        # Check if date column exists
        date_cols = [col for col in df.columns if any(term in col.lower() for term in ["date", "time", "day", "month", "year"])]
        
        if not date_cols:
            st.error("No date columns found. Time series analysis requires date information.")
            st.stop()
            
        # Let user select date column if multiple are found
        if len(date_cols) > 1:
            date_col = st.selectbox("Select date column", date_cols)
        else:
            date_col = date_cols[0]
            st.info(f"Using {date_col} for time series analysis")
        
        # Try to convert to datetime
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            # Filter out rows with invalid dates
            df_ts = df.dropna(subset=[date_col, "Unit Price (USD)"])
            # Set date as index for time series operations
            df_ts = df_ts.set_index(date_col)
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
            
            # Define time series prediction function
            def predict_price_timeseries(df, product):
                product_df = df[df["Product Group"] == product].copy()
                
                if product_df.empty:
                    return None, None
                
                # Group by date and calculate average price
                price_data = product_df.groupby(pd.Grouper(freq='M'))[["Unit Price (USD)"]].mean()
                
                # Drop missing values and check if we have enough data
                price_data = price_data.dropna()
                if len(price_data) < 5:
                    return price_data, None
                
                try:
                    # SARIMAX model for price forecasting - with simplified parameters for robustness
                    model = SARIMAX(price_data, order=(1,1,0), seasonal_order=(0,1,0,12))
                    results = model.fit(disp=False)
                    forecast = results.forecast(steps=6)  # Predict next 6 months
                    # Add evaluation metrics here
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
                    st.error(f"Prediction error: {str(e)}")
                    return price_data, None, None
            
            if st.button("Predict Price Trends"):
                with st.spinner("Analyzing price patterns..."):
                    price_history, price_forecast, metrics = predict_price_timeseries(df_ts, product_for_price)
                    
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

                            # Then display the dataframe - outside the if-else
                            st.dataframe(forecast_df.round(2))
                            
                            # Calculate price volatility
                            volatility = price_history["Unit Price (USD)"].std() / price_history["Unit Price (USD)"].mean() * 100
                            st.metric("Price Volatility", f"{volatility:.2f}%", 
                                     delta=None, delta_color="inverse")
                            
                            # Price trend analysis
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

                        # INSERT THE NEW METRICS CODE RIGHT HERE
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
                        # END OF NEW CODE

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
        else:
            st.error("Product Group column not found. Cannot perform product-specific time series analysis.")
    
    else:  # Feature-based Regression
        # FEATURE-BASED REGRESSION APPROACH
        st.subheader("Feature-based Price Prediction")
        
        # Function to build and train the price prediction model
        def build_regression_model(df):
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
                st.error(f"Error during model training: {str(e)}")
                return None, None, None, None, None, None, None, None
        
        if st.button("Build Price Prediction Model"):
            with st.spinner("Training regression model. This may take a moment..."):
                try:
                    result = build_regression_model(df)
                    
                    if result and len(result) >= 7:
                        model, cat_features, num_features, mae, r2, rmse, mape, cv_scores = result
                        
                        if model:
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
                            
                            # Interactive price prediction
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
                        else:
                            st.error("Couldn't build the model. Please check if the dataset contains the necessary features.")
                    else:
                        st.error("Insufficient data to build a reliable model. Please ensure you have enough data points with Unit Price information.")
                except Exception as e:
                    st.error(f"Error building model: {str(e)}")
                    st.info("Try with a different dataset or check if Unit Price values are valid numbers.")
