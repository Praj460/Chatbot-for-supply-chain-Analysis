import os
import streamlit as st
import pandas as pd
import chardet
import plotly.express as px
import google.generativeai as genai
from dotenv import load_dotenv
from statsmodels.tsa.statespace.sarimax import SARIMAX

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
tab1, tab2, tab3 = st.tabs(["🤖 Chatbot", "📈 Demand Forecasting", "📊 Graph Generator"])

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
            return None, None
        
        # Group by date and sum quantities
        sales_data = (filtered_df.groupby("Delivered to Client Date")
                    .agg({"Line Item Quantity": "sum"}))
        
        # Ensure data is not empty before forecasting
        if sales_data.empty or len(sales_data) < 5:  # Need at least 5 data points for SARIMAX
            return sales_data, None
        
        # Convert to weekly frequency and fill missing values
        sales_data = sales_data.asfreq('W', fill_value=0)
        
        try:
            model = SARIMAX(sales_data, order=(1,1,1), seasonal_order=(1,1,1,4))
            results = model.fit(disp=False)
            forecast = results.forecast(steps=4)  # Predict next 4 weeks
            return sales_data, forecast
        except Exception as e:
            st.error(f"Forecasting error: {str(e)}")
            return sales_data, None

    col1, col2 = st.columns(2)
    
    with col1:
        # Forecast for Product
        st.subheader("Product Demand Forecast")
        product_list = df["Product Group"].dropna().unique()
        selected_product = st.selectbox("Select Product", product_list)
        
        if st.button("Forecast Product Sales"):
            with st.spinner("Generating forecast..."):
                sales_data, forecast = forecast_sales(df, "Product Group", selected_product)
                if sales_data is not None:
                    if forecast is not None:
                        result_df = pd.concat([sales_data, forecast.to_frame("Forecast")])
                        st.line_chart(result_df)
                        st.write(f"Forecast for next 4 weeks: {forecast.tolist()}")
                    else:
                        st.warning("Not enough data to generate a forecast. Please select another product.")
                else:
                    st.warning("No data available for the selected product.")
    
    with col2:
        # Forecast for Region
        st.subheader("Regional Demand Forecast")
        region_list = df["Country"].dropna().unique()
        selected_region = st.selectbox("Select Region", region_list)
        
        if st.button("Forecast Regional Sales"):
            with st.spinner("Generating forecast..."):
                sales_data, forecast = forecast_sales(df, "Country", selected_region)
                if sales_data is not None:
                    if forecast is not None:
                        result_df = pd.concat([sales_data, forecast.to_frame("Forecast")])
                        st.line_chart(result_df)
                        st.write(f"Forecast for next 4 weeks: {forecast.tolist()}")
                    else:
                        st.warning("Not enough data to generate a forecast. Please select another region.")
                else:
                    st.warning("No data available for the selected region.")
    
    # Additional forecasting options
    st.subheader("Custom Demand Forecast")
    custom_column = st.selectbox("Select dimension to forecast by", 
                               df.select_dtypes(include=["object"]).columns.tolist(), 
                               help="Choose a categorical column to forecast by")
    
    if custom_column:
        custom_values = df[custom_column].dropna().unique()
        selected_value = st.selectbox(f"Select {custom_column}", custom_values)
        
        if st.button("Generate Custom Forecast"):
            with st.spinner("Generating forecast..."):
                sales_data, forecast = forecast_sales(df, custom_column, selected_value)
                if sales_data is not None:
                    if forecast is not None:
                        result_df = pd.concat([sales_data, forecast.to_frame("Forecast")])
                        st.line_chart(result_df)
                        st.write(f"Forecast for next 4 weeks: {forecast.tolist()}")
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
