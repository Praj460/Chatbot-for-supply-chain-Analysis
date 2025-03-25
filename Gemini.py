import os
import streamlit as st
import pandas as pd
import chardet
import plotly.express as px
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
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
st.title("📊 AI-Powered Data Analysis with Gemini")

# -----------------------------
# 📁 Load Dataset from Code
# -----------------------------
file_path = "/Users/prajwalanand/Documents/Gemini /SCMS_Delivery_History_Dataset_20150929.csv"  # Change this to your file path

# Detect encoding
with open(file_path, 'rb') as f:
    raw_data = f.read()
    detected_encoding = chardet.detect(raw_data)['encoding']

# Read CSV using detected encoding
df = pd.read_csv(file_path, encoding=detected_encoding, encoding_errors='replace')

# -----------------------------
# 🧹 Preprocessing & EDA
# -----------------------------
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

# -----------------------------
# 📜 Preview Data
# -----------------------------
st.subheader("📜 Preview of Dataset")
st.dataframe(df.head())

# -----------------------------
# 🗓️ Timeline Slicer
# -----------------------------
datetime_columns = [col for col in date_columns if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col])]

if datetime_columns:
    st.subheader("🗓️ Filter by Timeline")
    selected_time_col = st.selectbox("Select date/time column", datetime_columns)

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
            format="YYYY-MM-DD"
        )

        # Convert start and end back to datetime for filtering
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date)

        df = df[(df[selected_time_col] >= start_datetime) & (df[selected_time_col] <= end_datetime)]
        st.success(f"Filtered data from {start_date} to {end_date}")


# -----------------------------
# 🤖 Gemini Chatbot
# -----------------------------
st.subheader("🔍 Ask a Question About the Dataset")
user_query = st.text_area("Type your question here")

if st.button("Generate Analysis"):
    if user_query.strip():
        try:
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

            model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
            message = HumanMessage(content=structured_prompt)
            response = model.invoke([message])

            st.subheader("📊 AI Analysis")
            st.write(response.content)

        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a question before generating an analysis.")
#---------------------------
# Demand Forecasting Function
#---------------------------
def forecast_sales(df, filter_col, filter_value):
    filtered_df = df[df[filter_col] == filter_value]
    sales_data = (filtered_df.groupby("Delivered to Client Date")
                  .agg({"Line Item Quantity": "sum"})
                  .asfreq('W')  # Weekly frequency
                  .fillna(0))  # Fill missing weeks with 0
    
    model = SARIMAX(sales_data, order=(1,1,1), seasonal_order=(1,1,1,4))
    results = model.fit()
    forecast = results.forecast(steps=4)  # Predict next 4 weeks
    
    return sales_data, forecast

# Forecast for Product
st.subheader("📈 Forecast Sales for a Product")
product_list = df["Product Group"].dropna().unique()
selected_product = st.selectbox("Select Product", product_list)
if st.button("Forecast Product Sales"):
    sales_data, forecast = forecast_sales(df, "Product Group", selected_product)
    st.line_chart(pd.concat([sales_data, forecast.to_frame("Forecast")]))

# Forecast for Region
st.subheader("📈 Forecast Sales for a Region")
region_list = df["Country"].dropna().unique()
selected_region = st.selectbox("Select Region", region_list)
if st.button("Forecast Regional Sales"):
    sales_data, forecast = forecast_sales(df, "Country", selected_region)
    st.line_chart(pd.concat([sales_data, forecast.to_frame("Forecast")]))

# -----------------------------
# 📈 Plotly Visualizations
# -----------------------------
st.subheader("📈 Visualize Your Data")

chart_type = st.selectbox("Choose a chart type", ["Line", "Bar", "Histogram", "Scatter"])
numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
all_columns = df.columns.tolist()

x_col = st.selectbox("Select X-axis column", all_columns)
y_col = st.selectbox("Select Y-axis column", numeric_columns if chart_type != "Histogram" else all_columns)

if st.button("Generate Chart"):
    try:
        if chart_type == "Line":
            fig = px.line(df, x=x_col, y=y_col)
        elif chart_type == "Bar":
            fig = px.bar(df, x=x_col, y=y_col)
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=y_col)
        elif chart_type == "Scatter":
            fig = px.scatter(df, x=x_col, y=y_col)

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating chart: {str(e)}")
