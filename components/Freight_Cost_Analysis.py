import streamlit as st
import pandas as pd
import plotly.express as px
from utils.freight_utils import clean_freight_cost_column_with_id_priority

def render_freight_cost_tab(df):
    st.header("🚚 Freight Cost Analysis")

    # Clean and prepare data
    df = clean_freight_cost_column_with_id_priority(df)

    # Filter dropdowns
    country_list = sorted(df["Country"].dropna().unique())
    product_list = sorted(df["Product Group"].dropna().unique())

    selected_countries = st.multiselect(
    "Select Country", options=country_list, default=country_list, key="freight_country_select")
    selected_products = st.multiselect(
    "Select Product Group", options=product_list, default=product_list, key="freight_product_select")


    filtered_df = df[df["Country"].isin(selected_countries) & df["Product Group"].isin(selected_products)]

    if filtered_df.empty:
        st.warning("No data available for selected filters.")
        return

    # Summary stats
    st.subheader("Summary Statistics")
    avg_freight = filtered_df["Freight Cost (USD)"].mean()
    std_freight = filtered_df["Freight Cost (USD)"].std()
    min_freight = filtered_df["Freight Cost (USD)"].min()
    max_freight = filtered_df["Freight Cost (USD)"].max()

    col1, col2 = st.columns(2)
    col1.metric("Average Freight Cost", f"${avg_freight:.2f}")
    col2.metric("Freight Cost Volatility", f"{(std_freight/avg_freight*100):.2f}%")

    col3, col4 = st.columns(2)
    col3.metric("Min Freight Cost", f"${min_freight:.2f}")
    col4.metric("Max Freight Cost", f"${max_freight:.2f}")

    # Monthly trend
    st.subheader("📊 Monthly Avg Freight Cost (Seasonality)")
    filtered_df["Delivered to Client Date"] = pd.to_datetime(filtered_df["Delivered to Client Date"], errors='coerce')
    filtered_df = filtered_df.dropna(subset=["Delivered to Client Date", "Freight Cost (USD)"])
    filtered_df["Month_Num"] = filtered_df["Delivered to Client Date"].dt.month
    filtered_df["Month"] = filtered_df["Delivered to Client Date"].dt.month_name()

    monthly = (
        filtered_df.groupby(["Month_Num", "Month", filtered_df["Delivered to Client Date"].dt.year])["Freight Cost (USD)"]
        .mean()
        .reset_index(name="Avg Freight Cost")
    )

    seasonal = (
        monthly.groupby(["Month_Num", "Month"])["Avg Freight Cost"]
        .mean()
        .reset_index()
        .sort_values("Month_Num")
    )

    fig = px.line(
        seasonal,
        x="Month",
        y="Avg Freight Cost",
        markers=True,
        title="Average Freight Cost by Month (Seasonality)"
    )
    fig.update_layout(xaxis_title="Month", yaxis_title="Avg Freight Cost", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
