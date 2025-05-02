import streamlit as st
import pandas as pd
import plotly.express as px
from utils.forecasting import forecast_sales

def render_forecast_tab(df):
    st.header("📈 Demand Forecasting")
    st.subheader("Filter and Generate Forecast")

    # — Single-select filters —
    country_list = sorted(df["Country"].dropna().unique())
    selected_country = st.selectbox(
        "Select Country",
        country_list,
        key="fc_country"
    )

    product_list = sorted(df["Product Group"].dropna().unique())
    selected_product = st.selectbox(
        "Select Product Group",
        product_list,
        key="fc_product"
    )

    # — Number of months dropdown —
    months = st.selectbox(
        "Select Number of Months to Forecast",
        [1, 2, 3, 4, 5, 6],
        index=2,
        key="fc_months"
    )

    # — Generate button —
    if st.button("Generate Forecast", key="fc_button"):
        with st.spinner("Generating forecast…"):
            filt = df[
                (df["Country"] == selected_country) &
                (df["Product Group"] == selected_product)
            ].copy()

            if filt.empty:
                st.warning(
                    f"No data for Country {selected_country} & Product Group {selected_product}."
                )
                return

            # — Call SARIMA with monthly Box–Cox —
            history, full_forecast, metrics = forecast_sales(
                filt,
                date_col="Delivered to Client Date",
                qty_col="Line Item Quantity",
                freq="M",            # monthly aggregation
                transform="boxcox",
                debug=False
            )

            # Trim to user‐selected horizon
            forecast = full_forecast.iloc[:months]

            _render_results(
                history,
                forecast,
                metrics,
                f"{selected_country} | {selected_product}",
                months
            )

            # — Additional summary tables —
            display_top_manufacturing_sites(filt)
            display_top_vendors(filt)
            display_sites_and_vendors(filt)
            display_monthly_delivery_seasonality(filt)
            display_monthly_quantity_seasonality(filt)


def _render_results(history, forecast, metrics, label, months):
    # Error check
    if history is None or forecast is None or metrics.get("error"):
        st.error(metrics.get("error", "Insufficient data to forecast."))
        return

    # — Plot actual vs forecast —
    st.subheader(f"Forecast for {label} (next {months} mo)")
    df_plot = pd.concat([history, forecast.to_frame("Forecast")], axis=1)
    fig = px.line(
        df_plot,
        y=[history.columns[0], "Forecast"],
        title=f"{label}: Actual vs Forecast"
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=history.columns[0],
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # — Metrics display: RMSE / MAE / MAPE / Volatility —
    volatility = history.std().iloc[0] / history.mean().iloc[0] * 100

    st.subheader("Model Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "RMSE",
        f"{metrics['RMSE']:.2f}" if metrics.get('RMSE') is not None else "N/A"
    )
    c2.metric(
        "MAE",
        f"{metrics['MAE']:.2f}" if metrics.get('MAE') is not None else "N/A"
    )
    c3.metric(
        "MAPE",
        f"{metrics['MAPE']:.2f}%" if metrics.get('MAPE') is not None else "N/A"
    )
    c4.metric(
        "Volatility",
        f"{volatility:.2f}%"
    )

    # — Forecast table —
    st.subheader("Forecast Details")
    df_f = forecast.to_frame("Forecasted Quantity")
    # Format index as Mon YYYY
    df_f.index = df_f.index.to_period("M").strftime("%b %Y")
    st.dataframe(df_f.round(2))


# — Top-5 helper functions —

def display_top_manufacturing_sites(filtered_df):
    required = ["Manufacturing Site", "Line Item Quantity"]
    if not set(required).issubset(filtered_df.columns):
        st.warning("Manufacturing site data missing. Skipping table.")
        return

    st.subheader("Top 5 Manufacturing Sites by Total Quantity")
    df_site = filtered_df.dropna(subset=required)
    stats = (
        df_site
        .groupby("Manufacturing Site")["Line Item Quantity"]
        .sum()
        .nlargest(5)
        .reset_index()
    )
    stats.columns = ["Manufacturing Site", "Total Quantity"]
    st.table(stats)


def display_top_vendors(filtered_df):
    required = [
        "Vendor",
        "Line Item Quantity",
        "Scheduled Delivery Date",
        "Delivered to Client Date",
        "Freight Cost (USD)"
    ]
    if not set(required).issubset(filtered_df.columns):
        st.warning("Vendor data missing. Skipping table.")
        return

    st.subheader("Top 5 Vendors by Quantity & On-time Delivery")
    df_v = (
        filtered_df
        .dropna(subset=required)
        .assign(
            Scheduled=lambda d: pd.to_datetime(
                d["Scheduled Delivery Date"], errors="coerce"),
            Delivered=lambda d: pd.to_datetime(
                d["Delivered to Client Date"], errors="coerce")
        )
        .dropna(subset=["Scheduled", "Delivered"])
    )
    df_v["On Time"] = df_v["Delivered"] <= df_v["Scheduled"]

    vendor_stats = (
        df_v
        .groupby("Vendor")
        .agg(
            Total_Quantity=("Line Item Quantity", "sum"),
            Deliveries=("Vendor", "count"),
            On_Time_Pct=("On Time", "mean")
        )
        .nlargest(5, "Total_Quantity")
        .reset_index()
    )
    vendor_stats["On-time Delivery (%)"] = (vendor_stats["On_Time_Pct"] * 100).round(1)

    out = vendor_stats[[
        "Vendor",
        "Total_Quantity",
        "Deliveries",
        "On-time Delivery (%)"
    ]]
    out.columns = [
        "Vendor",
        "Total Quantity",
        "Deliveries",
        "On-time Delivery (%)"
    ]
    st.table(out)


def display_sites_and_vendors(filtered_df):
    required = ["Manufacturing Site", "Vendor", "Line Item Quantity"]
    if not set(required).issubset(filtered_df.columns):
        st.warning("Site/vendor data missing. Skipping table.")
        return

    st.subheader("Top 5 Sites with Vendors & Quantity")
    df_c = filtered_df.dropna(subset=required)
    top_sites = (
        df_c
        .groupby("Manufacturing Site")["Line Item Quantity"]
        .sum()
        .nlargest(5)
        .index
    )
    df_top = df_c[df_c["Manufacturing Site"].isin(top_sites)]
    combined = (
        df_top
        .groupby(["Manufacturing Site", "Vendor"])["Line Item Quantity"]
        .sum()
        .reset_index()
    )
    combined.columns = ["Manufacturing Site", "Vendor", "Total Quantity"]
    combined = combined.sort_values([
        "Manufacturing Site",
        "Total Quantity"
    ], ascending=[True, False])
    st.table(combined)

def display_monthly_delivery_seasonality(filtered_df):
    """📈 Monthly Seasonality: Avg Number of Deliveries"""
    st.subheader("📅 Monthly Seasonality: Avg Number of Deliveries")

    if "Delivered to Client Date" not in filtered_df.columns:
        st.warning("Delivered to Client Date column missing.")
        return

    # Prepare month buckets
    df = filtered_df.copy()
    df["Delivered to Client Date"] = pd.to_datetime(df["Delivered to Client Date"], errors="coerce")
    df = df.dropna(subset=["Delivered to Client Date"])
    df["Month"] = df["Delivered to Client Date"].dt.month_name()
    df["Month_Num"] = df["Delivered to Client Date"].dt.month

    # Count deliveries per month & year, then average across years
    monthly_counts = (
        df.groupby(["Month_Num", "Month", df["Delivered to Client Date"].dt.year])
          .size()
          .reset_index(name="Deliveries")
    )
    seasonal = (
        monthly_counts
        .groupby(["Month_Num", "Month"])["Deliveries"]
        .mean()
        .reset_index()
        .sort_values("Month_Num")
    )

    # Plot
    fig = px.line(
        seasonal,
        x="Month",
        y="Deliveries",
        markers=True,
        title="Average Monthly Deliveries"
    )
    fig.update_layout(xaxis_title="", yaxis_title="Avg # Deliveries", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


def display_monthly_quantity_seasonality(filtered_df):
    """📦 Monthly Seasonality: Avg Line-Item Quantity per Delivery"""
    st.subheader("📦 Monthly Seasonality: Avg Line-Item Quantity per Delivery")

    required = ["Delivered to Client Date", "Line Item Quantity"]
    if not set(required).issubset(filtered_df.columns):
        st.warning("Required columns missing (Delivered to Client Date, Line Item Quantity).")
        return

    df = filtered_df.copy()
    df["Delivered to Client Date"] = pd.to_datetime(df["Delivered to Client Date"], errors="coerce")
    df = df.dropna(subset=["Delivered to Client Date", "Line Item Quantity"])
    df["Month"] = df["Delivered to Client Date"].dt.month_name()
    df["Month_Num"] = df["Delivered to Client Date"].dt.month

    # Compute average quantity per delivery per month-year, then average across years
    monthly_qty = (
        df.groupby(["Month_Num", "Month", df["Delivered to Client Date"].dt.year])["Line Item Quantity"]
          .mean()
          .reset_index(name="Avg Quantity")
    )
    seasonal = (
        monthly_qty
        .groupby(["Month_Num", "Month"])["Avg Quantity"]
        .mean()
        .reset_index()
        .sort_values("Month_Num")
    )

    # Plot
    fig = px.line(
        seasonal,
        x="Month",
        y="Avg Quantity",
        markers=True,
        title="Average Monthly Line-Item Quantity"
    )
    fig.update_layout(xaxis_title="", yaxis_title="Avg Quantity per Delivery", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
