import streamlit as st
import pandas as pd
import plotly.express as px
from utils.price_forecasting import (
    preprocess_dataframe_for_forecast,
    prepare_timeseries_data,
    forecast_unit_price
)

def render_price_forecasting_tab(df):
    st.header("📈 Pharma Price Forecasting")

    # Required‐columns check
    required = {
        "Product Group", "Country", "Vendor", "Shipment Mode",
        "Manufacturing Site", "Dosage Form", "Sub Classification",
        "Delivered to Client Date", "Unit Price"
    }
    if not required.issubset(df.columns):
        st.error("Required columns are missing from the dataset!")
        return

    # 1️⃣ Mandatory Filters
    product_group = st.selectbox(
        "Select Product Group",
        sorted(df["Product Group"].dropna().unique())
    )
    df_pg = df[df["Product Group"] == product_group]

    country = st.selectbox(
        "Select Country",
        sorted(df_pg["Country"].dropna().unique())
    )
    df_ct = df_pg[df_pg["Country"] == country]

    # 2️⃣ Optional Filters (Select All)
    def multi_filter(label, col, base_df):
        opts = ["Select All"] + sorted(base_df[col].dropna().unique())
        sel = st.multiselect(label, opts, default=["Select All"])
        if "Select All" in sel:
            return base_df
        return base_df[base_df[col].isin(sel)]

    df_v = multi_filter("Select Vendor(s)", "Vendor", df_ct)
    df_sm = multi_filter("Select Shipment Mode(s)", "Shipment Mode", df_v)
    df_ms = multi_filter("Select Manufacturing Site(s)", "Manufacturing Site", df_sm)
    df_df = multi_filter("Select Dosage Form(s)", "Dosage Form", df_ms)
    df_sc = multi_filter("Select Sub Classification(s)", "Sub Classification", df_df)

    # 3️⃣ How many months to forecast?
    months = st.selectbox("Select Number of Months to Forecast", [1,2,3,4,5,6], index=3)

    # 4️⃣ Generate
    if st.button("Generate Forecast"):
        with st.spinner("Processing and forecasting future prices..."):
            final_df = df_sc.copy()
            if final_df.empty:
                st.warning("No data available for the selected combination.")
                return

            # Preprocess & build ts
            cleaned = preprocess_dataframe_for_forecast(final_df)
            ts_df = prepare_timeseries_data(cleaned, date_col="Delivered to Client Date")

            if len(ts_df) < 10:
                st.error("Not enough data for reliable forecasting (need ≥10 points).")
                return

            # Forecast
            hist, fcast, metrics = forecast_unit_price(ts_df, months)

            # ── FIX MONTHLY INDEX SO to_period("M") WORKS ──
            fcast.index = pd.date_range(
                start=hist.index[-1] + pd.DateOffset(months=1),
                periods=len(fcast),
                freq="M"
            )

            # — Plot history + forecast —
            st.subheader(f"Forecast for {product_group} in {country} (Next {months} Mo)")
            dfp = pd.concat([hist, fcast.to_frame("Forecast")], axis=1)
            fig = px.line(
                dfp,
                y=[hist.columns[0], "Forecast"],
                title=f"{product_group} in {country}: Actual vs Forecast"
            )
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Unit Price (USD)",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # — Summary statistics —
            avg = hist["Unit Price"].mean()
            vol = hist["Unit Price"].std() / avg * 100 if avg else 0
            mn, mx = hist["Unit Price"].min(), hist["Unit Price"].max()
            st.subheader("Summary Statistics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg Price",      f"${avg:.2f}")
            c2.metric("Volatility",     f"{vol:.2f}%")
            c3.metric("Min Price",      f"${mn:.2f}")
            c4.metric("Max Price",      f"${mx:.2f}")

            # — Model evaluation —
            if metrics:
                st.subheader("Model Evaluation Metrics")
                e1, e2 = st.columns(2)
                e1.metric("Mean Absolute Error (MAE)",      f"${metrics['mae']:.2f}")
                e2.metric("Root Mean Squared Error (RMSE)", f"${metrics['rmse']:.2f}")

            # — Forecast table —
            st.subheader("Forecasted Prices")
            df_f = fcast.to_frame("Forecasted Price (USD)").round(2)
            df_f.index = df_f.index.to_period("M").strftime("%b %Y")
            st.dataframe(df_f)

            # — Optional: show your seasonality chart too
            display_unit_price_seasonality(cleaned)



def display_forecast_results(history, forecast, metrics, product_group, country, forecast_weeks):
    st.subheader(f"Price Forecast for {product_group} in {country} (Next {forecast_weeks} Weeks)")

    # Set forecast index to continue from last history date
    forecast_index = pd.date_range(start=history.index[-1] + pd.Timedelta(days=7), periods=len(forecast), freq='W')
    forecast.index = forecast_index

    combined_df = pd.concat([history, forecast.to_frame("Forecasted Price")])

    fig = px.line(
        combined_df,
        y=["Unit Price", "Forecasted Price"] if "Forecasted Price" in combined_df.columns else ["Unit Price"],
        title=f"Forecasted Unit Price Trend for {product_group} in {country} (Next {forecast_weeks} Weeks)"
    )
    fig.update_layout(
        yaxis_title="Unit Price (USD)",
        xaxis_title="Week",
        xaxis_range=[combined_df.index.min(), combined_df.index.max()]  # ✅ Now safe
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecasted Prices")
    forecast_df = forecast.to_frame("Predicted Unit Price (USD)").round(2)
    st.dataframe(forecast_df)

    avg_price = history["Unit Price"].mean()
    volatility = history["Unit Price"].std() / avg_price * 100
    min_price = history["Unit Price"].min()
    max_price = history["Unit Price"].max()


    st.subheader("Summary Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Historical Price", f"${avg_price:.2f}")
    with col2:
        st.metric("Price Volatility", f"{volatility:.2f}%")

    col3, col4 = st.columns(2)
    with col3:
        st.metric("Min Price", f"${min_price:.2f}")
    with col4:
        st.metric("Max Price", f"${max_price:.2f}")

    if metrics:
        st.subheader("Model Evaluation Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Absolute Error (MAE)", f"${metrics['mae']:.2f}")
        with col2:
            st.metric("Root Mean Squared Error (RMSE)", f"${metrics['rmse']:.2f}")

def display_unit_price_seasonality(filtered_df):
    st.subheader("💰 Monthly Seasonality: Avg Unit Price")

    if "Delivered to Client Date" not in filtered_df.columns:
        st.warning("Delivered to Client Date column missing.")
        return

    # Handle both possible unit price columns
    unit_price_col = "Unit Price (USD)" if "Unit Price (USD)" in filtered_df.columns else "Unit Price"
    if unit_price_col not in filtered_df.columns:
        st.warning("Unit Price column missing.")
        return

    filtered_df["Delivered to Client Date"] = pd.to_datetime(
        filtered_df["Delivered to Client Date"], errors='coerce'
    )
    filtered_df = filtered_df.dropna(subset=["Delivered to Client Date", unit_price_col])
    filtered_df["Month_Num"] = filtered_df["Delivered to Client Date"].dt.month
    filtered_df["Month"] = filtered_df["Delivered to Client Date"].dt.month_name()

    monthly_data = (
        filtered_df.groupby(["Month_Num", "Month", filtered_df["Delivered to Client Date"].dt.year])[unit_price_col]
        .mean()
        .reset_index(name="Avg Unit Price")
    )

    seasonal_price = (
        monthly_data.groupby(["Month_Num", "Month"])["Avg Unit Price"]
        .mean()
        .reset_index()
        .sort_values("Month_Num")
    )

    fig = px.line(
        seasonal_price,
        x="Month",
        y="Avg Unit Price",
        markers=True,
        title="Monthly Seasonality: Avg Unit Price"
    )
    fig.update_layout(xaxis_title="Month", yaxis_title="Avg Unit Price", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
