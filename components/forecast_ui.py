import streamlit as st
import pandas as pd
import plotly.express as px
from utils.forecasting import (
    forecast_sales, 
    get_forecast_confidence_level, 
    get_model_quality_description,
    get_forecast_accuracy_description,
    calculate_reliability_score
)

def render_forecast_tab(df):
    st.header("📈 Demand Forecasting")
    render_product_forecast(df)
    render_custom_forecast(df)

def render_product_forecast(df):
    st.subheader("Product Demand Forecast")
    product_list = df["Product Group"].dropna().unique()
    selected_product = st.selectbox("Select Product", product_list)
    show_debug = st.checkbox("Show debug info", value=False, key="product_debug")

    if st.button("Forecast Product Sales"):
        with st.spinner("Generating forecast..."):
            if show_debug:
                sales_data, forecast, metrics, debug_info = forecast_sales(df, "Product Group", selected_product, debug=True)
                display_forecast_results(sales_data, forecast, metrics, "Product Group", selected_product, debug_info)
            else:
                sales_data, forecast, metrics = forecast_sales(df, "Product Group", selected_product)
                display_forecast_results(sales_data, forecast, metrics, "Product Group", selected_product)

def render_custom_forecast(df):
    st.subheader("Custom Demand Forecast")
    st.write("Select any dimension to forecast demand by:")

    custom_column = st.selectbox("Select dimension to forecast by", df.select_dtypes(include=["object"]).columns.tolist())

    if custom_column:
        custom_values = df[custom_column].dropna().unique()
        selected_value = st.selectbox(f"Select {custom_column}", custom_values)
        show_debug = st.checkbox("Show debug info", value=False, key="custom_debug")

        if st.button("Generate Custom Forecast"):
            with st.spinner("Generating forecast..."):
                if show_debug:
                    sales_data, forecast, metrics, debug_info = forecast_sales(df, custom_column, selected_value, debug=True)
                    display_forecast_results(sales_data, forecast, metrics, custom_column, selected_value, debug_info)
                else:
                    sales_data, forecast, metrics = forecast_sales(df, custom_column, selected_value)
                    display_forecast_results(sales_data, forecast, metrics, custom_column, selected_value)

def display_forecast_results(sales_data, forecast, metrics, dimension_name, dimension_value, debug_info=None):
    if debug_info:
        with st.expander("Debug Information", expanded=True):
            st.subheader("Data Processing Debug Info")
            for key, value in debug_info.items():
                st.text(f"{key}: {value}")
            
            if 'model_choice' in debug_info:
                st.subheader("Model Selection")
                st.text(f"Selected model: {debug_info['model_choice']}")
            if 'stationarity_test' in debug_info:
                st.text(f"Stationarity test: {debug_info['stationarity_test']}")
                if 'adf_p_value' in debug_info:
                    st.text(f"ADF p-value: {debug_info['adf_p_value']}")
            if sales_data is not None:
                st.subheader("Raw Sales Data")
                st.dataframe(sales_data.head(10))

    if sales_data is not None:
        if forecast is not None:
            result_df = pd.concat([sales_data, forecast.to_frame("Forecast")])
            fig = px.line(result_df, y=["Line Item Quantity", "Forecast"],
                          title=f"Demand Forecast for {dimension_name}: {dimension_value}")
            fig.update_layout(yaxis_title="Quantity", xaxis_title="Date", legend_title="Data Type", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            display_metrics(metrics, sales_data)
            display_forecast_statistics(sales_data, forecast)
            display_forecast_details(forecast)
            display_confidence_indicator(sales_data, metrics.get('reliability_score'))
            display_model_details(metrics)
        else:
            if metrics and "error" in metrics:
                st.error(f"Error: {metrics['error']}")
                if sales_data is not None and not sales_data.empty:
                    st.subheader("Available Data Statistics")
                    st.write(f"Number of data points: {len(sales_data)}")
                    st.write(f"Date range: {sales_data.index.min()} to {sales_data.index.max()}")
                    st.write(f"Total quantity: {sales_data['Line Item Quantity'].sum()}")
                    st.subheader("Available Data")
                    st.dataframe(sales_data)
            else:
                st.warning(f"Not enough data to generate a forecast for {dimension_value}.")
    else:
        st.warning(f"No data available for the selected {dimension_name}.")

def display_metrics(metrics, sales_data):
    st.subheader("Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    rmse = metrics.get('RMSE')
    mae = metrics.get('MAE')
    mape = metrics.get('MAPE')
    r2 = metrics.get('R2')

    with col1:
        st.metric("RMSE", f"{rmse:.2f}" if rmse is not None else "N/A")
        st.caption("Root Mean Square Error (lower is better)")
    with col2:
        st.metric("MAE", f"{mae:.2f}" if mae is not None else "N/A")
        st.caption("Mean Absolute Error (lower is better)")
    with col3:
        mape_display = f"{mape:.2f}%" if mape is not None else "N/A"
        st.metric("MAPE", mape_display)
        st.caption("Mean Absolute Percentage Error")
    with col4:
        st.metric("R²", f"{r2:.2f}" if r2 is not None else "N/A")
        st.caption("Coefficient of Determination (higher is better)")

    quality = get_model_quality_description(r2)
    if mae is not None and mape is not None:
        st.info(f"This model shows a {quality} predictive power. Avg prediction error: {mae:.1f} units ({mape:.1f}%).")
    else:
        st.info(f"This model shows a {quality} predictive power.")

    reliability_score = metrics.get('reliability_score')
    if reliability_score is not None:
        st.progress(reliability_score/100)
        st.caption(f"Forecast Reliability Score: {reliability_score:.0f}%")

def display_forecast_statistics(sales_data, forecast):
    historical_mean = sales_data['Line Item Quantity'].mean()
    forecast_mean = forecast.mean()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Weekly Demand", f"{int(historical_mean)}", delta=f"{int(forecast_mean - historical_mean)}")
    with col2:
        growth_rate = ((forecast_mean / historical_mean) - 1) * 100 if historical_mean > 0 else (100 if forecast_mean > 0 else 0)
        st.metric("Forecast Growth", f"{growth_rate:.1f}%", delta=f"{growth_rate:.1f}%", delta_color="normal")

def display_forecast_details(forecast):
    forecast_steps = len(forecast)
    period_text = "weeks" if forecast_steps <= 8 else "periods"
    st.subheader(f"Forecast for next {forecast_steps} {period_text}")
    forecast_df = forecast.to_frame("Forecasted Quantity")
    forecast_df.index = forecast_df.index.strftime('%b %d, %Y')
    st.dataframe(forecast_df.astype(int))

def display_confidence_indicator(sales_data, reliability_score=None):
    data_points = len(sales_data)
    confidence, color = get_forecast_confidence_level(data_points, reliability_score)
    date_span = (sales_data.index.max() - sales_data.index.min()).days if not sales_data.empty else None
    date_info = f" and a time span of {date_span} days" if date_span else ""
    st.markdown(f"<p>Forecast confidence: <span style='color:{color};font-weight:bold'>{confidence}</span> (based on {data_points} data points{date_info})</p>", unsafe_allow_html=True)

def display_model_details(metrics):
    with st.expander("Model Details"):
        if metrics and "model_params" in metrics:
            order = metrics["model_params"]["order"]
            seasonal_order = metrics["model_params"]["seasonal_order"]
            description = metrics["model_params"]["description"]
            st.write(f"Model: {description}")
            st.code(f"order={order}, seasonal_order={seasonal_order}")
            st.info("Model parameters were automatically selected based on time series patterns.")
        else:
            st.write("SARIMAX Model Parameters:")
            st.code("order=(1,1,1), seasonal_order=(1,1,1,4)")

        if metrics and "error" not in metrics:
            if "forecast_accuracy" in metrics:
                accuracy_desc, status = metrics["forecast_accuracy"]
                getattr(st, status)(accuracy_desc)
            elif metrics.get("MAPE") is not None:
                accuracy_desc, status = get_forecast_accuracy_description(metrics["MAPE"])
                getattr(st, status)(accuracy_desc)
            if "reliability_score" in metrics and metrics["reliability_score"] is not None:
                score = metrics["reliability_score"]
                if score >= 70:
                    st.success(f"Reliability score: {score:.0f}% - High confidence in forecast.")
                elif score >= 40:
                    st.info(f"Reliability score: {score:.0f}% - Moderate confidence in forecast.")
                else:
                    st.warning(f"Reliability score: {score:.0f}% - Low confidence. Consider more data.")
                st.caption("Reliability score is derived from R² and MAPE.")
            if "confidence" in metrics:
                conf = metrics["confidence"]
                st.markdown(f"<p>Model confidence level: <span style='color:{conf['color']};font-weight:bold'>{conf['level']}</span></p>", unsafe_allow_html=True)
