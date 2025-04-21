import streamlit as st
import plotly.express as px
import pandas as pd

def render_visualization_tab(df, date_columns):
    st.header("📊 Graph Generator")

    # -----------------------------
    # 🗓️ Timeline Slicer
    # -----------------------------
    datetime_columns = [col for col in date_columns if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col])]
    filtered_df = df.copy()

    if datetime_columns:
        st.subheader("🗓️ Filter Visualizations by Timeline")
        selected_time_col = st.selectbox("Select date/time column", datetime_columns, key="time_col_viz")

        min_date = df[selected_time_col].min()
        max_date = df[selected_time_col].max()

        if pd.isnull(min_date) or pd.isnull(max_date):
            st.warning(f"The selected column '{selected_time_col}' does not contain valid dates to filter.")
        else:
            start_date, end_date = st.slider(
                "Select date range",
                min_value=min_date.date(),
                max_value=max_date.date(),
                value=(min_date.date(), max_date.date()),
                format="YYYY-MM-DD",
                key="date_slider_viz"
            )
            start_datetime = pd.to_datetime(start_date)
            end_datetime = pd.to_datetime(end_date)

            filtered_df = df[(df[selected_time_col] >= start_datetime) & (df[selected_time_col] <= end_datetime)]
            st.success(f"Filtered data from {start_date} to {end_date} ({len(filtered_df)} records)")

    # -----------------------------
    # 📈 Chart Controls
    # -----------------------------
    chart_type = st.selectbox("Choose a chart type", 
                              ["Line", "Bar", "Histogram", "Scatter", "Pie", "Box", "Heatmap"],
                              help="Select the type of chart you want to create")

    all_columns = filtered_df.columns.tolist()
    numeric_columns = filtered_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = filtered_df.select_dtypes(include=["object"]).columns.tolist()

    if chart_type in ["Line", "Bar", "Scatter"]:
        x_col = st.selectbox("Select X-axis column", all_columns, key="x_axis")
        y_col = st.selectbox("Select Y-axis column", numeric_columns, key="y_axis")
        color_by = st.selectbox("Color by (optional)", ["None"] + categorical_columns, key="color")

    elif chart_type == "Histogram":
        x_col = st.selectbox("Select column for histogram", all_columns, key="hist_x")
        y_col = None
        bins = st.slider("Number of bins", 5, 100, 20)
        color_by = st.selectbox("Color by (optional)", ["None"] + categorical_columns, key="hist_color")

    elif chart_type == "Pie":
        values_col = st.selectbox("Select values column", numeric_columns, key="pie_values")
        names_col = st.selectbox("Select names column", categorical_columns, key="pie_names")
        x_col, y_col = names_col, values_col
        color_by = "None"

    elif chart_type == "Box":
        x_col = st.selectbox("Select category column (X-axis)", categorical_columns, key="box_x")
        y_col = st.selectbox("Select value column (Y-axis)", numeric_columns, key="box_y")
        color_by = st.selectbox("Color by (optional)", ["None"] + categorical_columns, key="box_color")

    elif chart_type == "Heatmap":
        x_col = st.selectbox("Select X-axis column", categorical_columns, key="heat_x")
        y_col = st.selectbox("Select Y-axis column", categorical_columns, key="heat_y")
        values_col = st.selectbox("Select values column", numeric_columns, key="heat_values")
        color_by = values_col

    chart_title = st.text_input("Chart Title", "Supply Chain Analysis")

    if st.button("Generate Chart", key="gen_chart"):
        try:
            with st.spinner("Creating visualization..."):
                fig = None
                color_param = None if color_by == "None" else color_by

                if chart_type == "Line":
                    fig = px.line(filtered_df, x=x_col, y=y_col, color=color_param, title=chart_title)
                elif chart_type == "Bar":
                    fig = px.bar(filtered_df, x=x_col, y=y_col, color=color_param, title=chart_title)
                elif chart_type == "Histogram":
                    fig = px.histogram(filtered_df, x=x_col, color=color_param, nbins=bins, title=chart_title)
                elif chart_type == "Scatter":
                    fig = px.scatter(filtered_df, x=x_col, y=y_col, color=color_param, title=chart_title)
                elif chart_type == "Pie":
                    fig = px.pie(filtered_df, values=y_col, names=x_col, title=chart_title)
                elif chart_type == "Box":
                    fig = px.box(filtered_df, x=x_col, y=y_col, color=color_param, title=chart_title)
                elif chart_type == "Heatmap":
                    pivot_table = filtered_df.pivot_table(index=y_col, columns=x_col, values=values_col, aggfunc='mean')
                    fig = px.imshow(pivot_table, title=chart_title, labels=dict(color=values_col))

                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("**Note:** Use the camera icon to download the chart as an image.")

        except Exception as e:
            st.error(f"Error generating chart: {str(e)}")
            st.info("Try selecting different columns or a different chart type.")
