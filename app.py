import streamlit as st
#from utils.firestore_loader import load_data_from_firestore
from utils.google_sheets_loader import load_data_from_sheets
from components.chatbot_ui import render_chatbot_tab
from components.forecast_ui import render_forecast_tab
from components.visualization_ui import render_visualization_tab
from components.price_prediction_ui import render_price_prediction_tab

# ---------------------------
# 🔄 Load dataset 
# ---------------------------
#collection_name = "supply_chain_data"  # This should match your Firestore collection name
#df, date_columns = load_data_from_firestore(collection_name)

df, date_columns = load_data_from_sheets()
# ---------------------------
# 🧠 Main UI
# ---------------------------
st.title("📊 Data-Driven Optimization of Pharma Manufacturing and Distribution")

st.subheader("📜 Preview of Dataset")
st.dataframe(df.head())

# ---------------------------
# 🚀 Tabs
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🤖 Chatbot",
    "📈 Demand Forecasting",
    "📊 Graph Generator",
    "💰 Unit Price Prediction"
])

with tab1:
    render_chatbot_tab(df)

with tab2:
    render_forecast_tab(df)

with tab3:
    render_visualization_tab(df, date_columns)

with tab4:
    render_price_prediction_tab(df)
