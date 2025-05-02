import streamlit as st

PRIMARY_COLOR = "#00b4d8"
BUTTON_BG = "#0077b6"  
BUTTON_HOVER = "#21867a"
BUTTON_TEXT_COLOR = "#ffffff"

SERVICES = [
    ("🤖 PharmaBot", "Get instant answers about products, availability, pricing, and operational questions.", "chatbot"),
    ("📈 Demand Forecasting and Analysis", "Visual representation of predicted demand for various pharmaceutical products.", "forecast"),
    ("📊 Graph Generator", "Interactive graphs showing vendor performance, distribution trends, and bottlenecks.", "visualization"),
    ("💰 Unit Price Prediction and Analysis", "Predict future prices of medicines based on trends, demand, and supply.", "price"),
    ("🚛 Shipment Mode Analysis", "Analyze how costs change by Air, Ocean, Truck, and other modes.", "shipment"),
    ("📤 Submit Record", "Submit new shipment records directly to the database.", "data_entry"),
    ("🚚 Freight Cost Analysis", "Track and compare freight charges across modes and suppliers.", "freight")
]

def render_homepage():
    st.markdown(f"""
        <h1 style='text-align:center; font-size:3.5rem; color:{PRIMARY_COLOR};'>PharmaFlow</h1>
        <p style='text-align:center; font-size:1.1rem;'>
            PharmaFlow delivers end-to-end pharmaceutical supply chain solutions—from accurate demand forecasting and dynamic price prediction to detailed freight cost analysis and interactive data visualizations. Featuring a powerful chatbot assistant, customizable analytics dashboards, shipment mode insights, bulk record submission, and comprehensive reporting, PharmaFlow equips manufacturers, vendors, and stakeholders with the visibility and tools needed for confident, data-driven decisions.
        </p>
        <h2 style='text-align:center; margin-top:3rem;'>Our Services</h2>
        <style>
            div.stButton > button {{
                background-color: {BUTTON_BG};
                color: {BUTTON_TEXT_COLOR};
                padding: 1.5rem;
                margin: 0.75rem auto;
                width: 80%;
                text-align: center;
                border-radius: 12px;
                font-size: 1.1rem;
                line-height: 1.6;
                border: none;
                transition: background 0.3s ease;
                display: block;
            }}
            div.stButton > button:hover {{
                background-color: {BUTTON_HOVER};
            }}
        </style>
    """, unsafe_allow_html=True)

    for label, description, target in SERVICES:
        if st.button(f"{label}\n\n{description}", key=target):
            st.session_state.page = target
