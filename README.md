# AI-Powered Supply Chain Analysis

A Streamlit app for **pharma supply-chain analytics** with live Google Sheets integration, AI-driven chatbot (Google Gemini), demand forecasting (SARIMA & deep learning), visualization, and unit-price prediction.

---

## 🚀 Features

1. **🤖 PharmaBot Chatbot**  
   - Ask any question about your shipment data  
   - Context-aware RAG pipeline with embedding-based chunk retrieval  
   - Structured JSON prompts for accurate, data-driven answers  
   - Follow-up memory and “Generate Visualization” toggle  

2. **📈 Demand Forecasting**  
   - Traditional SARIMAX forecasts with reliability scoring  
   - Deep-learning N-BEATS forecasting option via `darts`  
   - Debug info, model metrics (RMSE, MAPE, R²), and confidence indicator  

3. **📊 Interactive Visualization**  
   - Timeline slicers, multi-type charts (line, bar, histogram, pie, box, heatmap)  
   - Filter by date, product, vendor, or any dimension  
   - Responsive design and download via chart toolbar  

4. **💰 Unit Price Prediction**  
   - RandomForest regression for per-unit price  
   - Combined with time-series pack-price forecasting  
   - Key driver analysis and prediction intervals  

5. **🔄 Live Data Backend**  
   - Google Sheets API (no manual CSV uploads)  
   - Service-account auth for secure, real-time sync  
   - Outlier removal built into loader  

---

## 🛠️ Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/Praj460/Chatbot-for-supply-chain-Analysis.git
   cd Chatbot-for-supply-chain-Analysis
