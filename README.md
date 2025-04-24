# Data-Driven Optimization of Pharma Manufacturing and Distribution

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

### Create & activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows

##Install dependencies
pip install -r requirements.txt

##Set up Google Sheets API

1.Create a service account in GCP and enable the Google Sheets API & Google Drive API

2.Download the service_account.json key file and place it in the project root

3.Share your Google Sheet (named Supply_Chain_Data) with the service-account email

##Running the App

streamlit run app.py
