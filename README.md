Here’s a comprehensive **README.md** you can drop into the root of your repo. It covers everything—from setup to features—so newcomers (and future you) can get running in minutes.

```markdown
# AI-Powered Supply Chain Analysis

A Streamlit app for **pharma supply-chain analytics** with live Google Sheets integration, AI-driven chatbot (Google Gemini), demand forecasting (SARIMA & deep learning), visualization, and unit-price prediction.

---

## 🚀 Features

1. **🤖 PharmaBot Chatbot**  
   - Ask any question about your shipment data  
   - Context-aware RAG pipeline with embedding-based chunk retrieval  
   - Structured JSON prompts for accurate, data-driven answers  
   - Follow-up memory and optional “Generate Visualization” toggle  

2. **📈 Demand Forecasting**  
   - Traditional SARIMAX forecasts with reliability scoring  
   - Deep-learning N-BEATS forecasting option via `darts`  
   - Debug info, model metrics (RMSE, MAPE, R²), and confidence indicator  

3. **📊 Interactive Visualization**  
   - Timeline slicers, multi-type charts (line, bar, histogram, pie, box, heatmap)  
   - Filter by date, product, vendor, or any dimension  
   - Downloadable (camera icon) and responsive design  

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
   ```

2. **Create & activate a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Google Sheets API**  
   - Create a service account in GCP, enable Sheets & Drive APIs  
   - Download `service_account.json` and place it in the project root  
   - Share your Google Sheet (named `Supply_Chain_Data`) with the service-account email

5. **(Optional) Install Darts for deep forecasting**  
   ```bash
   pip install "darts[torch]"
   ```

---

## ⚙️ Configuration

1. Copy `.env.example` to `.env` and set your Google API keys:
   ```ini
   GOOGLE_API_KEY=your_gemini_api_key
   ```
2. Ensure `service_account.json` is in the root (gitignored).

---

## 🚀 Running the App

```bash
streamlit run app.py
```

- **Chatbot** tab: Ask questions or visualize answers  
- **Demand Forecasting**: Toggle SARIMA vs. Deep (N-BEATS)  
- **Graph Generator**: Build custom charts with filters  
- **Unit Price Prediction**: Predict per-unit costs  

---

## 📁 Project Structure

```
├── api/
│   └── gemini_chat.py          # Gemini API wrapper
├── components/
│   ├── chatbot_ui.py           # RAG chatbot + viz button
│   ├── forecast_ui.py          # SARIMA & deep forecasting UI
│   ├── price_prediction_ui.py  # Unit price prediction UI
│   └── visualization_ui.py     # General chart builder UI
├── utils/
│   ├── google_sheets_loader.py # Live data loader + outlier cleaning
│   ├── forecasting.py          # SARIMAX forecasting logic
│   ├── deep_forecasting.py     # N-BEATS DL forecasting logic
│   ├── price_prediction.py     # RandomForest price model
│   └── data_loader.py          # Legacy CSV loader (if needed)
├── app.py                      # Main Streamlit app
├── requirements.txt
├── README.md
└── .env                        # environment variables (gitignored)
```

---

## 🤝 Contributing

1. Fork this repo  
2. Create a feature branch: `git checkout -b feature/YourFeature`  
3. Commit your changes: `git commit -m "Add awesome feature"`  
4. Push to your branch: `git push origin feature/YourFeature`  
5. Open a Pull Request  

---

