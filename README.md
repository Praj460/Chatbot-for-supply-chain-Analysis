# AI-Powered Data Analysis with Gemini

## 📌 Overview
This project is a **Streamlit-based AI-powered supply chain analysis tool** utilizing **Google's Gemini AI**, **Plotly**, and **SARIMAX** for forecasting. It enables users to analyze procurement, order tracking, pricing, and delivery records interactively.

## 🚀 Features
- **📊 Data Preprocessing & EDA**: Cleans and formats datasets for analysis.
- **🤖 AI Chatbot**: Leverages Google Gemini AI to answer dataset-related queries.
- **📈 Forecasting**: Uses SARIMAX to predict product demand trends.
- **📉 Interactive Data Visualization**: Provides customizable charts (Line, Bar, Scatter, Histogram) using Plotly.

## 🔧 Installation
### Prerequisites
Ensure you have **Python 3.8+** and **pip** installed.

```bash
pip install -r requirements.txt
```

### API Key Setup
1. Create a `.env` file in the project directory.
2. Add your Google AI API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## 🚀 How to Run the Application
### Clone the repository:
```bash
git clone https://github.com/your-repo-url.git
cd your-repo-directory
```

### Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Run the Streamlit app:
```bash
streamlit run Gemini.py
```

### Access the app in your browser:
Open [http://localhost:8501](http://localhost:8501) to interact with the application.

## 💻 Usage
- Upload your dataset.
- Use the chatbot to analyze procurement insights.
- Visualize data with interactive charts.
- Forecast demand trends for products and regions.

## 📦 Dependencies
- `streamlit`
- `pandas`
- `chardet`
- `plotly`
- `google-generativeai`
- `python-dotenv`
- `langchain-core`
- `langchain-google-genai`
- `statsmodels`

## 🔮 Future Enhancements
- Add real-time API integration for live data fetching.
- Enhance forecasting models with deep learning techniques.
- Improve chatbot accuracy with fine-tuned prompts.



