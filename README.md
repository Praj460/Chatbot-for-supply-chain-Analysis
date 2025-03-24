📊 AI-Powered Data Analysis with Gemini

This application allows users to upload datasets, perform exploratory data analysis (EDA), clean and preprocess data, filter by timelines, ask questions about the data, and visualize data using interactive charts. It leverages Google Gemini (Gemini 2.0) for AI-powered insights and Streamlit for the user interface.

🛠️ Requirements

Make sure you have the following installed:

Python 3.x

Streamlit

pandas

plotly

chardet

langchain_google_genai

google.generativeai

dotenv

You can install the dependencies by running:

bash
Copy
Edit
pip install -r requirements.txt
⚡ Project Setup
Download and Configure the .env file:

Create a .env file in the root of your project.

Add your Google API key in the .env file:

ini
Copy
Edit
GOOGLE_API_KEY=your_google_api_key_here
File Path Configuration:

In the code, specify the correct file path for the dataset you want to load.

python
Copy
Edit
file_path = "/path/to/your/dataset.csv"
🔧 Features
Dataset Upload and Preprocessing:

The dataset is loaded directly from a specified file path.

Data is preprocessed by:

Detecting encoding.

Converting numerical and date columns.

Handling missing values in categorical and numerical columns.

Exploratory Data Analysis (EDA):

Summary statistics and data preview are displayed.

Missing values are handled appropriately.

Date columns are converted to datetime format for analysis.

Timeline Filtering:

Users can filter the dataset based on a selected date range.

AI-Powered Insights:

Users can ask questions about the dataset.

AI (Gemini 2.0) analyzes the data and provides insights based on the user's query.

Interactive Data Visualizations:

Users can create visualizations (Line, Bar, Histogram, Scatter plots) using Plotly.

🚀 How to Run the Application
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-repo-url.git
cd your-repo-directory
Install the required dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
Access the app in your browser: Open http://localhost:8501 to interact with the application.

🧑‍💻 Usage
Upload and Process Dataset: The application automatically reads the dataset from the specified file path.

Data Filtering: Use the timeline slicer to filter the data by a specific range of dates.

Ask AI: Enter a question related to the dataset, and the Gemini-powered AI will provide insights.

Visualize Data: Select a chart type and columns to generate visualizations that help analyze your data better.

📄 Limitations
The dataset must be in CSV format.

The app only works with datasets that contain specific columns for dates and numerical data types.

Google API key is required for generating AI-powered insights.

🔒 Security
Make sure not to expose your .env file publicly. Keep your Google API key safe by not sharing it.

🛠️ Built With
Streamlit: Framework for building data apps.

pandas: Data manipulation and analysis library.

Plotly: Visualization library for creating interactive charts.

Gemini (Google Generative AI): AI model used for generating insights from data.

langchain_google_genai: A wrapper for Google’s generative AI model for question-answering tasks.
