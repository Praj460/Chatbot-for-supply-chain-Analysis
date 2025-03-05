import os
import streamlit as st
import pandas as pd
import chardet
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure Google AI API
if not api_key:
    st.error("GOOGLE_API_KEY is not set in the .env file")
    st.stop()

genai.configure(api_key=api_key)

# Streamlit UI
st.title("📊 AI-Powered Data Analysis with Gemini")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        file_extension = uploaded_file.name.split(".")[-1]

        if file_extension == "csv":
            # Detect encoding
            raw_data = uploaded_file.read()
            detected_encoding = chardet.detect(raw_data)['encoding']

            # Read file with detected encoding (newer pandas versions)
            uploaded_file.seek(0)  # Reset file position
            df = pd.read_csv(uploaded_file, encoding=detected_encoding, encoding_errors='replace')

        else:
            df = pd.read_excel(uploaded_file)

        # Display dataset
        st.subheader("📜 Preview of Dataset")
        st.dataframe(df.head())

        # Convert dataset to a text-friendly format
        dataset_summary = df.describe().to_string()  # Get statistical summary
        dataset_text = f"Dataset Columns: {list(df.columns)}\n\nSummary:\n{dataset_summary}"

        # User input
        st.subheader("🔍 Ask a Question About the Dataset")
        user_query = st.text_area("Type your question here")

        if st.button("Generate Analysis"):
            if user_query.strip():
                try:
                    structured_prompt = f"""
            You are an AI specialized in supply chain and procurement analytics. 
            Given the dataset containing procurement details, order tracking, pricing, and delivery records, 
            analyze the data to provide insights for different stakeholders like suppliers, manufacturers, and 
            clients. 

            **Dataset Overview:**
            - Columns: {list(df.columns)}
            - Summary: {df.describe().to_string()[:1000]}  # Truncate long summaries

            **User Query:**
            {user_query}

            **Instructions:**
            - Provide a concise, data-driven response.
            - Avoid multiple messages; return a single final answer.
            - If needed, include relevant statistics or insights.
            """

                    # Send dataset summary + user query to Gemini AI
                    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")  # Change model if needed
                    message = HumanMessage(content=structured_prompt)
                    response = model.invoke([message])  # Ensures a single final answer

                    # Display AI response
                    st.subheader("📊 AI Analysis")
                    st.write(response.content)  # Corrected: Directly access content

                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a question before generating an analysis.")
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

