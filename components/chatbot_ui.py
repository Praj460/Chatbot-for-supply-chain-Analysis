import streamlit as st
import pandas as pd
from api.gemini_chat import generate_gemini_response
from sentence_transformers import SentenceTransformer, util
import json

# ----------------------------
# ✅ Load model safely (Mac M2)
# ----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# ----------------------------
# 🔁 Split DataFrame into chunks
# ----------------------------
def chunk_dataframe(df, chunk_size=200):
    return [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

# ----------------------------
# 🧠 Embed JSON-style text for better structure
# ----------------------------
def get_most_relevant_chunks(df_chunks, query, model, top_k=2):
    json_chunks = []

    for chunk in df_chunks:
        chunk = chunk.copy()

        # 🔁 Convert NaT / Timestamps to string
        chunk = chunk.applymap(lambda x: str(x) if pd.isna(x) or isinstance(x, pd.Timestamp) else x)

        # 🔁 Convert chunk to JSON string
        json_str = json.dumps(chunk.to_dict(orient="records"), indent=2)
        json_chunks.append(json_str)

    # 🔍 Embed the chunks and compare with query
    query_embedding = model.encode(query, convert_to_tensor=True)
    chunk_embeddings = model.encode(json_chunks, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
    top_indices = scores.argsort(descending=True)[:top_k]

    return [json_chunks[i] for i in top_indices]


# ----------------------------
# 🧠 Streamlit Tab with RAG
# ----------------------------
def render_chatbot_tab(df):
    st.header("🤖 PharmaBot")

    user_query = st.text_area("Type your question here")

    if st.button("Generate Analysis"):
        if not user_query.strip():
            st.warning("Please enter a question.")
            return

        model = load_embedding_model()
        df_chunks = chunk_dataframe(df)
        relevant_chunks = get_most_relevant_chunks(df_chunks, user_query, model)
        structured_data = "\n\n".join(relevant_chunks)

        column_description_text = """
You are analyzing structured shipment data with fields like:
- Country, Vendor, Product Group, Dosage Form
- Quantity, Line Item Value, Unit Price, Pack Price
- Scheduled Delivery Date, Delivered Date, Freight Cost, Weight, Manufacturing Site
Each entry is a JSON object with these fields and values.
"""

        structured_prompt = f"""
You are a supply chain analyst reviewing structured pharmaceutical shipment records (in JSON format).

{column_description_text}

Here are the relevant records:
{structured_data}

User Question:
{user_query}

Instructions:
- Use the data directly — don't explain what you would do.
- Provide insights by referencing specific field values.
- Do NOT include code. Do NOT invent data.
- Return a concise, data-backed answer.
"""

        response = generate_gemini_response(structured_prompt)
        st.subheader("📊 AI Analysis")
        st.write(response)
