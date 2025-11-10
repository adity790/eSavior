import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from openai import OpenAI


lifespan_model = joblib.load('lifespan_model.pkl')
disposal_model = joblib.load('disposal_model.pkl')
current_price_model = joblib.load('price_model.pkl')
dump_price_model = joblib.load('dumb_model.pkl')

categorical_cols = ['device_type', 'department', 'condition', 'status', 'warranty_status', 'Damage_History', 'Replacement_Type']

label_encoders = {col: joblib.load(f'label_encoder_{col}.pkl') for col in categorical_cols}

feature_cols = ['device_type', 'department', 'condition', 'status', 'warranty_status',
                'power_rating_watt', 'previous_repairs', 'device_age_years', 'days_since_last_repair','Original_Price', 
                'Damage_History', 'Replacement_Type']

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

def chat_with_gpt(prompt):
    response = client.chat.completions.create(
        model="deepseek/deepseek-chat-v3.1:free",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

st.set_page_config(page_title="eSavior - E-Waste Dashboard", layout="wide")
st.title("♻️ eSavior Dashboard: E-Waste Forecast & Repair Assistant")

tab1, tab2 = st.tabs(["📊 E-Waste Forecast", "🛠️ RepairGPT Assistant"])

with tab1:
    st.header("📈 Forecast Devices Likely to Become E-Waste")
    uploaded_file = st.file_uploader("Upload Test Device CSV", type=["csv"])

    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)

        current_year = datetime.now().year
        test_df['purchase_date'] = pd.to_datetime(test_df['purchase_date'], errors='coerce')
        test_df['last_repair_date'] = pd.to_datetime(test_df['last_repair_date'], errors='coerce')
        test_df['device_age_years'] = current_year - test_df['purchase_date'].dt.year
        test_df['days_since_last_repair'] = (datetime.now() - test_df['last_repair_date']).dt.days.fillna(-1).astype(int)

        for col in categorical_cols:
            test_df[col] = label_encoders[col].transform(test_df[col])

        X_test = test_df[feature_cols]

        test_df['Predicted Lifespan (years)'] = lifespan_model.predict(X_test)
        test_df['Disposal Ready (1=Yes)'] = disposal_model.predict(X_test)
        test_df['predicted_price'] = current_price_model.predict(X_test)
        test_df['predicted_dumb_price'] = dump_price_model.predict(X_test)


        st.session_state['predictions'] = test_df

        st.subheader("🔍 Prediction Results")
        st.dataframe(test_df[['device_id', 'Predicted Lifespan (years)', 'predicted_price', 'predicted_dumb_price']])

with tab2:
    st.header("🔧 Chat with RepairGPT to Fix Your Device")
    user_query = st.text_input("Describe your device issue:", "My laptop won't start.")

    if st.button("Ask RepairGPT"):
        if user_query:
            with st.spinner("Thinking..."):
                response = chat_with_gpt(user_query)
            st.success("Repair Suggestion:")
            st.markdown(response)



st.markdown("""
---
🌿 Supports Sustainable Campuses
""")
