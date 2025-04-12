import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random

st.set_page_config(page_title="Adolescence Wellness", layout="wide")
st.markdown("<h1 style='text-align: center;'>Adolescence Wellness 💫</h1>", unsafe_allow_html=True)

# Load model and feature columns
@st.cache_resource
def load_model_and_features():
    model = joblib.load("model.pkl")
    df = pd.read_csv("mental_health_dataset_with_labels.csv")

    if "Family_income" in df.columns:
        df["Family_income"] = df["Family_income"].map({"Stable": 0, "Decreased": 1})
    elif "Family income" in df.columns:
        df["Family_income"] = df["Family income"].map({"Stable": 0, "Decreased": 1})

    if "Physical_activity" in df.columns:
        df["Physical_activity"] = df["Physical_activity"].map({"High": 0, "Low": 1})
    elif "Physical activity" in df.columns:
        df["Physical_activity"] = df["Physical activity"].map({"High": 0, "Low": 1})

    feature_cols = ['Age', 'Gender', 'Sleep_hours', 'Social_media_usage',
                    'Family_income', 'Physical_activity', 'School_support']
    return model, feature_cols

# Recommendations
def get_suggestions(prediction):
    if prediction == 0:
        return "🟢 You're doing well! Keep maintaining your healthy routine. 😊"
    elif prediction == 1:
        return "🟡 You may be under moderate stress. Try relaxation techniques, talk to a friend, and get enough rest. 🌿"
    else:
        return "🔴 High mental distress detected. It's okay to ask for help. Talk to a trusted adult or counselor. ❤️"

# Emergency support
def emergency_help():
    st.markdown(\"\"\"
    ### 🚨 Emergency Tips
    - Contact a trusted teacher, parent, or counselor 📞
    - Call your local child helpline 📱
    - Take 3 deep breaths and drink water 🧘
    - Write down your thoughts in a journal 📓
    - You matter. Never hesitate to seek help 💗
    \"\"\")

# Chatbot responses
def chatbot_response(user_input):
    user_input = user_input.lower()
    if "sad" in user_input or "depressed" in user_input:
        return "I'm really sorry you're feeling this way. You're not alone. 💛"
    elif "happy" in user_input:
        return "That's wonderful! Keep spreading joy. 🌞"
    elif "help" in user_input:
        return "I'm here to help. Tell me more so I can support you."
    else:
        return random.choice([
            "I'm here for you 💬",
            "You're stronger than you think.",
            "Would you like to talk about your day?",
            "Everything will be okay 💫"
        ])

# Main function
def main():
    model, feature_cols = load_model_and_features()

    menu = ["Prediction Tool", "Chatbot Friend", "Emergency"]
    choice = st.sidebar.radio("Choose a feature", menu)

    if choice == "Prediction Tool":
        st.subheader("🧠 Mental Health Risk Prediction")

        age = st.slider("Age", 10, 19)
        gender = st.selectbox("Gender", ["Male", "Female"])
        sleep = st.slider("Average Sleep Hours", 0.0, 12.0, step=0.5)
        social = st.slider("Social Media Usage (hours/day)", 0.0, 10.0, step=0.5)
        family = st.selectbox("Family Income Status", ["Stable", "Decreased"])
        activity = st.selectbox("Physical Activity Level", ["High", "Low"])
        support = st.slider("School Support Rating", 1, 10)

        input_df = pd.DataFrame({
            'Age': [age],
            'Gender': [0 if gender == "Male" else 1],
            'Sleep_hours': [sleep],
            'Social_media_usage': [social],
            'Family_income': [0 if family == "Stable" else 1],
            'Physical_activity': [0 if activity == "High" else 1],
            'School_support': [support]
        })

        if st.button("Predict"):
            prediction = model.predict(input_df)[0]
            label = ["Low Risk", "Moderate Risk", "High Risk"][prediction]
            st.success(f"Prediction: {label}")
            st.info(get_suggestions(prediction))

    elif choice == "Chatbot Friend":
        st.subheader("💬 Talk to Your Virtual Friend")
        user_input = st.text_input("How are you feeling?")
        if st.button("Send"):
            if user_input.strip():
                response = chatbot_response(user_input)
                st.markdown(f"**Bot:** {response}")

    elif choice == "Emergency":
        st.subheader("🚨 Emergency Help")
        emergency_help()

if __name__ == "__main__":
    main()
