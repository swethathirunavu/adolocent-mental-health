import streamlit as st
import pandas as pd
import joblib
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Load and Cache Data and Model ---
@st.cache_resource
def load_model():
    data = pd.read_csv("mental_health_dataset_with_labels.csv")
    # Preprocess data
    # Encode categorical variables
    data['Family_income'] = data['Family_income'].map({'Stable': 0, 'Decreased': 1})
    data['Do_you_work'] = data['Do_you_work'].map({'Yes': 1, 'No': 0})
    # Add other necessary preprocessing steps here

    # Define features and target
    features = ['Family_income', 'Do_you_work', 'Age', 'Education_Qualification', 'Physical_Activity']
    X = data[features]
    y = data['Depression_Level']  # Assuming this is the target column

    # Train model
    model = RandomForestClassifier()
    model.fit(X, y)
    return model, features

# --- Chatbot Class ---
class MentalHealthChatbot:
    def __init__(self):
        self.responses = {
            "hello": [
                "Hey there! I'm glad you're here. Want to talk about how you're feeling?",
                "Hi! You're not alone. I'm here to support you. How's your day going?"
            ],
            "insecure": [
                "You are more capable than you think. Everyone has their own pace and path.",
                "Remember, your worth isn't defined by others. You have something special in you!"
            ],
            "bully": [
                "I'm sorry to hear that. No one deserves to be bullied. You are strong for enduring this.",
                "Bullying is never your fault. Talk to someone you trust and don’t keep it in."
            ],
            "stress": [
                "Take a deep breath. You’re doing the best you can, and that’s enough.",
                "Pause, breathe, and remind yourself how far you’ve come. You’ve got this!"
            ],
            "anxiety": [
                "It’s okay to feel anxious. Focus on one thing you can control right now.",
                "Try grounding techniques — they really help bring your mind back to the present."
            ],
            "depression": [
                "You are not alone. Even on dark days, your presence matters deeply.",
                "Small steps still count. Let’s just take today one moment at a time."
            ],
            "default": [
                "I'm here for you. Want to talk about what’s bothering you?",
                "It’s okay to share. Sometimes saying it out loud makes a difference."
            ]
        }

    def respond(self, message):
        message = message.lower()
        for key in self.responses:
            if key in message:
                return random.choice(self.responses[key])
        return random.choice(self.responses["default"])

# --- Recommendations ---
def get_recommendations(level):
    suggestions = []
    if level == 'Severe':
        suggestions.append("🌿 You might be feeling overwhelmed. It's okay. Try journaling or speaking to a counselor.")
    elif level == 'Moderate':
        suggestions.append("🧘 Consider short breathing exercises or guided meditation daily.")
    elif level == 'Mild':
        suggestions.append("🎧 Try listening to soothing music or practicing mindfulness before bed.")
    else:
        suggestions.append("💪 You’re doing better than you think. Keep up the good work!")
    return suggestions

# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="Adolescence Wellness", layout="wide")
    model, features = load_model()
    chatbot = MentalHealthChatbot()

    st.sidebar.title("Menu")
    page = st.sidebar.radio("Go to", ["Home", "Assessment", "Chatbot", "Emergency Resources"])

    if page == "Home":
        st.title("🧠 Welcome to Adolescence Wellness")
        st.subheader("We care about how you feel.")

        with st.form("user_info"):
            name = st.text_input("Your Name")
            age = st.slider("Your Age", 10, 100, 18)
            gender = st.selectbox("Gender", ["Prefer not to say", "Female", "Male", "Other"])
            env = st.selectbox("Where are you coming from?", ["School", "College", "Workplace", "Other"])
            submitted = st.form_submit_button("Save & Continue")
            if submitted:
                st.success(f"Hello {name}, we’re here to support you on this journey. Let’s move forward together 💖")

    elif page == "Assessment":
        st.title("📊 Mental Health Assessment")
        with st.form("assessment_form"):
            col1, col2 = st.columns(2)
            with col1:
                family_income = st.selectbox("Family Income", ["Stable", "Decreased"])
                do_you_work = st.selectbox("Do you work?", ["Yes", "No"])
                age = st.slider("Your Age", 10, 100, 18)
            with col2:
                education = st.selectbox("Education Qualification", ["<= Secondary School", "Higher"])
                physical_activity = st.selectbox("Physical Activity", ["Inactive(<1/2 hour)", "Active(>1/2 hour)"])

            submitted = st.form_submit_button("Analyze")
            if submitted:
                # Encode inputs
                input_data = {
                    'Family_income': 0 if family_income == 'Stable' else 1,
                    'Do_you_work': 1 if do_you_work == 'Yes' else 0,
                    'Age': age,
                    'Education_Qualification': 0 if education == '<= Secondary School' else 1,
                    'Physical_Activity': 0 if physical_activity == 'Inactive(<1/2 hour)' else 1
                }
                input_df = pd.DataFrame([input_data])

                # Predict
                prediction = model.predict(input_df)[0]
                st.subheader("💡 Your Mental State Overview")
                st.write(f"Predicted Depression Level: {prediction}")

                st.subheader("🌼 Gentle Suggestions for You")
                for rec in get_recommendations(prediction):
                    st.write(f"- {rec}")

    elif page == "Chatbot":
        st.title("💬 Talk to Your Support Chatbot")

        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append({
                "role": "assistant
::contentReference[oaicite:8]{index=8}
 
