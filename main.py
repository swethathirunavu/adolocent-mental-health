import streamlit as st
import joblib
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Cached Data Loading and Model ---
@st.cache_resource
def load_data_and_models():
    data = pd.read_csv("mental_health_dataset_with_labels.csv")
    X = data[['SUMSTRESS', 'SUMANXIETY', 'SUMDEPRESS', 'CVTOTAL']]
    y_stress = data['STRESSLEVELS'].map({
        'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Extremely severe': 4
    })
    X_train, X_test, y_train, y_test = train_test_split(X, y_stress, test_size=0.2)
    model = RandomForestClassifier().fit(X_train, y_train)
    return {
        'stress_model': model,
        'anxiety_model': model,
        'depression_model': model
    }

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
def get_recommendations(stress, anxiety, depression):
    suggestions = []

    if stress >= 4:
        suggestions.append("🌿 You might be feeling overwhelmed. It's okay. Try journaling or speaking to a counselor.")
    elif stress >= 2:
        suggestions.append("🧘 Consider short breathing exercises or guided meditation daily.")

    if anxiety >= 4:
        suggestions.append("💬 Talking to someone can help. Even 10 minutes with a friend can make a difference.")
    elif anxiety >= 2:
        suggestions.append("🎧 Try listening to soothing music or practicing mindfulness before bed.")

    if depression >= 4:
        suggestions.append("🤍 You're strong for pushing through. Take a small step like taking a walk or reading something positive.")
    elif depression >= 2:
        suggestions.append("📖 Try writing down three small things you're grateful for today.")

    suggestions.append("💪 You’re doing better than you think. Progress takes time, and you're on your way.")
    return suggestions

# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="Mental Health Support", layout="wide")
    resources = load_data_and_models()
    chatbot = MentalHealthChatbot()

    st.sidebar.title("Menu")
    page = st.sidebar.radio("Go to", ["Home", "Assessment", "Chatbot", "Emergency Resources"])

    if page == "Home":
        st.title("🧠 Welcome to Mental Health Support System")
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
            col1, col2, col3 = st.columns(3)
            with col1:
                stress = st.slider("Stress Level (1-10)", 1, 10, 5)
            with col2:
                anxiety = st.slider("Anxiety Level (1-10)", 1, 10, 5)
            with col3:
                depression = st.slider("Depression Level (1-10)", 1, 10, 5)

            submitted = st.form_submit_button("Analyze")
            if submitted:
                input_df = pd.DataFrame({
                    'SUMSTRESS': [stress],
                    'SUMANXIETY': [anxiety],
                    'SUMDEPRESS': [depression],
                    'CVTOTAL': [5]
                })

                stress_lvl = resources['stress_model'].predict(input_df)[0]
                severity = ["Normal", "Mild", "Moderate", "Severe", "Extremely severe"]

                st.subheader("💡 Your Mental State Overview")
                cols = st.columns(3)
                for idx, label in enumerate(["Stress", "Anxiety", "Depression"]):
                    cols[idx].metric(label, severity[stress_lvl])

                st.subheader("🌼 Gentle Suggestions for You")
                for rec in get_recommendations(stress_lvl, stress_lvl, stress_lvl):
                    st.write(f"- {rec}")

    elif page == "Chatbot":
        st.title("💬 Talk to Your Support Chatbot")

        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Hi! I'm here to support you. Are you feeling insecure, anxious, or something else?"
            })

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if prompt := st.chat_input("How are you feeling today?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            response = chatbot.respond(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)

    elif page == "Emergency Resources":
        st.title("🛟 Emergency Resources - You're Not Alone")
        st.write("""
        ### 🌍 International Help Lines
        - **🇮🇳 India**: iCall – +91 9152987821
        - **🇺🇸 USA**: Suicide & Crisis Lifeline – Dial 988
        - **🇬🇧 UK**: Samaritans – 116 123
        - **🇨🇦 Canada**: Talk Suicide – 1-833-456-4566
        - **🇦🇺 Australia**: Lifeline – 13 11 14
        - **🇸🇬 Singapore**: Samaritans of Singapore (SOS) – 1800 221 4444
        - **🌐 Global**: [IASP Crisis Centres](https://www.iasp.info/resources/Crisis_Centres/)
        
        Please reach out — You deserve care, support, and love 💙
        """)

if __name__ == "__main__":
    main()
