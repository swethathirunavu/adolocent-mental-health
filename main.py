import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import random

# --- Cached Data Loading and Model ---
@st.cache_resource
def load_data_and_models():
    data = pd.read_csv("mental_health_dataset_with_labels.csv")
    
    features = data[[
        'GENDER', 'AGE', 'MARITAL_STATUS', 'EDUCATION_LEVEL',
        'UNIVERSITY_STATUS', 'ACTIVE_SOCIAL_MEDIA', 'TIME_SPENT_SOCIAL_MEDIA',
        'CVTOTAL', 'CVPUBLICHUMILIATION', 'CVMALICE', 'CVUNWANTEDCONTACT',
        'MEANPUBLICHUMILIATION', 'MEANMALICE', 'MEANDECEPTION', 'MEANUNWANTEDCONTACT'
    ]]
    
    features = pd.get_dummies(features, columns=[
        'GENDER', 'AGE', 'MARITAL_STATUS', 'EDUCATION_LEVEL',
        'UNIVERSITY_STATUS', 'ACTIVE_SOCIAL_MEDIA', 'TIME_SPENT_SOCIAL_MEDIA'
    ])
    
    y_stress = data['STRESSLEVELS'].map({
        'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Extremely severe': 4
    })
    y_anxiety = data['ANXIETYLEVELS'].map({
        'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Extremely severe': 4
    })
    y_depress = data['Depression'].map({
        'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Extremely severe': 4
    })

    stress_model = RandomForestClassifier().fit(features, y_stress)
    anxiety_model = RandomForestClassifier().fit(features, y_anxiety)
    depression_model = RandomForestClassifier().fit(features, y_depress)
    
    return {
        'stress_model': stress_model,
        'anxiety_model': anxiety_model,
        'depression_model': depression_model,
        'feature_columns': features.columns
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
                "Bullying is never your fault. Talk to someone you trust and don't keep it in."
            ],
            "stress": [
                "Take a deep breath. You're doing the best you can, and that's enough.",
                "Pause, breathe, and remind yourself how far you've come. You've got this!"
            ],
            "anxiety": [
                "It's okay to feel anxious. Focus on one thing you can control right now.",
                "Try grounding techniques — they really help bring your mind back to the present."
            ],
            "depression": [
                "You are not alone. Even on dark days, your presence matters deeply.",
                "Small steps still count. Let's just take today one moment at a time."
            ],
            "default": [
                "I'm here for you. Want to talk about what's bothering you?",
                "It's okay to share. Sometimes saying it out loud makes a difference."
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
    severity = ["Normal", "Mild", "Moderate", "Severe", "Extremely severe"]

    if stress >= 3:
        suggestions.append("🌿 You might be feeling overwhelmed. Try journaling or speaking to a counselor.")
    elif stress >= 1:
        suggestions.append("🧘 Short breathing exercises or guided meditation can help reduce stress.")

    if anxiety >= 3:
        suggestions.append("💬 Talking to someone you trust can help manage anxious feelings.")
    elif anxiety >= 1:
        suggestions.append("🎧 Soothing music or mindfulness practices before bed can calm anxiety.")

    if depression >= 3:
        suggestions.append("🤍 Take small steps like a short walk or reading something uplifting.")
    elif depression >= 1:
        suggestions.append("📖 Writing down three things you're grateful for each day can help.")

    suggestions.append(f"💡 Your assessment: Stress - {severity[stress]}, Anxiety - {severity[anxiety]}, Depression - {severity[depression]}")
    suggestions.append("💪 Remember, progress takes time. You're stronger than you think.")
    return suggestions

# --- Main App ---
def main():
    st.set_page_config(page_title="MindCare App", layout="wide")
    resources = load_data_and_models()
    chatbot = MentalHealthChatbot()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Mental Health Assessment", "Chatbot", "Emergency Resources"])

    if page == "Home":
        st.title("🧠 MindCare: Your Mental Health Companion")
        st.subheader("We help assess your mental wellbeing based on your experiences")
        st.markdown("""
        This tool helps evaluate your mental health status based on various life factors and experiences.
        By answering questions about your demographics, social media use, and experiences,
        we can provide insights into your stress, anxiety, and depression levels.
        """)
        with st.expander("How it works"):
            st.write("""
            1. Go to **Mental Health Assessment**  
            2. Answer a few simple questions  
            3. Get personalized mental health insights  
            4. Chat with our emotional assistant 🤖  
            """)

    elif page == "Mental Health Assessment":
        st.title("📝 Mental Health Assessment")
        with st.form("form"):
            col1, col2 = st.columns(2)
            with col1:
                gender = st.selectbox("Gender", ["Female", "Male"])
                age = st.selectbox("Age", ["<18 years old", "19-24 years old", ">25 years old"])
                marital_status = st.selectbox("Marital Status", ["Single", "Married"])
            with col2:
                education = st.selectbox("Education Level", ["Diploma/Foundation", "Bachelor Degree", "Postgraduate studies"])
                university = st.selectbox("University Type", ["Private Universities", "Public Universities"])
                social_media = st.selectbox("Social Media Activity", ["Less active", "Active", "Very active"])
                social_media_time = st.selectbox("Time Spent on Social Media", ["1-2 hours", "3-6 hours", "7-10 hours", "> 11 hours", "Whole day"])
            cyber_exp = st.slider("Cyberbullying Experience (1-10)", 1, 10, 3)
            public_humiliation = st.slider("Public Humiliation (1-10)", 1, 10, 2)
            unwanted_contact = st.slider("Unwanted Contact (1-10)", 1, 10, 2)
            submit = st.form_submit_button("Get My Report")

        if submit:
            input_data = {
                'GENDER': gender, 'AGE': age, 'MARITAL_STATUS': marital_status,
                'EDUCATION_LEVEL': education, 'UNIVERSITY_STATUS': university,
                'ACTIVE_SOCIAL_MEDIA': social_media, 'TIME_SPENT_SOCIAL_MEDIA': social_media_time,
                'CVTOTAL': cyber_exp * 10, 'CVPUBLICHUMILIATION': public_humiliation * 5,
                'CVMALICE': cyber_exp * 5, 'CVUNWANTEDCONTACT': unwanted_contact * 5,
                'MEANPUBLICHUMILIATION': public_humiliation,
                'MEANMALICE': cyber_exp, 'MEANDECEPTION': cyber_exp,
                'MEANUNWANTEDCONTACT': unwanted_contact
            }

            input_df = pd.DataFrame([input_data])
            input_df = pd.get_dummies(input_df)
            for col in resources['feature_columns']:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[resources['feature_columns']]

            stress = resources['stress_model'].predict(input_df)[0]
            anxiety = resources['anxiety_model'].predict(input_df)[0]
            depression = resources['depression_model'].predict(input_df)[0]

            severity = ["Normal", "Mild", "Moderate", "Severe", "Extremely severe"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Stress Level", severity[stress])
            col2.metric("Anxiety Level", severity[anxiety])
            col3.metric("Depression Level", severity[depression])

            st.bar_chart(pd.DataFrame({
                "Category": ["Stress", "Anxiety", "Depression"],
                "Level": [stress, anxiety, depression]
            }).set_index("Category"))

            st.subheader("🌱 Recommendations")
            for rec in get_recommendations(stress, anxiety, depression):
                st.write(f"- {rec}")

    elif page == "Chatbot":
        st.title("💬 Chat with MindCare Bot")
        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Hi there! I'm here to listen and support you. How are you feeling today?"
            })

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_input = st.chat_input("Type your message...")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            reply = chatbot.respond(user_input)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.chat_message("user").write(user_input)
            st.chat_message("assistant").write(reply)

    elif page == "Emergency Resources":
        st.title("🚨 Emergency Mental Health Resources")
        st.markdown("""
        - **National Helpline**: 9152987821  
        - **iCall (TISS)**: +91 9152987821 | Email: icall@tiss.edu  
        - **AASRA Helpline**: 91-22-27546669 / 27546667  
        - **Vandrevala Foundation**: 1860 266 2345 / 1800 233 3330  
        """)
        st.info("If you're in immediate danger, contact emergency services or a trusted adult.")

# --- Run the app ---
if __name__ == "__main__":
    main()
