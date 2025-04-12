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
    
    # Prepare features and targets
    features = data[['GENDER', 'AGE', 'MARITAL_STATUS', 'EDUCATION_LEVEL', 
                    'UNIVERSITY_STATUS', 'ACTIVE_SOCIAL_MEDIA', 'TIME_SPENT_SOCIAL_MEDIA',
                    'CVTOTAL', 'CVPUBLICHUMILIATION', 'CVMALICE', 'CVUNWANTEDCONTACT']]
    
    # Convert categorical features
    features = pd.get_dummies(features, columns=['GENDER', 'AGE', 'MARITAL_STATUS', 
                                               'EDUCATION_LEVEL', 'UNIVERSITY_STATUS',
                                               'ACTIVE_SOCIAL_MEDIA', 'TIME_SPENT_SOCIAL_MEDIA'])
    
    # Prepare targets
    y_stress = data['STRESSLEVELS'].map({
        'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Extremely severe': 4
    })
    y_anxiety = data['ANXIETYLEVELS'].map({
        'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Extremely severe': 4
    })
    y_depress = data['Depression'].map({
        'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Extremely severe': 4
    })
    
    # Train models
    X_train, X_test, y_train, y_test = train_test_split(features, y_stress, test_size=0.2)
    stress_model = RandomForestClassifier().fit(X_train, y_train)
    
    X_train, X_test, y_train, y_test = train_test_split(features, y_anxiety, test_size=0.2)
    anxiety_model = RandomForestClassifier().fit(X_train, y_train)
    
    X_train, X_test, y_train, y_test = train_test_split(features, y_depress, test_size=0.2)
    depression_model = RandomForestClassifier().fit(X_train, y_train)
    
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

    if stress >= 3:  # Moderate or higher
        suggestions.append("🌿 You might be feeling overwhelmed. It's okay. Try journaling or speaking to a counselor.")
    elif stress >= 1:  # Mild
        suggestions.append("🧘 Consider short breathing exercises or guided meditation daily.")

    if anxiety >= 3:
        suggestions.append("💬 Talking to someone can help. Even 10 minutes with a friend can make a difference.")
    elif anxiety >= 1:
        suggestions.append("🎧 Try listening to soothing music or practicing mindfulness before bed.")

    if depression >= 3:
        suggestions.append("🤍 You're strong for pushing through. Take a small step like taking a walk or reading something positive.")
    elif depression >= 1:
        suggestions.append("📖 Try writing down three small things you're grateful for today.")

    suggestions.append(f"💪 Your current levels - Stress: {severity[stress]}, Anxiety: {severity[anxiety]}, Depression: {severity[depression]}")
    suggestions.append("💪 You're doing better than you think. Progress takes time, and you're on your way.")
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
            age = st.selectbox("Age", ["<18 years old", "19-24 years old", ">25 years old"])
            gender = st.selectbox("Gender", ["Female", "Male"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
            education = st.selectbox("Education Level", ["Diploma/Foundation", "Bachelor Degree", "Postgraduate studies"])
            university = st.selectbox("University Type", ["Private Universities", "Public Universities"])
            social_media = st.selectbox("Social Media Activity", ["Less active", "Active", "Very active"])
            social_media_time = st.selectbox("Time Spent on Social Media", ["1-2 hours", "3-6 hours", "7-10 hours", "> 11 hours", "Whole day"])
            
            submitted = st.form_submit_button("Save & Continue")
            if submitted:
                st.success(f"Hello {name}, we're here to support you on this journey. Let's move forward together 💖")

    elif page == "Assessment":
        st.title("📊 Mental Health Assessment")
        st.write("Please provide some information about your experiences to help us understand your mental health status.")
        
        with st.form("assessment_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Personal Information")
                age = st.selectbox("Age", ["<18 years old", "19-24 years old", ">25 years old"])
                gender = st.selectbox("Gender", ["Female", "Male"])
                marital_status = st.selectbox("Marital Status", ["Single", "Married"])
                education = st.selectbox("Education Level", ["Diploma/Foundation", "Bachelor Degree", "Postgraduate studies"])
                university = st.selectbox("University Type", ["Private Universities", "Public Universities"])
                
            with col2:
                st.subheader("Social Media & Cyber Experiences")
                social_media = st.selectbox("Social Media Activity", ["Less active", "Active", "Very active"])
                social_media_time = st.selectbox("Time Spent on Social Media", ["1-2 hours", "3-6 hours", "7-10 hours", "> 11 hours", "Whole day"])
                cv_total = st.slider("Cyber Victimization Experiences (0-100)", 0, 100, 15)
                cv_public = st.slider("Public Humiliation Experiences (0-50)", 0, 50, 2)
                cv_malice = st.slider("Malicious Behavior Experiences (0-50)", 0, 50, 5)
                cv_contact = st.slider("Unwanted Contact Experiences (0-50)", 0, 50, 8)
            
            submitted = st.form_submit_button("Analyze My Mental Health")
            
            if submitted:
                # Prepare input data
                input_data = {
                    'GENDER': gender,
                    'AGE': age,
                    'MARITAL_STATUS': marital_status,
                    'EDUCATION_LEVEL': education,
                    'UNIVERSITY_STATUS': university,
                    'ACTIVE_SOCIAL_MEDIA': social_media,
                    'TIME_SPENT_SOCIAL_MEDIA': social_media_time,
                    'CVTOTAL': cv_total,
                    'CVPUBLICHUMILIATION': cv_public,
                    'CVMALICE': cv_malice,
                    'CVUNWANTEDCONTACT': cv_contact
                }
                
                # Convert to DataFrame
                input_df = pd.DataFrame([input_data])
                
                # One-hot encode categorical variables
                input_df = pd.get_dummies(input_df)
                
                # Ensure all feature columns are present (add missing with 0)
                for col in resources['feature_columns']:
                    if col not in input_df.columns:
                        input_df[col] = 0
                
                # Reorder columns to match training data
                input_df = input_df[resources['feature_columns']]
                
                # Make predictions
                stress_pred = resources['stress_model'].predict(input_df)[0]
                anxiety_pred = resources['anxiety_model'].predict(input_df)[0]
                depression_pred = resources['depression_model'].predict(input_df)[0]
                
                severity = ["Normal", "Mild", "Moderate", "Severe", "Extremely severe"]
                
                st.subheader("🔍 Your Mental Health Assessment Results")
                
                cols = st.columns(3)
                cols[0].metric("Stress Level", severity[stress_pred])
                cols[1].metric("Anxiety Level", severity[anxiety_pred])
                cols[2].metric("Depression Level", severity[depression_pred])
                
                st.subheader("📊 Results Visualization")
                graph_df = pd.DataFrame({
                    "Category": ["Stress", "Anxiety", "Depression"],
                    "Score": [stress_pred, anxiety_pred, depression_pred]
                })
                st.bar_chart(graph_df.set_index("Category"))
                
                st.subheader("🌱 Personalized Recommendations")
                for rec in get_recommendations(stress_pred, anxiety_pred, depression_pred):
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

 
