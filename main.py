import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random

# --- Cached Data Loading and Model ---
@st.cache_resource
def load_data_and_models():
    data = pd.read_csv("mental_health_dataset_with_labels.csv")
    
    # Prepare features - using relevant columns from your dataset
    features = data[[
        'GENDER', 'AGE', 'MARITAL_STATUS', 'EDUCATION_LEVEL',
        'UNIVERSITY_STATUS', 'ACTIVE_SOCIAL_MEDIA', 'TIME_SPENT_SOCIAL_MEDIA',
        'CVTOTAL', 'CVPUBLICHUMILIATION', 'CVMALICE', 'CVUNWANTEDCONTACT',
        'MEANPUBLICHUMILIATION', 'MEANMALICE', 'MEANDECEPTION', 'MEANUNWANTEDCONTACT'
    ]]
    
    # Convert categorical features
    features = pd.get_dummies(features, columns=[
        'GENDER', 'AGE', 'MARITAL_STATUS', 'EDUCATION_LEVEL',
        'UNIVERSITY_STATUS', 'ACTIVE_SOCIAL_MEDIA', 'TIME_SPENT_SOCIAL_MEDIA'
    ])
    
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

    if stress >= 3:  # Moderate or higher
        suggestions.append("🌿 You might be feeling overwhelmed. Try journaling or speaking to a counselor.")
    elif stress >= 1:  # Mild
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

# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="Mental Health Support", layout="wide")
    resources = load_data_and_models()
    chatbot = MentalHealthChatbot()

    st.sidebar.title("Menu")
    page = st.sidebar.radio("Go to", ["Home", "Mental Health Assessment", "Chatbot", "Emergency Resources"])

    if page == "Home":
        st.title("🧠 MindCare: Your Mental Health Companion")
        st.subheader("We help assess your mental wellbeing based on your experiences")
        
        st.write("""
        This tool helps evaluate your mental health status based on various life factors and experiences.
        By answering questions about your demographics, social media use, and experiences, we can provide
        insights into your stress, anxiety, and depression levels.
        """)
        
        with st.expander("How it works"):
            st.write("""
            1. Go to the **Mental Health Assessment** page
            2. Answer questions about your experiences
            3. Our system will analyze your responses
            4. Receive personalized insights and recommendations
            """)

    elif page == "Mental Health Assessment":
        st.title("📝 Mental Health Assessment")
        st.write("Please answer these questions to help us understand your experiences and mental wellbeing.")

        with st.form("assessment_form"):
            st.subheader("Personal Information")
            col1, col2 = st.columns(2)
            
            with col1:
                gender = st.selectbox("Gender", ["Female", "Male"])
                age = st.selectbox("Age", ["<18 years old", "19-24 years old", ">25 years old"])
                marital_status = st.selectbox("Marital Status", ["Single", "Married"])
                
            with col2:
                education = st.selectbox("Education Level", ["Diploma/Foundation", "Bachelor Degree", "Postgraduate studies"])
                university = st.selectbox("University Type", ["Private Universities", "Public Universities"])
            
            st.subheader("Social Media & Online Experiences")
            col1, col2 = st.columns(2)
            
            with col1:
                social_media = st.selectbox("How active are you on social media?", 
                                          ["Less active", "Active", "Very active"])
                social_media_time = st.selectbox("Daily time spent on social media", 
                                               ["1-2 hours", "3-6 hours", "7-10 hours", "> 11 hours", "Whole day"])
                
            with col2:
                cyber_exp = st.slider("How often have you experienced cyber issues? (1-10)", 1, 10, 3)
                public_humiliation = st.slider("Experienced public humiliation online? (1-10)", 1, 10, 1)
                unwanted_contact = st.slider("Received unwanted online contact? (1-10)", 1, 10, 1)
            
            submitted = st.form_submit_button("Evaluate My Mental Health")
            
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
                    'CVTOTAL': cyber_exp * 10,  # Scale to match dataset
                    'CVPUBLICHUMILIATION': public_humiliation * 5,
                    'CVMALICE': cyber_exp * 5,
                    'CVUNWANTEDCONTACT': unwanted_contact * 5,
                    'MEANPUBLICHUMILIATION': public_humiliation,
                    'MEANMALICE': cyber_exp,
                    'MEANDECEPTION': cyber_exp,
                    'MEANUNWANTEDCONTACT': unwanted_contact
                }
                
                # Convert to DataFrame
                input_df = pd.DataFrame([input_data])
                input_df = pd.get_dummies(input_df)
                
                # Ensure all feature columns are present
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
                chart_data = pd.DataFrame({
                    "Category": ["Stress", "Anxiety", "Depression"],
                    "Level": [stress_pred, anxiety_pred, depression_pred]
                })
                st.bar_chart(chart_data.set_index("Category"), height=300)
                
                st.subheader("🌱 Personalized Recommendations")
                for rec in get_recommendations(stress_pred, anxiety_pred, depression_pred):
                    st.write(f"- {rec}")

    elif page == "Chatbot":
        st.title("💬 MindCare Chat Companion")
        st.write("Talk to our supportive chatbot about how you're feeling")

        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Hi there! I'm here to listen and support you. How are you feeling today?"
            })

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if prompt := st.chat_input("Type your message here..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            response = chatbot.respond(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)

    elif page == "Emergency Resources":
        st.title("🆘 Immediate Help Resources")
        st.write("If you're in crisis or need immediate support, please reach out to these resources:")
        
        st.subheader("🌎 International Helplines")
        cols = st.columns(2)
        
        with cols[0]:
            st.write("""
            - **India**: iCall – +91 9152987821
            - **USA**: Suicide & Crisis Lifeline – Dial 988
            - **UK**: Samaritans – 116 123
            - **Canada**: Talk Suicide – 1-833-456-4566
            """)
            
        with cols[1]:
            st.write("""
            - **Australia**: Lifeline – 13 11 14
            - **Singapore**: SOS – 1800 221 4444
            - **Global**: [IASP Crisis Centres](https://www.iasp.info/resources/Crisis_Centres/)
            """)
        
        st.subheader("📱 Text/Chat Support")
        st.write("""
        - **Crisis Text Line**: Text HOME to 741741 (US/UK/Canada)
        - [7 Cups](https://www.7cups.com/): Free online chat with trained listeners
        """)
        
        st.subheader("💡 Remember")
        st.write("""
        - You are not alone in this
        - Reaching out is a sign of strength
        - Your feelings are valid and important
        """)

if __name__ == "__main__":
    main()
