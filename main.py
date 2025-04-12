import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import random
import os

# --- Cached Data Loading and Model ---

@st.cache_data
def load_data_and_models():
    # Check if dataset exists
    if not os.path.exists("mental_health_dataset_with_labels.csv"):
        raise FileNotFoundError("mental_health_dataset_with_labels.csv not found in the current directory")

    # Load dataset
    df1 = pd.read_csv("mental_health_dataset_with_labels.csv")
    
    # Drop NaN rows (or use df1.fillna(0) if you prefer filling)
    df1 = df1.dropna(subset=['Depression', 'Stress', 'Anxiety'])
    
    # Separate features and targets
    features = df1.drop(['Depression', 'Stress', 'Anxiety', 'Bullied', 'Insecure'], axis=1, errors='ignore')
    
    # Convert categorical columns to string type to avoid encoding errors
    for col in features.select_dtypes(include=['object']).columns:
        features[col] = features[col].astype(str)
    
    # Create dummy variables to ensure consistent columns
    features_encoded = pd.get_dummies(features)
    feature_columns = features_encoded.columns.tolist()
    
    # Train models for the three conditions used in the assessment
    depression_model = RandomForestClassifier(random_state=42).fit(features_encoded, df1['Depression'])
    stress_model = RandomForestClassifier(random_state=42).fit(features_encoded, df1['Stress'])
    anxiety_model = RandomForestClassifier(random_state=42).fit(features_encoded, df1['Anxiety'])
    
    # Get the unique values from the target columns to understand their range
    depression_values = df1['Depression'].unique().tolist()
    stress_values = df1['Stress'].unique().tolist()
    anxiety_values = df1['Anxiety'].unique().tolist()
    
    return {
        'depression_model': depression_model,
        'stress_model': stress_model, 
        'anxiety_model': anxiety_model,
        'feature_columns': feature_columns,
        'data': df1,
        'depression_values': depression_values,
        'stress_values': stress_values,
        'anxiety_values': anxiety_values
    }


# --- Chatbot Class ---
class MentalHealthChatbot:
    def __init__(self):
        self.responses = {
            "hello": [
                "Hey there! I'm glad you're here. Want to talk about how you're feeling?",
                "Hi! You're not alone. I'm here to support you. How's your day going?"
            ],
            "i am insecure": [
                "You are more capable than you think. Everyone has their own pace and path.",
                "Remember, your worth isn't defined by others. You have something special in you!"
            ],
            "everyone are bullying me": [
                "I'm sorry to hear that. No one deserves to be bullied. You are strong for enduring this.",
                "Bullying is never your fault. Talk to someone you trust and don't keep it in."
            ],
            "i am very stressed": [
                "Take a deep breath. You're doing the best you can, and that's enough.",
                "Pause, breathe, and remind yourself how far you've come. You've got this!"
            ],
            "i have anxiety": [
                "It's okay to feel anxious. Focus on one thing you can control right now.",
                "Try grounding techniques — they really help bring your mind back to the present."
            ],
            "i am feeling depressed": [
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
    
    # Make sure the indices are within valid range
    stress_index = min(max(stress - 1, 0), len(severity)-1)  # Adjust by -1 since your data uses 1-5 but indices are 0-4
    anxiety_index = min(max(anxiety - 1, 0), len(severity)-1)
    depression_index = min(max(depression - 1, 0), len(severity)-1)

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

    suggestions.append(f"💡 Your assessment: Stress - {severity[stress_index]}, Anxiety - {severity[anxiety_index]}, Depression - {severity[depression_index]}")
    suggestions.append("💪 Remember, progress takes time. You're stronger than you think.")
    return suggestions

# --- Input Data Preprocessing ---
def preprocess_input(input_data, feature_columns):
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Convert categorical columns to string to match training data
    for col in input_df.select_dtypes(include=['object']).columns:
        input_df[col] = input_df[col].astype(str)
    
    # Create dummy variables
    input_df = pd.get_dummies(input_df)
    
    # Handle missing columns
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Keep only columns used in training
    missing_cols = set(feature_columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    
    return input_df[feature_columns]

# --- Main App ---
def main():
    st.set_page_config(page_title="MindCare App", layout="wide")
    
    try:
        resources = load_data_and_models()
        st.session_state["resources"] = resources
    except FileNotFoundError:
        st.error("ERROR: The mental_health_dataset_with_labels.csv file was not found. Please make sure it's in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"ERROR loading data and models: {e}")
        st.stop()
        
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
            try:
                # Create input data dictionary matching your dataset structure
                input_data = {
                    'GENDER': gender, 
                    'AGE': age, 
                    'MARITAL_STATUS': marital_status,
                    'EDUCATION_LEVEL': education, 
                    'UNIVERSITY_STATUS': university,
                    'ACTIVE_SOCIAL_MEDIA': social_media, 
                    'TIME_SPENT_SOCIAL_MEDIA': social_media_time,
                    'CVTOTAL': cyber_exp * 10, 
                    'CVPUBLICHUMILIATION': public_humiliation * 5,
                    'CVMALICE': cyber_exp * 5, 
                    'CVUNWANTEDCONTACT': unwanted_contact * 5,
                    'MEANPUBLICHUMILIATION': public_humiliation,
                    'MEANMALICE': cyber_exp, 
                    'MEANDECEPTION': cyber_exp,
                    'MEANUNWANTEDCONTACT': unwanted_contact
                }

                # Preprocess input
                input_df = preprocess_input(input_data, resources['feature_columns'])
                
                # Add error handling for predictions
                try:
                    # Get predictions
                    stress_pred = resources['stress_model'].predict(input_df)[0]
                    anxiety_pred = resources['anxiety_model'].predict(input_df)[0]
                    depression_pred = resources['depression_model'].predict(input_df)[0]
                    
                    # Convert to integers if needed
                    stress = int(stress_pred)
                    anxiety = int(anxiety_pred)
                    depression = int(depression_pred)
                    
                    # Show the predicted levels
                    severity = ["Normal", "Mild", "Moderate", "Severe", "Extremely severe"]
                    stress_index = min(max(stress - 1, 0), len(severity)-1)  # Adjust for 1-based indices
                    anxiety_index = min(max(anxiety - 1, 0), len(severity)-1)
                    depression_index = min(max(depression - 1, 0), len(severity)-1)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Stress Level", severity[stress_index])
                    col2.metric("Anxiety Level", severity[anxiety_index])
                    col3.metric("Depression Level", severity[depression_index])

                    # Show bar chart
                    chart_data = pd.DataFrame({
                        "Category": ["Stress", "Anxiety", "Depression"],
                        "Level": [stress, anxiety, depression]
                    }).set_index("Category")
                    st.bar_chart(chart_data)

                    # Show recommendations
                    st.subheader("🌱 Recommendations")
                    for rec in get_recommendations(stress, anxiety, depression):
                        st.write(f"- {rec}")
                    
                    # Show the raw prediction values for debugging
                    with st.expander("Debug Information"):
                        st.write(f"Raw predictions: Stress={stress_pred}, Anxiety={anxiety_pred}, Depression={depression_pred}")
                    
                except (IndexError, ValueError) as e:
                    st.error(f"Error with model predictions: {e}")
                    st.write("Debug info:")
                    st.write(f"Raw predictions: Stress={resources['stress_model'].predict(input_df)[0]}, " 
                             f"Anxiety={resources['anxiety_model'].predict(input_df)[0]}, "
                             f"Depression={resources['depression_model'].predict(input_df)[0]}")
                    st.write(f"Expected value ranges: Depression={resources['depression_values']}, " 
                             f"Anxiety={resources['anxiety_values']}, Stress={resources['stress_values']}")
            
            except Exception as e:
                st.error(f"An error occurred: {e}. Please ensure your dataset structure matches the expected format.")
                st.error("If this is your first time running the app, please check if the dataset file exists.")

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
