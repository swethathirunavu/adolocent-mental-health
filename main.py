import streamlit as st
import joblib
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Data Loading and Model Training (Cached) ---
@st.cache_resource
def load_data_and_models():
    # Load your dataset (updated path)
    data = pd.read_csv("mental_health_dataset_with_labels.csv")
    
    # Preprocessing
    X = data[['SUMSTRESS', 'SUMANXIETY', 'SUMDEPRESS', 'CVTOTAL']]
    y_stress = data['STRESSLEVELS'].map({
        'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Extremely severe': 4
    })
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y_stress, test_size=0.2)
    stress_model = RandomForestClassifier().fit(X_train, y_train)
    
    return {
        'stress_model': stress_model,
        'anxiety_model': stress_model,  # Replace with separate models if needed
        'depression_model': stress_model
    }

# --- Chatbot Class ---
class MentalHealthChatbot:
    def __init__(self):
        self.responses = {
            "hello": [
                "Hello! How are you feeling today?",
                "Hi there! What's on your mind?"
            ],
            "stress": [
                "Stress can feel overwhelming. Try the 4-7-8 breathing technique.",
                "When stressed, take a 5-minute walk outside. Nature helps reset your mind."
            ],
            "anxiety": [
                "Name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste.",
                "Anxiety lies to you. This feeling is temporary."
            ],
            "depression": [
                "You matter. Reach out to someone you trust today.",
                "Small wins count - even getting out of bed is progress."
            ],
            "default": [
                "I'm here to listen. Can you tell me more?",
                "That sounds difficult. Would coping strategies help?"
            ]
        }

    def respond(self, message):
        message = message.lower()
        if any(word in message for word in ["hi", "hello", "hey"]):
            return random.choice(self.responses["hello"])
        elif "stress" in message:
            return random.choice(self.responses["stress"])
        elif any(word in message for word in ["anxious", "anxiety"]):
            return random.choice(self.responses["anxiety"])
        elif any(word in message for word in ["depress", "sad", "hopeless"]):
            return random.choice(self.responses["depression"])
        else:
            return random.choice(self.responses["default"])

# --- Recommendation Engine ---
def get_recommendations(stress, anxiety, depression):
    recommendations = []

    if stress >= 4:
        recommendations.extend([
            "🚨 Contact a mental health professional immediately",
            "Practice box breathing (4s in, 4s hold, 4s out)"
        ])
    elif stress >= 2:
        recommendations.append("🧘 Try 10-minute guided meditation")

    if anxiety >= 4:
        recommendations.extend([
            "⚠️ Schedule a therapist appointment",
            "Limit caffeine and screen time before bed"
        ])

    if depression >= 4:
        recommendations.extend([
            "🆘 Reach out to a crisis hotline if needed",
            "Small step: Brush your teeth and change clothes today"
        ])

    recommendations.append("💡 Remember: Progress isn't linear")
    return recommendations

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Mental Health Support", layout="wide")
    
    # Load models
    resources = load_data_and_models()
    chatbot = MentalHealthChatbot()
    
    # Sidebar
    st.sidebar.title("Menu")
    page = st.sidebar.radio("Go to", ["Home", "Assessment", "Chatbot", "Emergency Resources"])
    
    # Home Page
    if page == "Home":
        st.title("Mental Health Support System")
        st.image("https://img.freepik.com/free-vector/mental-health-concept-illustration_114360-8924.jpg", width=400)
        st.write("""
        This app provides:
        - 📊 Mental health assessment using AI
        - 💬 24/7 support chatbot
        - 🛟 Emergency resources
        """)

    # Assessment Page
    elif page == "Assessment":
        st.title("Mental Health Assessment")
        
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
                input_data = pd.DataFrame({
                    'SUMSTRESS': [stress],
                    'SUMANXIETY': [anxiety],
                    'SUMDEPRESS': [depression],
                    'CVTOTAL': [5]  # Placeholder value
                })
                
                stress_lvl = resources['stress_model'].predict(input_data)[0]
                severity = ["Normal", "Mild", "Moderate", "Severe", "Extremely severe"]
                
                st.subheader("Results")
                cols = st.columns(3)
                cols[0].metric("Stress", severity[stress_lvl])
                cols[1].metric("Anxiety", severity[stress_lvl])
                cols[2].metric("Depression", severity[stress_lvl])
                
                st.subheader("Recommendations")
                for rec in get_recommendations(stress_lvl, stress_lvl, stress_lvl):
                    st.write(f"• {rec}")

    # Chatbot Page
    elif page == "Chatbot":
        st.title("Support Chatbot")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
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

    # Emergency Page
    elif page == "Emergency Resources":
        st.title("Emergency Help")
        st.write("""
        ### Immediate Crisis Support:
        - ☎️ **National Suicide Prevention Lifeline**: 988 (US)
        - 🌎 **International Association for Suicide Prevention**: [iasp.info/resources](https://www.iasp.info/resources/Crisis_Centres/)
        - 💬 **Crisis Text Line**: Text HOME to 741741 (US)
        """)
        st.image("https://img.freepik.com/free-vector/hand-drawn-mental-health-illustration_23-2149072258.jpg", width=300)

if __name__ == "__main__":
    main()
