import streamlit as st
import pandas as pd
import joblib

# Load ML model (replace with your actual filename)
model = joblib.load("mental_health_model.pkl")

# Function to get suggestions based on state
def get_supportive_suggestions(state):
    suggestions = {
        "Healthy": "Keep up the positive habits! Stay connected and take breaks 🌼",
        "Mild Stress": "Take deep breaths, try journaling, and talk to a friend 🧘‍♀️",
        "Depressed": "You're not alone. Please reach out to a counselor or loved one ❤️",
        "Highly Depressed": "Seek help immediately. Contact a mental health professional or helpline ☎️",
    }
    return suggestions.get(state, "Stay strong. We are here for you 💛")

# App UI
st.set_page_config(page_title="Adolescent Mental Health Support", page_icon="🧠")

st.title("🧠 Adolescent Mental Health Prediction & Support")

# Sidebar navigation
page = st.sidebar.selectbox("Choose a page", ["Home", "Chat Support"])

if page == "Home":
    st.header("📊 Predict Stress/Depression Levels")

    option = st.radio("Choose Input Type", ["Manual Entry", "Upload CSV"])

    if option == "Manual Entry":
        age = st.slider("Age", 10, 25, 16)
        cyberbullying = st.selectbox("Cyberbullying Experience", ["Yes", "No"])
        anxiety = st.slider("Anxiety Level (0-10)", 0, 10, 5)
        insecurity = st.slider("Insecurity Level (0-10)", 0, 10, 5)

        if st.button("Predict"):
            input_df = pd.DataFrame([{
                "age": age,
                "cyberbullying": 1 if cyberbullying == "Yes" else 0,
                "anxiety": anxiety,
                "insecurity": insecurity
            }])

            prediction = model.predict(input_df)[0]
            st.success(f"🧠 Mental Health Status: **{prediction}**")
            st.info(get_supportive_suggestions(prediction))

    else:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            predictions = model.predict(data)
            data["Prediction"] = predictions
            st.write(data)
            st.info("Suggestions for first row:")
            st.success(get_supportive_suggestions(predictions[0]))

elif page == "Chat Support":
    st.header("💬 Chat with Your Support Buddy")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hi there! I'm your support buddy 🤗. How are you feeling today?"
        })

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

    user_input = st.chat_input("Tell me how you’re feeling...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Dummy response (replace with sentiment analysis or LLM later)
        response = "Thanks for sharing. I'm here with you 💛"
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").markdown(response)

 
