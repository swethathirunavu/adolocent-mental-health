import streamlit as st
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Load ML Model ---
@st.cache_resource
def load_model_and_features():
    df = pd.read_csv("mental_health_dataset_with_labels.csv")

    feature_cols = [
        "Are_you_worried", "Are_you_relaxed", "Are_you_restless", "Are_you_annoyed", "Are_you_afraid",
        "How_enthusiastic", "Are_you_hopeless", "Sleep_cycle", "Are_you_tired", "Appetite",
        "Regret_pregnancy", "Focus_level", "Isolation_level", "Permissiveness", "Health_issues",
        "Family_income", "Physical_activity"
    ]

    label = "Depression_Level"
    df["Family_income"] = df["Family_income"].map({"Stable": 0, "Decreased": 1})
    df["Physical_activity"] = df["Physical_activity"].map({"Inactive(<1/2 hour)": 0, "Active(>1/2 hour)": 1})

    X, y = df[feature_cols], df[label]
    model = RandomForestClassifier().fit(*train_test_split(X, y, test_size=0.2)[::2])
    return model, feature_cols

# --- Smart Suggestions ---
def get_recommendations(level):
    tips = []
    if level == "Severe":
        tips = [
            "⚠️ It looks like you're going through a tough time. Please talk to a trusted adult or mental health professional.",
            "📞 If you're in India, call 9152987821 (iCall) or 9152987821 (Vandrevala Foundation). You're not alone.",
            "🫂 Reach out to a friend. You deserve support and healing.",
        ]
    elif level == "Moderate":
        tips = [
            "📝 Try journaling your emotions daily. Writing can help clarify your thoughts.",
            "💬 Talk to a friend or counselor about what you're feeling.",
            "🚶‍♀️ Get outside for a short walk and fresh air—it really helps!",
        ]
    elif level == "Mild":
        tips = [
            "🌼 Practice deep breathing or light meditation.",
            "🎨 Engage in something creative or playful to lift your mood.",
        ]
    elif level == "Normal":
        tips = [
            "🌞 Keep up the good self-care! Continue your healthy habits.",
            "🧘 Stay consistent with things that make you feel grounded.",
        ]
    tips.append("💖 You matter. Keep choosing small acts of self-kindness.")
    return tips

# --- Affirmation Messages ---
def get_affirmation():
    affirmations = [
        "🌟 You are doing your best, and that is enough.",
        "🌈 Brighter days are ahead. Just breathe.",
        "🧡 Your mental health matters. You're not alone.",
        "✨ Healing isn't linear. Be patient with yourself."
    ]
    return random.choice(affirmations)

# --- Friendly Chatbot ---
def seraphina_chat(user_input):
    responses = {
        "hello": "Hey there! I'm Seraphina. How are you feeling today?",
        "sad": "I'm sorry you're feeling this way. Want to talk about it?",
        "help": "I'm here for you. You can also reach out to someone you trust ❤️",
        "thank you": "Anytime. You're not alone 🌻",
    }
    for key in responses:
        if key in user_input.lower():
            return responses[key]
    return "I'm here to listen and support you. Tell me more 💬"

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Seraphina: Your Mental Health Ally", layout="wide")
    st.title("🌿 Seraphina: Your Insight Partner")
    st.write("Fill out the form below. I'll help assess your emotional state and support you with gentle guidance 💛")

    model, feature_cols = load_model_and_features()

    with st.form("wellness_form"):
        st.header("📋 Self-Check Assessment")

        cols = st.columns(3)
        input_data = {}
        inputs = {
            "Are_you_worried": [1, 2, 3], "Are_you_relaxed": [1, 2, 3],
            "Are_you_restless": [1, 2, 3], "Are_you_annoyed": [1, 2, 3],
            "Are_you_afraid": [1, 2, 3], "How_enthusiastic": [1, 2, 3],
            "Are_you_hopeless": [1, 2, 3], "Sleep_cycle": [1, 2, 3],
            "Are_you_tired": [1, 2, 3], "Appetite": [1, 2, 3],
            "Regret_pregnancy": [1, 2, 3], "Focus_level": [1, 2, 3],
            "Isolation_level": [1, 2, 3], "Permissiveness": [1, 2, 3],
        }

        for i, (key, options) in enumerate(inputs.items()):
            input_data[key] = cols[i % 3].select_slider(key.replace("_", " "), options=options)

        input_data["Health_issues"] = cols[0].select_slider("Do you have health issues?", options=[0, 1])
        input_data["Family_income"] = 1 if cols[1].selectbox("Family Income", ["Stable", "Decreased"]) == "Decreased" else 0
        input_data["Physical_activity"] = 1 if cols[2].selectbox("Physical Activity", ["Inactive(<1/2 hour)", "Active(>1/2 hour)"]) == "Active(>1/2 hour)" else 0

        submitted = st.form_submit_button("💡 Analyze Me")
        if submitted:
            input_df = pd.DataFrame([input_data])
            result = model.predict(input_df)[0]

            st.success(f"🧠 Detected Level: **{result}**")
            st.subheader("🌻 Seraphina's Suggestions")
            for rec in get_recommendations(result):
                st.write("- " + rec)

            st.info(get_affirmation())

    st.divider()

    st.subheader("💬 Talk to Seraphina")
    chat = st.chat_input("Say something...")
    if chat:
        reply = seraphina_chat(chat)
        st.write(f"**You:** {chat}")
        st.write(f"**Seraphina:** {reply}")

# Run the app
if __name__ == "__main__":
    main()
