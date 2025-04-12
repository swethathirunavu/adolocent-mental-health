import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Mental Health Predictor", layout="wide")

@st.cache_resource
def load_data_and_models():
    data = pd.read_csv("mental_health_dataset_with_labels.csv")

    X = data[['SUMSTRESS', 'SUMANXIETY', 'SUMDEPRESS', 'CVTOTAL']]

    y_stress = data['STRESSLEVELS'].map({
        'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Extremely severe': 4
    })
    y_anxiety = data['ANXIETYLEVELS'].map({
        'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Extremely severe': 4
    })
    y_depression = data['DEPRESSLEVELS'].map({
        'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Extremely severe': 4
    })

    stress_model = RandomForestClassifier().fit(X, y_stress)
    anxiety_model = RandomForestClassifier().fit(X, y_anxiety)
    depression_model = RandomForestClassifier().fit(X, y_depression)

    return {
        'stress_model': stress_model,
        'anxiety_model': anxiety_model,
        'depression_model': depression_model,
        'data': data
    }

def show_comparison_chart(user_vals, avg_vals):
    fig = go.Figure(data=[
        go.Bar(name='You', x=['Stress', 'Anxiety', 'Depression'], y=user_vals),
        go.Bar(name='Average', x=['Stress', 'Anxiety', 'Depression'], y=avg_vals)
    ])
    fig.update_layout(title='Your Levels vs. Dataset Average', barmode='group')
    st.plotly_chart(fig)

resources = load_data_and_models()
data = resources['data']

st.title("🧠 Mental Health Assessment")
st.write("This app predicts your mental health status and compares your scores with dataset averages.")

stress = st.slider("Your Stress Score", 0, 21, 5)
anxiety = st.slider("Your Anxiety Score", 0, 21, 5)
depression = st.slider("Your Depression Score", 0, 21, 5)
cvtotal = st.slider("Cyber Victimization Score", 0, 40, 10)

input_df = pd.DataFrame([[stress, anxiety, depression, cvtotal]],
                        columns=['SUMSTRESS', 'SUMANXIETY', 'SUMDEPRESS', 'CVTOTAL'])

if st.button("Analyze My Mental Health"):
    stress_lvl = resources['stress_model'].predict(input_df)[0]
    anxiety_lvl = resources['anxiety_model'].predict(input_df)[0]
    depression_lvl = resources['depression_model'].predict(input_df)[0]

    levels = ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely severe']
    st.subheader("🔍 Prediction Results")
    st.write(f"**Stress Level:** {levels[stress_lvl]}")
    st.write(f"**Anxiety Level:** {levels[anxiety_lvl]}")
    st.write(f"**Depression Level:** {levels[depression_lvl]}")

    # Show user vs dataset average chart
    show_comparison_chart(
        [stress, anxiety, depression],
        [
            round(data['SUMSTRESS'].mean(), 2),
            round(data['SUMANXIETY'].mean(), 2),
            round(data['SUMDEPRESS'].mean(), 2)
        ]
    )
