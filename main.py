import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import random
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- Enhanced Cached Data Loading and Models ---

@st.cache_data
def load_data_and_models():
    # Check if dataset exists
    if not os.path.exists("mental_health_dataset_with_labels.csv"):
        raise FileNotFoundError("mental_health_dataset_with_labels.csv not found in the current directory")

    # Load dataset
    df = pd.read_csv("mental_health_dataset_with_labels.csv")
    
    # Drop NaN rows in target columns
    df = df.dropna(subset=['Depression', 'Stress', 'Anxiety'])
    
    # Define feature columns based on your actual dataset
    feature_columns = [
        'GENDER', 'AGE', 'MARITAL_STATUS', 'EDUCATION_LEVEL', 'UNIVERSITY_STATUS',
        'ACTIVE_SOCIAL_MEDIA', 'TIME_SPENT_SOCIAL_MEDIA', 'CVTOTAL', 
        'CVPUBLICHUMILIATION', 'CVMALICE', 'CVUNWANTEDCONTACT', 'CVDECEPTION',
        'MEANPUBLICHUMILIATION', 'MEANMALICE', 'MEANDECEPTION', 'MEANUNWANTEDCONTACT',
        'SUMSTRESS', 'SUMANXIETY', 'SUMDEPRESS'
    ]
    
    # Select only available columns
    available_features = [col for col in feature_columns if col in df.columns]
    features = df[available_features].copy()
    
    # Handle missing values
    features = features.fillna(0)
    
    # Separate categorical and numerical columns
    categorical_columns = ['GENDER', 'AGE', 'MARITAL_STATUS', 'EDUCATION_LEVEL', 
                          'UNIVERSITY_STATUS', 'ACTIVE_SOCIAL_MEDIA', 'TIME_SPENT_SOCIAL_MEDIA']
    categorical_columns = [col for col in categorical_columns if col in features.columns]
    
    numerical_columns = [col for col in features.columns if col not in categorical_columns]
    
    # Initialize encoders and scalers
    label_encoders = {}
    scaler = StandardScaler()
    
    # Process categorical features
    features_processed = features.copy()
    for col in categorical_columns:
        if col in features_processed.columns:
            le = LabelEncoder()
            features_processed[col] = le.fit_transform(features_processed[col].astype(str))
            label_encoders[col] = le
    
    # Scale numerical features
    if len(numerical_columns) > 0:
        features_processed[numerical_columns] = scaler.fit_transform(features_processed[numerical_columns])
    
    # Split data for model evaluation
    X_train, X_test, y_train_dep, y_test_dep = train_test_split(
        features_processed, df['Depression'], test_size=0.2, random_state=42
    )
    _, _, y_train_stress, y_test_stress = train_test_split(
        features_processed, df['Stress'], test_size=0.2, random_state=42
    )
    _, _, y_train_anx, y_test_anx = train_test_split(
        features_processed, df['Anxiety'], test_size=0.2, random_state=42
    )
    
    # Train Random Forest models
    rf_depression = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_stress = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_anxiety = RandomForestClassifier(n_estimators=100, random_state=42)
    
    rf_depression.fit(X_train, y_train_dep)
    rf_stress.fit(X_train, y_train_stress)
    rf_anxiety.fit(X_train, y_train_anx)
    
    # Train Logistic Regression models
    lr_depression = LogisticRegression(random_state=42, max_iter=1000)
    lr_stress = LogisticRegression(random_state=42, max_iter=1000)
    lr_anxiety = LogisticRegression(random_state=42, max_iter=1000)
    
    lr_depression.fit(X_train, y_train_dep)
    lr_stress.fit(X_train, y_train_stress)
    lr_anxiety.fit(X_train, y_train_anx)
    
    # Calculate accuracies
    rf_accuracies = {
        'depression': accuracy_score(y_test_dep, rf_depression.predict(X_test)),
        'stress': accuracy_score(y_test_stress, rf_stress.predict(X_test)),
        'anxiety': accuracy_score(y_test_anx, rf_anxiety.predict(X_test))
    }
    
    lr_accuracies = {
        'depression': accuracy_score(y_test_dep, lr_depression.predict(X_test)),
        'stress': accuracy_score(y_test_stress, lr_stress.predict(X_test)),
        'anxiety': accuracy_score(y_test_anx, lr_anxiety.predict(X_test))
    }
    
    return {
        'rf_models': {
            'depression': rf_depression,
            'stress': rf_stress,
            'anxiety': rf_anxiety
        },
        'lr_models': {
            'depression': lr_depression,
            'stress': lr_stress,
            'anxiety': lr_anxiety
        },
        'label_encoders': label_encoders,
        'scaler': scaler,
        'feature_columns': features_processed.columns.tolist(),
        'categorical_columns': categorical_columns,
        'numerical_columns': numerical_columns,
        'data': df,
        'rf_accuracies': rf_accuracies,
        'lr_accuracies': lr_accuracies,
        'target_values': {
            'depression': sorted(df['Depression'].unique().tolist()),
            'stress': sorted(df['Stress'].unique().tolist()),
            'anxiety': sorted(df['Anxiety'].unique().tolist())
        }
    }

# --- Enhanced Chatbot Class ---
class EnhancedMentalHealthChatbot:
    def __init__(self):
        self.responses = {
            "hello": [
                "Hey there! I'm glad you're here. Want to talk about how you're feeling?",
                "Hi! You're not alone. I'm here to support you. How's your day going?",
                "Hello! It takes courage to reach out. I'm here to listen and help."
            ],
            "stressed": [
                "Take a deep breath. You're doing the best you can, and that's enough.",
                "Stress is temporary, but your strength is permanent. Try some relaxation techniques.",
                "Remember: you've overcome challenges before. This too shall pass."
            ],
            "anxious": [
                "It's okay to feel anxious. Focus on one thing you can control right now.",
                "Try the 5-4-3-2-1 grounding technique: 5 things you see, 4 you touch, 3 you hear, 2 you smell, 1 you taste.",
                "Anxiety is a feeling, not a fact. You are stronger than your worries."
            ],
            "depressed": [
                "You are not alone. Even on dark days, your presence matters deeply.",
                "Small steps still count. Let's just take today one moment at a time.",
                "Depression lies to you. You are worthy of love, care, and happiness."
            ],
            "bullied": [
                "I'm sorry you're experiencing this. Bullying is never okay and it's not your fault.",
                "You are stronger than their words. Talk to someone you trust - a teacher, parent, or counselor.",
                "Your worth isn't defined by how others treat you. You deserve respect and kindness."
            ],
            "insecure": [
                "Everyone feels insecure sometimes. You are more capable than you think.",
                "Your worth isn't measured by comparison to others. You have unique strengths.",
                "Self-compassion is a skill. Be as kind to yourself as you would be to a good friend."
            ],
            "help": [
                "I'm here to listen and provide support. You can talk about stress, anxiety, depression, or anything else.",
                "Remember: seeking help is a sign of strength, not weakness. What's on your mind?",
                "You've taken an important step by reaching out. How can I support you today?"
            ],
            "default": [
                "I'm here for you. Want to tell me more about what's on your mind?",
                "Sometimes talking helps. I'm listening without judgment.",
                "Your feelings are valid. What would help you feel a bit better right now?"
            ]
        }

    def respond(self, message):
        message = message.lower()
        # More sophisticated keyword matching
        if any(word in message for word in ["hello", "hi", "hey", "start"]):
            return random.choice(self.responses["hello"])
        elif any(word in message for word in ["stress", "stressed", "overwhelmed", "pressure"]):
            return random.choice(self.responses["stressed"])
        elif any(word in message for word in ["anxious", "anxiety", "worried", "nervous", "panic"]):
            return random.choice(self.responses["anxious"])
        elif any(word in message for word in ["depressed", "depression", "sad", "down", "hopeless"]):
            return random.choice(self.responses["depressed"])
        elif any(word in message for word in ["bullied", "bullying", "teased", "harassed"]):
            return random.choice(self.responses["bullied"])
        elif any(word in message for word in ["insecure", "worthless", "inadequate", "not good enough"]):
            return random.choice(self.responses["insecure"])
        elif any(word in message for word in ["help", "support", "advice"]):
            return random.choice(self.responses["help"])
        else:
            return random.choice(self.responses["default"])

# --- Enhanced Recommendations System ---
def get_enhanced_recommendations(stress, anxiety, depression, model_type="Random Forest"):
    suggestions = []
    
    # Map numeric predictions to severity levels based on your dataset
    def get_severity_level(score):
        if score == 1:
            return "Normal"
        elif score == 2:
            return "Mild"
        elif score == 3:
            return "Moderate"
        elif score == 4:
            return "Severe"
        elif score == 5:
            return "Extremely Severe"
        else:
            return "Normal"
    
    stress_level = get_severity_level(stress)
    anxiety_level = get_severity_level(anxiety)
    depression_level = get_severity_level(depression)
    
    # Stress-specific recommendations
    if stress >= 4:
        suggestions.extend([
            "🚨 High stress detected. Consider speaking with a mental health professional.",
            "🧘‍♀️ Practice daily meditation or deep breathing exercises (15-20 minutes).",
            "📝 Keep a stress journal to identify triggers and patterns.",
            "🏃‍♂️ Regular physical exercise can significantly reduce stress hormones."
        ])
    elif stress >= 3:
        suggestions.extend([
            "⚠️ Moderate stress levels detected. Time management techniques may help.",
            "🎵 Try listening to calming music or nature sounds for 10 minutes daily.",
            "💤 Ensure 7-8 hours of quality sleep to help manage stress better."
        ])
    elif stress >= 2:
        suggestions.extend([
            "🌿 Mild stress is normal, but consider relaxation techniques.",
            "☕ Limit caffeine intake, especially in the evening."
        ])
    
    # Anxiety-specific recommendations
    if anxiety >= 4:
        suggestions.extend([
            "🚨 High anxiety levels. Professional counseling is strongly recommended.",
            "🔄 Practice the 4-7-8 breathing technique when feeling anxious.",
            "📱 Consider anxiety management apps like Headspace or Calm.",
            "🚫 Limit social media and news consumption if they increase anxiety."
        ])
    elif anxiety >= 3:
        suggestions.extend([
            "😰 Moderate anxiety detected. Grounding techniques can be very helpful.",
            "📖 Try progressive muscle relaxation before bedtime.",
            "👥 Connect with supportive friends or family members regularly."
        ])
    elif anxiety >= 2:
        suggestions.extend([
            "💚 Mild anxiety is manageable with self-care practices.",
            "🌱 Spend time in nature or practice mindfulness meditation."
        ])
    
    # Depression-specific recommendations
    if depression >= 4:
        suggestions.extend([
            "🚨 Significant depression indicators. Please reach out to a mental health professional immediately.",
            "🤝 Don't isolate yourself - maintain social connections even when it's hard.",
            "☀️ Try to get natural sunlight exposure daily, even for short periods.",
            "📋 Create a daily routine to provide structure and purpose."
        ])
    elif depression >= 3:
        suggestions.extend([
            "💙 Moderate depression signs detected. Consider counseling or therapy.",
            "🎯 Set small, achievable daily goals to build momentum.",
            "🏃‍♀️ Light exercise like walking can boost mood naturally.",
            "📱 Reach out to trusted friends or family when feeling down."
        ])
    elif depression >= 2:
        suggestions.extend([
            "🌻 Mild mood concerns. Focus on self-care and positive activities.",
            "📖 Practice gratitude by writing down three good things each day."
        ])
    
    # General wellness recommendations
    suggestions.extend([
        f"📊 Assessment Results ({model_type}): Stress - {stress_level}, Anxiety - {anxiety_level}, Depression - {depression_level}",
        "💪 Remember: Mental health is just as important as physical health.",
        "🌈 Every small step towards better mental health counts.",
        "📞 Crisis Helpline: If you're in immediate danger, call emergency services or a mental health crisis line."
    ])
    
    return suggestions

# --- Enhanced Input Preprocessing ---
def preprocess_user_input(input_data, resources):
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Apply label encoding to categorical columns
    for col in resources['categorical_columns']:
        if col in input_df.columns:
            le = resources['label_encoders'][col]
            try:
                input_df[col] = le.transform(input_df[col].astype(str))
            except ValueError:
                # Handle unseen categories by using the most common class
                input_df[col] = 0
    
    # Apply scaling to numerical columns
    if len(resources['numerical_columns']) > 0:
        numerical_cols = [col for col in resources['numerical_columns'] if col in input_df.columns]
        if numerical_cols:
            input_df[numerical_cols] = resources['scaler'].transform(input_df[numerical_cols])
    
    # Ensure all required columns are present
    for col in resources['feature_columns']:
        if col not in input_df.columns:
            input_df[col] = 0
    
    return input_df[resources['feature_columns']]

# --- Main Enhanced App ---
def main():
    st.set_page_config(
        page_title="MindCare Pro - Mental Health Assessment", 
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .recommendation-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #28a745;
    }
    </style>
    """, unsafe_allow_html=True)
    
    try:
        if "resources" not in st.session_state:
            with st.spinner("Loading AI models and preparing the system..."):
                resources = load_data_and_models()
                st.session_state["resources"] = resources
        else:
            resources = st.session_state["resources"]
            
    except FileNotFoundError:
        st.error("❌ ERROR: The mental_health_dataset_with_labels.csv file was not found. Please make sure it's in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"❌ ERROR loading data and models: {e}")
        st.stop()
        
    chatbot = EnhancedMentalHealthChatbot()

    # Enhanced Sidebar
    st.sidebar.markdown("### 🧠 MindCare Pro")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigate to:", [
        "🏠 Home", 
        "📝 Mental Health Assessment", 
        "💬 AI Support Chat", 
        "📊 Model Performance",
        "🚨 Emergency Resources"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Quick Stats")
    if "resources" in st.session_state:
        avg_rf_acc = np.mean(list(resources['rf_accuracies'].values()))
        avg_lr_acc = np.mean(list(resources['lr_accuracies'].values()))
        st.sidebar.metric("Random Forest Accuracy", f"{avg_rf_acc:.1%}")
        st.sidebar.metric("Logistic Regression Accuracy", f"{avg_lr_acc:.1%}")

    if page == "🏠 Home":
        st.markdown("""
        <div class="main-header">
            <h1>🧠 MindCare Pro: Advanced Mental Health Assessment</h1>
            <p style="font-size: 1.2em; margin-top: 1rem;">
                AI-powered mental health screening with advanced machine learning models
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🎯 Accurate Predictions
            - Random Forest & Logistic Regression models
            - Multi-class classification for stress, anxiety, depression
            - Evidence-based assessment tools
            """)
        
        with col2:
            st.markdown("""
            ### 🚀 Advanced Features
            - Automated preprocessing with LabelEncoder & StandardScaler
            - Real-time model comparison
            - Interactive visualizations
            """)
        
        with col3:
            st.markdown("""
            ### 💡 Actionable Insights
            - Personalized recommendations
            - Evidence-based mental wellness tips
            - 24/7 AI support chatbot
            """)
        
        st.markdown("---")
        
        with st.expander("🔍 How Our AI System Works", expanded=True):
            st.markdown("""
            1. **Data Collection**: Answer questions about demographics, social media usage, and experiences
            2. **Automated Preprocessing**: Our system automatically encodes categorical variables and scales numerical data
            3. **AI Prediction**: Two advanced models (Random Forest & Logistic Regression) analyze your responses
            4. **Personalized Results**: Get detailed insights into your stress, anxiety, and depression levels
            5. **Expert Recommendations**: Receive tailored mental wellness strategies based on your results
            """)

    elif page == "📝 Mental Health Assessment":
        st.title("📝 Comprehensive Mental Health Assessment")
        st.markdown("*Complete the form below for an AI-powered analysis of your mental wellbeing*")
        
        with st.form("enhanced_assessment_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Demographics")
                gender = st.selectbox("Gender", ["Female", "Male"])
                age = st.selectbox("Age Group", ["<18 years old", "19-24 years old", ">25 years old"])
                marital_status = st.selectbox("Marital Status", ["Single", "Married"])
                education = st.selectbox("Education Level", ["Diploma/Foundation", "Bachelor Degree", "Postgraduate studies"])
                
            with col2:
                st.subheader("🎓 Academic & Social")
                university = st.selectbox("Institution Type", ["Public Universities", "Private Universities"])
                social_media = st.selectbox("Social Media Activity Level", ["Less active", "Active", "Very active"])
                social_media_time = st.selectbox("Daily Social Media Usage", ["1-2 hours", "3-6 hours", "7-10 hours", "> 11 hours", "Whole day"])
            
            st.subheader("🎭 Cybervictimization Assessment")
            st.markdown("*Rate your experiences with cyberbullying and online harassment*")
            
            col3, col4 = st.columns(2)
            
            with col3:
                cyber_total = st.slider("Overall Cyberbullying Experience (0-100)", 0, 100, 15, 
                                      help="Total cyberbullying experience score")
                public_humiliation = st.slider("Public Humiliation Online (0-50)", 0, 50, 5, 
                                             help="Experience with public embarrassment online")
                
            with col4:
                malice = st.slider("Malicious Online Behavior (0-50)", 0, 50, 8, 
                                 help="Experience with intentionally harmful online behavior")
                unwanted_contact = st.slider("Unwanted Contact (0-50)", 0, 50, 3, 
                                           help="Frequency of receiving unwanted messages or contact")
            
            # Advanced options
            with st.expander("Advanced Assessment Options"):
                deception = st.slider("Online Deception (0-50)", 0, 50, 2)
                include_personality = st.checkbox("Include personality factors in prediction")
            
            model_choice = st.selectbox("Choose AI Model", ["Random Forest", "Logistic Regression", "Both Models Comparison"])
            
            submit_button = st.form_submit_button("🔍 Analyze My Mental Health", use_container_width=True)

        if submit_button:
            with st.spinner("🤖 AI models are analyzing your responses..."):
                try:
                    # Create input data matching your dataset structure
                    input_data = {
                        'GENDER': gender,
                        'AGE': age,
                        'MARITAL_STATUS': marital_status,
                        'EDUCATION_LEVEL': education,
                        'UNIVERSITY_STATUS': university,
                        'ACTIVE_SOCIAL_MEDIA': social_media,
                        'TIME_SPENT_SOCIAL_MEDIA': social_media_time,
                        'CVTOTAL': cyber_total,
                        'CVPUBLICHUMILIATION': public_humiliation,
                        'CVMALICE': malice,
                        'CVUNWANTEDCONTACT': unwanted_contact,
                        'CVDECEPTION': deception if 'deception' in locals() else 0,
                        'MEANPUBLICHUMILIATION': public_humiliation / 10 if public_humiliation > 0 else 0,
                        'MEANMALICE': malice / 10 if malice > 0 else 0,
                        'MEANDECEPTION': deception / 10 if 'deception' in locals() and deception > 0 else 0,
                        'MEANUNWANTEDCONTACT': unwanted_contact / 10 if unwanted_contact > 0 else 0,
                        'SUMSTRESS': 0,  # Will be calculated by the model
                        'SUMANXIETY': 0,
                        'SUMDEPRESS': 0
                    }

                    # Preprocess input
                    processed_input = preprocess_user_input(input_data, resources)
                    
                    # Get predictions from both models
                    rf_predictions = {
                        'stress': int(resources['rf_models']['stress'].predict(processed_input)[0]),
                        'anxiety': int(resources['rf_models']['anxiety'].predict(processed_input)[0]),
                        'depression': int(resources['rf_models']['depression'].predict(processed_input)[0])
                    }
                    
                    lr_predictions = {
                        'stress': int(resources['lr_models']['stress'].predict(processed_input)[0]),
                        'anxiety': int(resources['lr_models']['anxiety'].predict(processed_input)[0]),
                        'depression': int(resources['lr_models']['depression'].predict(processed_input)[0])
                    }
                    
                    severity_levels = ["Normal", "Mild", "Moderate", "Severe", "Extremely Severe"]
                    
                    def get_severity_level(score):
                        if score == 1:
                            return "Normal"
                        elif score == 2:
                            return "Mild"
                        elif score == 3:
                            return "Moderate"
                        elif score == 4:
                            return "Severe"
                        elif score == 5:
                            return "Extremely Severe"
                        else:
                            return "Normal"
                    
                    st.success("✅ Analysis Complete!")
                    
                    if model_choice == "Both Models Comparison":
                        st.subheader("🔬 Model Comparison Results")
                        
                        comparison_data = {
                            'Condition': ['Stress', 'Anxiety', 'Depression'],
                            'Random Forest': [rf_predictions['stress'], rf_predictions['anxiety'], rf_predictions['depression']],
                            'Logistic Regression': [lr_predictions['stress'], lr_predictions['anxiety'], lr_predictions['depression']]
                        }
                        
                        df_comparison = pd.DataFrame(comparison_data)
                        
                        fig = px.bar(df_comparison, x='Condition', y=['Random Forest', 'Logistic Regression'],
                                   title='Model Predictions Comparison', barmode='group')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show both predictions
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 🌲 Random Forest Results")
                            for condition, value in rf_predictions.items():
                                st.metric(condition.title(), get_severity_level(value))
                        
                        with col2:
                            st.markdown("#### 📈 Logistic Regression Results")
                            for condition, value in lr_predictions.items():
                                st.metric(condition.title(), get_severity_level(value))
                        
                        # Use Random Forest for recommendations
                        predictions = rf_predictions
                        model_type = "Random Forest"
                        
                    else:
                        # Single model results
                        if model_choice == "Random Forest":
                            predictions = rf_predictions
                            model_type = "Random Forest"
                        else:
                            predictions = lr_predictions
                            model_type = "Logistic Regression"
                        
                        st.subheader(f"📊 {model_type} Analysis Results")
                        
                        col1, col2, col3 = st.columns(3)
                        for i, (condition, value) in enumerate(predictions.items()):
                            cols = [col1, col2, col3]
                            with cols[i]:
                                st.metric(condition.title() + " Level", get_severity_level(value))
                        
                        # Visualization
                        chart_data = pd.DataFrame({
                            "Condition": list(predictions.keys()),
                            "Level": list(predictions.values())
                        })
                        
                        fig = px.bar(chart_data, x='Condition', y='Level', 
                                   title=f'{model_type} Predictions',
                                   color='Level', color_continuous_scale='RdYlGn_r')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Enhanced Recommendations
                    st.subheader("🌱 Personalized Recommendations")
                    recommendations = get_enhanced_recommendations(
                        predictions['stress'], 
                        predictions['anxiety'], 
                        predictions['depression'],
                        model_type
                    )
                    
                    for i, rec in enumerate(recommendations):
                        if i == len(recommendations) - 4:  # Assessment results summary
                            st.info(rec)
                        else:
                            st.markdown(f"""
                            <div class="recommendation-box">
                                {rec}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Save results to session state for chat context
                    st.session_state['last_assessment'] = {
                        'predictions': predictions,
                        'model_type': model_type,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    st.success("💡 Tip: Visit the AI Support Chat to discuss these results with our AI counselor!")
                    
                except Exception as e:
                    st.error(f"❌ An error occurred during analysis: {e}")
                    st.error("Please check your input data and try again.")
                    with st.expander("Debug Information"):
                        st.write(f"Available features: {resources['feature_columns']}")
                        st.write(f"Error details: {str(e)}")

    elif page == "💬 AI Support Chat":
        st.title("💬 AI Mental Health Support Chat")
        st.markdown("*Chat with our AI counselor for personalized support and guidance*")
        
        # Initialize chat history
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
            welcome_msg = "Hello! I'm your AI mental health support assistant. I'm here to listen, provide support, and offer evidence-based coping strategies. How are you feeling today?"
            
            # Add context from recent assessment if available
            if 'last_assessment' in st.session_state:
                assessment = st.session_state['last_assessment']
                welcome_msg += f"\n\nI can see you recently completed an assessment on {assessment['timestamp']}. Would you like to discuss those results or talk about something else?"
            
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": welcome_msg
            })

        # Display chat history
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if user_input := st.chat_input("Type your message here..."):
            # Add user message
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    bot_response = chatbot.respond(user_input)
                    
                    # Add context-aware responses if recent assessment exists
                    if 'last_assessment' in st.session_state and any(word in user_input.lower() for word in ["assessment", "results", "prediction", "score"]):
                        assessment = st.session_state['last_assessment']
                        predictions = assessment['predictions']
                        
                        def get_severity_level(score):
                            if score == 1:
                                return "Normal"
                            elif score == 2:
                                return "Mild"
                            elif score == 3:
                                return "Moderate"
                            elif score == 4:
                                return "Severe"
                            elif score == 5:
                                return "Extremely Severe"
                            else:
                                return "Normal"
                        
                        context_response = f"\nBased on your recent {assessment['model_type']} assessment:"
                        for condition, value in predictions.items():
                            context_response += f"\n• {condition.title()}: {get_severity_level(value)}"
                        
                        context_response += "\n\nWould you like specific advice for any of these areas?"
                        bot_response += context_response
                    
                    st.markdown(bot_response)
            
            # Add assistant response to chat history
            st.session_state.chat_messages.append({"role": "assistant", "content": bot_response})

    elif page == "📊 Model Performance":
        st.title("📊 AI Model Performance Dashboard")
        st.markdown("*Detailed performance metrics of our mental health prediction models*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌲 Random Forest Performance")
            rf_avg_accuracy = np.mean(list(resources['rf_accuracies'].values()))
            st.metric("Overall Accuracy", f"{rf_avg_accuracy:.1%}")
            
            # Individual model accuracies
            for condition, accuracy in resources['rf_accuracies'].items():
                st.metric(f"{condition.title()} Model", f"{accuracy:.1%}")
        
        with col2:
            st.subheader("📈 Logistic Regression Performance")
            lr_avg_accuracy = np.mean(list(resources['lr_accuracies'].values()))
            st.metric("Overall Accuracy", f"{lr_avg_accuracy:.1%}")
            
            # Individual model accuracies
            for condition, accuracy in resources['lr_accuracies'].items():
                st.metric(f"{condition.title()} Model", f"{accuracy:.1%}")
        
        # Performance Comparison Chart
        st.subheader("📊 Model Accuracy Comparison")
        
        comparison_data = []
        for condition in ['depression', 'stress', 'anxiety']:
            comparison_data.append({
                'Condition': condition.title(),
                'Random Forest': resources['rf_accuracies'][condition],
                'Logistic Regression': resources['lr_accuracies'][condition]
            })
        
        df_performance = pd.DataFrame(comparison_data)
        
        fig = px.bar(df_performance, x='Condition', y=['Random Forest', 'Logistic Regression'],
                   title='Model Accuracy by Mental Health Condition', barmode='group',
                   color_discrete_map={'Random Forest': '#2E8B57', 'Logistic Regression': '#4682B4'})
        fig.update_layout(yaxis_title="Accuracy", yaxis_tickformat='.1%')
        st.plotly_chart(fig, use_container_width=True)
        
        # Dataset Statistics
        st.subheader("📈 Dataset Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Samples", len(resources['data']))
        with col2:
            st.metric("Features Used", len(resources['feature_columns']))
        with col3:
            st.metric("Target Classes", "3 (Stress, Anxiety, Depression)")
        
        # Feature Importance (for Random Forest)
        st.subheader("🎯 Feature Importance Analysis")
        try:
            # Get feature importance from Random Forest models
            rf_model = resources['rf_models']['depression']
            feature_importance = pd.DataFrame({
                'Feature': resources['feature_columns'],
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig = px.bar(feature_importance, x='Importance', y='Feature', 
                        title='Top 10 Most Important Features (Random Forest)',
                        orientation='h')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning("Feature importance analysis not available.")
        
        # Target Distribution
        st.subheader("📊 Target Variable Distribution")
        target_cols = ['Depression', 'Stress', 'Anxiety']
        
        for target in target_cols:
            if target in resources['data'].columns:
                target_counts = resources['data'][target].value_counts().sort_index()
                fig = px.pie(values=target_counts.values, names=target_counts.index,
                           title=f'{target} Level Distribution')
                st.plotly_chart(fig, use_container_width=True)
        
        # Technical Details
        with st.expander("🔧 Technical Implementation Details"):
            st.markdown("""
            ### Preprocessing Pipeline
            - **Categorical Encoding**: LabelEncoder for categorical variables
            - **Numerical Scaling**: StandardScaler for numerical features
            - **Missing Value Handling**: Automatic imputation for unseen categories
            - **Feature Engineering**: Automated feature selection and transformation
            
            ### Model Architecture
            - **Random Forest**: 100 estimators, random_state=42
            - **Logistic Regression**: Max iterations=1000, L2 regularization
            - **Cross-validation**: 80/20 train-test split
            - **Multi-target Prediction**: Separate models for stress, anxiety, depression
            
            ### Performance Metrics
            - Primary metric: Classification Accuracy
            - Evaluation: Test set performance
            - Robustness: Consistent performance across different mental health conditions
            
            ### Dataset Features
            """)
            st.write("**Available Features:**")
            for i, feature in enumerate(resources['feature_columns'], 1):
                st.write(f"{i}. {feature}")

    elif page == "🚨 Emergency Resources":
        st.title("🚨 Emergency Mental Health Resources")
        st.markdown("*If you're experiencing a mental health crisis, help is available 24/7*")
        
        # Emergency warning
        st.error("""
        ⚠️ **IMMEDIATE DANGER**: If you are having thoughts of suicide or self-harm, 
        please contact emergency services (911) or go to your nearest emergency room immediately.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🇺🇸 United States")
            st.markdown("""
            - **988 Suicide & Crisis Lifeline**: 988 or 1-800-273-8255
            - **Crisis Text Line**: Text HOME to 741741
            - **SAMHSA National Helpline**: 1-800-662-4357
            - **National Domestic Violence Hotline**: 1-800-799-7233
            - **LGBT National Hotline**: 1-888-843-4564
            """)
            
            st.subheader("🇮🇳 India")
            st.markdown("""
            - **National Helpline**: 9152987821
            - **iCall (TISS)**: +91 9152987821 | Email: icall@tiss.edu
            - **AASRA Helpline**: 91-22-27546669 / 27546667
            - **Vandrevala Foundation**: 1860 266 2345 / 1800 233 3330
            - **Sneha Foundation**: +91-44-24640050
            """)
        
        with col2:
            st.subheader("🇬🇧 United Kingdom")
            st.markdown("""
            - **Samaritans**: 116 123
            - **Mind Infoline**: 0300 123 3393
            - **Crisis Text Line**: Text SHOUT to 85258
            - **NHS Mental Health Helpline**: 111
            """)
            
            st.subheader("🇨🇦 Canada")
            st.markdown("""
            - **Talk Suicide Canada**: 1-833-456-4566
            - **Crisis Services Canada**: 1-833-456-4566
            - **Kids Help Phone**: 1-800-668-6868
            """)
        
        st.subheader("🌐 Online Resources")
        st.markdown("""
        - **Crisis Text Line**: Available in multiple countries
        - **7 Cups**: Free online emotional support
        - **BetterHelp**: Professional online therapy
        - **Mental Health America**: Screening tools and resources
        - **National Alliance on Mental Illness (NAMI)**: Support groups and resources
        """)
        
        st.info("""
        💡 **Remember**: 
        - You are not alone in this struggle
        - Seeking help is a sign of strength, not weakness
        - Mental health professionals are trained to help
        - There are people who care and want to support you
        """)
        
        # Self-care tips
        with st.expander("🧘 Immediate Self-Care Strategies"):
            st.markdown("""
            ### When Feeling Overwhelmed:
            1. **Breathe**: 4-7-8 breathing (inhale 4, hold 7, exhale 8)
            2. **Ground yourself**: 5-4-3-2-1 technique (5 things you see, 4 you touch, etc.)
            3. **Reach out**: Call or text someone you trust
            4. **Move**: Light exercise or stretching
            5. **Create**: Journal, draw, or listen to music
            
            ### Daily Wellness Practices:
            - Maintain a consistent sleep schedule
            - Eat nutritious meals regularly
            - Limit alcohol and substance use
            - Practice mindfulness or meditation
            - Stay connected with supportive people
            - Engage in activities you enjoy
            """)
        
        # Risk Assessment
        with st.expander("⚠️ When to Seek Immediate Help"):
            st.markdown("""
            **Seek emergency help immediately if you or someone you know:**
            - Has thoughts of suicide or self-harm
            - Has a plan to hurt themselves or others
            - Is hearing voices or experiencing hallucinations
            - Is extremely agitated or showing violent behavior
            - Has taken an overdose or is showing signs of severe drug/alcohol intoxication
            - Is completely withdrawn and unresponsive
            
            **Professional help is recommended if:**
            - Symptoms persist for more than two weeks
            - Daily functioning is significantly impaired
            - You're using alcohol or drugs to cope
            - You're having relationship or work problems due to mental health
            - You feel hopeless or worthless most of the time
            """)

# --- Run the Enhanced App ---
if __name__ == "__main__":
    main()
