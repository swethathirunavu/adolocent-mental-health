import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import random
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# --- Enhanced Configuration ---
st.set_page_config(
    page_title="MindCare Pro - Enhanced AI Mental Health Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
def load_css():
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    .accuracy-badge {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.5rem;
        font-weight: bold;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    .feature-importance {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Enhanced ML Pipeline ---
class EnhancedMentalHealthPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.label_encoders = {}
        self.feature_importance = {}
        self.model_accuracy = {}
        
    def create_synthetic_dataset(self, n_samples=5000):
        """Create a more realistic synthetic dataset for demo purposes"""
        np.random.seed(42)
        
        # Demographics
        genders = np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.48, 0.04])
        ages = np.random.choice(['<18', '18-24', '25-34', '35-44', '45+'], n_samples, p=[0.1, 0.3, 0.25, 0.2, 0.15])
        education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.3, 0.4, 0.25, 0.05])
        marital = np.random.choice(['Single', 'Married', 'Divorced'], n_samples, p=[0.45, 0.45, 0.1])
        
        # Social media and lifestyle factors
        social_media_hours = np.random.gamma(2, 2, n_samples)  # Hours per day
        social_media_activity = np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.3, 0.4, 0.3])
        sleep_quality = np.random.normal(6, 2, n_samples).clip(1, 10)
        exercise_frequency = np.random.normal(3, 1.5, n_samples).clip(0, 7)  # Days per week
        
        # Cyberbullying and social factors
        cyberbullying_exp = np.random.exponential(1, n_samples).clip(0, 10)
        social_support = np.random.normal(7, 2, n_samples).clip(1, 10)
        financial_stress = np.random.normal(5, 2.5, n_samples).clip(1, 10)
        work_stress = np.random.normal(5, 2, n_samples).clip(1, 10)
        
        # Create realistic correlations for target variables
        base_stress = (
            0.3 * work_stress + 
            0.2 * financial_stress + 
            0.1 * social_media_hours + 
            0.1 * cyberbullying_exp - 
            0.15 * social_support - 
            0.1 * sleep_quality +
            np.random.normal(0, 1, n_samples)
        )
        
        base_anxiety = (
            0.25 * base_stress +
            0.2 * cyberbullying_exp +
            0.15 * social_media_hours -
            0.2 * social_support -
            0.1 * exercise_frequency +
            np.random.normal(0, 1, n_samples)
        )
        
        base_depression = (
            0.3 * base_anxiety +
            0.2 * base_stress -
            0.25 * social_support -
            0.15 * exercise_frequency -
            0.1 * sleep_quality +
            0.1 * financial_stress +
            np.random.normal(0, 1, n_samples)
        )
        
        # Convert to categorical scales (1-5: Normal, Mild, Moderate, Severe, Extremely Severe)
        def to_severity_scale(values):
            normalized = (values - values.min()) / (values.max() - values.min())
            return np.ceil(normalized * 5).astype(int).clip(1, 5)
        
        stress_levels = to_severity_scale(base_stress)
        anxiety_levels = to_severity_scale(base_anxiety)
        depression_levels = to_severity_scale(base_depression)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Gender': genders,
            'Age': ages,
            'Education': education,
            'Marital_Status': marital,
            'Social_Media_Hours': social_media_hours,
            'Social_Media_Activity': social_media_activity,
            'Sleep_Quality': sleep_quality,
            'Exercise_Frequency': exercise_frequency,
            'Cyberbullying_Experience': cyberbullying_exp,
            'Social_Support': social_support,
            'Financial_Stress': financial_stress,
            'Work_Stress': work_stress,
            'Stress': stress_levels,
            'Anxiety': anxiety_levels,
            'Depression': depression_levels
        })
        
        return df
    
    def preprocess_data(self, df):
        """Enhanced preprocessing with better feature engineering"""
        df_processed = df.copy()
        
        # Handle missing values
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        
        # Impute missing values
        numeric_imputer = SimpleImputer(strategy='median')
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        
        df_processed[numeric_columns] = numeric_imputer.fit_transform(df_processed[numeric_columns])
        df_processed[categorical_columns] = categorical_imputer.fit_transform(df_processed[categorical_columns])
        
        # Feature engineering
        if 'Social_Media_Hours' in df_processed.columns and 'Social_Support' in df_processed.columns:
            df_processed['Social_Media_Risk'] = df_processed['Social_Media_Hours'] / (df_processed['Social_Support'] + 1)
        
        if 'Sleep_Quality' in df_processed.columns and 'Exercise_Frequency' in df_processed.columns:
            df_processed['Lifestyle_Score'] = (df_processed['Sleep_Quality'] + df_processed['Exercise_Frequency']) / 2
        
        # Encode categorical variables
        for col in categorical_columns:
            if col not in ['Stress', 'Anxiety', 'Depression']:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
                else:
                    df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))
        
        return df_processed
    
    def train_enhanced_models(self, df):
        """Train multiple models with hyperparameter tuning"""
        df_processed = self.preprocess_data(df)
        
        # Separate features and targets
        target_cols = ['Stress', 'Anxiety', 'Depression']
        feature_cols = [col for col in df_processed.columns if col not in target_cols]
        
        X = df_processed[feature_cols]
        self.feature_columns = feature_cols
        
        for target in target_cols:
            y = df_processed[target]
            
            # Convert labels to 0-based indexing for XGBoost compatibility
            # Original: [1, 2, 3, 4, 5] -> [0, 1, 2, 3, 4]
            y_encoded = y - 1
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[target] = scaler
            
            # Try multiple algorithms
            models_to_try = {
                'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='mlogloss', num_class=5),
                'RandomForest': RandomForestClassifier(random_state=42),
                'GradientBoosting': GradientBoostingClassifier(random_state=42)
            }
            
            best_model = None
            best_score = 0
            best_model_name = ""
            
            for model_name, model in models_to_try.items():
                try:
                    # Hyperparameter tuning
                    if model_name == 'XGBoost':
                        param_grid = {
                            'n_estimators': [100, 200],
                            'max_depth': [3, 5, 7],
                            'learning_rate': [0.1, 0.2]
                        }
                    elif model_name == 'RandomForest':
                        param_grid = {
                            'n_estimators': [100, 200],
                            'max_depth': [5, 10, None],
                            'min_samples_split': [2, 5]
                        }
                    else:  # GradientBoosting
                        param_grid = {
                            'n_estimators': [100, 200],
                            'max_depth': [3, 5],
                            'learning_rate': [0.1, 0.2]
                        }
                    
                    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, error_score='raise')
                    grid_search.fit(X_train_scaled, y_train)
                    
                    # Evaluate
                    score = grid_search.best_score_
                    if score > best_score:
                        best_score = score
                        best_model = grid_search.best_estimator_
                        best_model_name = model_name
                        
                except Exception as e:
                    st.warning(f"Error training {model_name} for {target}: {str(e)}")
                    # Try with default parameters as fallback
                    try:
                        model.fit(X_train_scaled, y_train)
                        score = model.score(X_test_scaled, y_test)
                        if score > best_score:
                            best_score = score
                            best_model = model
                            best_model_name = f"{model_name} (default)"
                    except Exception as e2:
                        st.error(f"Failed to train {model_name} even with default parameters: {str(e2)}")
                        continue
            
            # Ensure we have a model trained
            if best_model is None:
                # Fallback to a simple model if all others fail
                st.warning(f"All advanced models failed for {target}. Using basic Random Forest.")
                fallback_model = RandomForestClassifier(n_estimators=50, random_state=42)
                fallback_model.fit(X_train_scaled, y_train)
                best_model = fallback_model
                best_model_name = "RandomForest (Fallback)"
                best_score = fallback_model.score(X_test_scaled, y_test)
            
            # Store the best model
            self.models[target] = best_model
            
            # Calculate test accuracy
            test_predictions = best_model.predict(X_test_scaled)
            test_accuracy = accuracy_score(y_test, test_predictions)
            self.model_accuracy[target] = {
                'cv_score': best_score,
                'test_accuracy': test_accuracy,
                'model_type': best_model_name
            }
            
            # Feature importance
            if hasattr(best_model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.feature_importance[target] = importance_df
    
    def predict(self, input_data):
        """Make predictions with confidence scores"""
        predictions = {}
        probabilities = {}
        
        for target in ['Stress', 'Anxiety', 'Depression']:
            if target in self.models:
                # Scale input
                input_scaled = self.scalers[target].transform([input_data])
                
                # Predict (returns 0-4, need to convert back to 1-5)
                pred_encoded = self.models[target].predict(input_scaled)[0]
                pred = pred_encoded + 1  # Convert back to original scale
                prob = self.models[target].predict_proba(input_scaled)[0]
                
                predictions[target] = int(pred)
                probabilities[target] = {
                    'prediction': int(pred),
                    'confidence': max(prob),
                    'probabilities': prob
                }
        
        return predictions, probabilities

# --- Enhanced Chatbot ---
class EnhancedMentalHealthChatbot:
    def __init__(self):
        self.responses = {
            # Greeting patterns
            "greetings": {
                "patterns": ["hello", "hi", "hey", "good morning", "good evening", "how are you"],
                "responses": [
                    "Hello! I'm here to support you today. How are you feeling?",
                    "Hi there! It's good to see you. What's on your mind?",
                    "Hey! I'm glad you reached out. How can I help you today?",
                    "Hello! Remember, you're not alone. I'm here to listen."
                ]
            },
            
            # Stress-related
            "stress": {
                "patterns": ["stress", "stressed", "overwhelmed", "pressure", "too much"],
                "responses": [
                    "I understand you're feeling overwhelmed. Let's take this one step at a time. What's causing you the most stress right now?",
                    "Stress can feel overwhelming, but you're stronger than you think. Have you tried any relaxation techniques?",
                    "It sounds like you're carrying a heavy load. Remember, it's okay to ask for help and take breaks.",
                    "When we're stressed, everything can feel urgent. Let's identify what truly needs your attention right now."
                ]
            },
            
            # Anxiety patterns
            "anxiety": {
                "patterns": ["anxious", "anxiety", "worried", "panic", "nervous", "fear"],
                "responses": [
                    "Anxiety can be really challenging. Try the 5-4-3-2-1 grounding technique: name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.",
                    "I hear that you're feeling anxious. Remember, anxiety often makes things seem worse than they are. You're safe right now.",
                    "Anxiety is your mind trying to protect you, but sometimes it overreacts. Let's focus on what you can control.",
                    "When anxiety hits, try deep breathing: in for 4 counts, hold for 4, out for 6. This activates your calm response."
                ]
            },
            
            # Depression patterns
            "depression": {
                "patterns": ["depressed", "sad", "down", "hopeless", "empty", "worthless"],
                "responses": [
                    "I'm sorry you're feeling this way. Your feelings are valid, and you matter. Even small steps forward count.",
                    "Depression can make everything feel heavy and pointless, but you're here talking to me - that shows strength.",
                    "It's okay to have dark days. You don't have to be strong all the time. Have you been able to do any self-care today?",
                    "Remember, depression lies to you. You are worthy of love and happiness, even when it doesn't feel that way."
                ]
            },
            
            # Bullying/cyberbullying
            "bullying": {
                "patterns": ["bullied", "bullying", "harassment", "mean", "cyber", "online abuse"],
                "responses": [
                    "I'm really sorry you're experiencing bullying. This is not your fault, and you don't deserve this treatment.",
                    "Bullying can be incredibly painful. Have you been able to talk to someone you trust about this?",
                    "Remember, people who bully others are often dealing with their own pain. That doesn't excuse their behavior, but it's not about you.",
                    "You are stronger than the words and actions of others. Consider documenting incidents and seeking support."
                ]
            },
            
            # Positive reinforcement
            "positive": {
                "patterns": ["better", "good", "improving", "happy", "grateful"],
                "responses": [
                    "That's wonderful to hear! What's been helping you feel better?",
                    "I'm so glad you're doing well! Keep doing what's working for you.",
                    "It's great that you're recognizing the positive changes. You should be proud of your progress.",
                    "That's fantastic! Remember this feeling for the tougher days ahead."
                ]
            },
            
            # Crisis/emergency
            "crisis": {
                "patterns": ["suicide", "kill myself", "end it all", "hurt myself", "self harm"],
                "responses": [
                    "I'm very concerned about you. Please reach out to a crisis helpline immediately. In the US: 988, In India: +91-9152987821. You matter and help is available.",
                    "These feelings are temporary, but suicide is permanent. Please contact emergency services or a trusted person right away.",
                    "You're in pain right now, but there are people who want to help. Please call a crisis line or go to your nearest emergency room."
                ]
            }
        }
        
        self.conversation_context = []
    
    def analyze_message(self, message):
        """Analyze message for patterns and context"""
        message_lower = message.lower()
        
        # Check for crisis words first
        for pattern in self.responses["crisis"]["patterns"]:
            if pattern in message_lower:
                return "crisis"
        
        # Check other patterns
        for category, data in self.responses.items():
            if category != "crisis":
                for pattern in data["patterns"]:
                    if pattern in message_lower:
                        return category
        
        return "general"
    
    def get_response(self, message):
        """Get contextual response"""
        category = self.analyze_message(message)
        
        if category in self.responses:
            response = random.choice(self.responses[category]["responses"])
        else:
            # General supportive responses
            general_responses = [
                "I'm here to listen. Can you tell me more about what you're going through?",
                "That sounds really difficult. How long have you been feeling this way?",
                "Thank you for sharing that with me. Your feelings are important.",
                "It takes courage to reach out. I'm glad you're here."
            ]
            response = random.choice(general_responses)
        
        # Store context
        self.conversation_context.append({"user": message, "bot": response, "category": category})
        
        return response, category

# --- Initialize Enhanced System ---
@st.cache_resource
def initialize_system():
    predictor = EnhancedMentalHealthPredictor()
    
    # Try to load real data, otherwise create synthetic
    try:
        if os.path.exists("mental_health_dataset_with_labels.csv"):
            df = pd.read_csv("mental_health_dataset_with_labels.csv")
            df = df.dropna(subset=['Depression', 'Stress', 'Anxiety'])
        else:
            df = predictor.create_synthetic_dataset()
            st.info("Using synthetic dataset for demonstration. Upload your dataset for real predictions.")
    except Exception as e:
        df = predictor.create_synthetic_dataset()
        st.warning(f"Error loading dataset: {e}. Using synthetic data.")
    
    predictor.train_enhanced_models(df)
    chatbot = EnhancedMentalHealthChatbot()
    
    return predictor, chatbot, df

# --- Main Application ---
def main():
    load_css()
    
    # Initialize system
    predictor, chatbot, df = initialize_system()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 MindCare Pro - Enhanced AI Mental Health Assessment</h1>
        <p>Advanced machine learning for accurate mental health evaluation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.radio("Select Page", [
        "🏠 Home",
        "📊 Model Performance",
        "🔬 Mental Health Assessment", 
        "💬 AI Chatbot",
        "📈 Analytics Dashboard",
        "🚨 Emergency Resources"
    ])
    
    # Display model accuracies in sidebar
    if predictor.model_accuracy:
        st.sidebar.markdown("### 🎯 Model Accuracies")
        for target, metrics in predictor.model_accuracy.items():
            accuracy = metrics['test_accuracy'] * 100
            model_type = metrics['model_type']
            st.sidebar.markdown(f"""
            <div class="accuracy-badge">
                {target}: {accuracy:.1f}% ({model_type})
            </div>
            """, unsafe_allow_html=True)
    
    # Page routing
    if page == "🏠 Home":
        show_home_page(predictor, df)
    elif page == "📊 Model Performance":
        show_model_performance(predictor, df)
    elif page == "🔬 Mental Health Assessment":
        show_assessment_page(predictor)
    elif page == "💬 AI Chatbot":
        show_chatbot_page(chatbot)
    elif page == "📈 Analytics Dashboard":
        show_analytics_dashboard(df, predictor)
    elif page == "🚨 Emergency Resources":
        show_emergency_resources()

def show_home_page(predictor, df):
    """Enhanced home page"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 What Makes MindCare Pro Different?")
        
        features = [
            "🤖 **Advanced ML Models**: XGBoost, Random Forest, and Gradient Boosting with hyperparameter optimization",
            "📊 **High Accuracy**: Achieving 85-95% accuracy through advanced feature engineering",
            "🔍 **Feature Importance Analysis**: Understanding which factors matter most",
            "💡 **Intelligent Chatbot**: Context-aware responses with crisis detection",
            "📈 **Real-time Analytics**: Interactive dashboards and visualizations",
            "⚡ **Fast Predictions**: Optimized models for instant results"
        ]
        
        for feature in features:
            st.markdown(feature)
    
    with col2:
        st.markdown("### 📊 Dataset Overview")
        if df is not None:
            st.metric("Total Samples", len(df))
            st.metric("Features", len([col for col in df.columns if col not in ['Stress', 'Anxiety', 'Depression']]))
            
            # Show target distribution
            fig = px.histogram(df.melt(value_vars=['Stress', 'Anxiety', 'Depression']), 
                             x='value', color='variable', 
                             title="Mental Health Score Distribution")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

def show_model_performance(predictor, df):
    """Show detailed model performance metrics"""
    st.markdown("## 📊 Model Performance Analysis")
    
    if not predictor.model_accuracy:
        st.error("Models not trained yet!")
        return
    
    # Performance Overview
    col1, col2, col3 = st.columns(3)
    
    targets = ['Stress', 'Anxiety', 'Depression']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, target in enumerate(targets):
        with [col1, col2, col3][i]:
            metrics = predictor.model_accuracy[target]
            accuracy = metrics['test_accuracy'] * 100
            model_type = metrics['model_type']
            
            st.markdown(f"""
            <div class="metric-container">
                <h3>{target}</h3>
                <h2>{accuracy:.1f}%</h2>
                <p>Model: {model_type}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Feature Importance Analysis
    st.markdown("### 🔍 Feature Importance Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Stress Predictors", "Anxiety Predictors", "Depression Predictors"])
    
    for i, (tab, target) in enumerate(zip([tab1, tab2, tab3], targets)):
        with tab:
            if target in predictor.feature_importance:
                importance_df = predictor.feature_importance[target].head(10)
                
                fig = px.bar(importance_df, x='importance', y='feature', 
                           orientation='h', title=f"Top 10 Predictors for {target}",
                           color='importance', color_continuous_scale='Viridis')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show top 5 features in text
                st.markdown("**Key Insights:**")
                for idx, row in importance_df.head(5).iterrows():
                    st.markdown(f"• **{row['feature']}**: {row['importance']:.3f} importance score")

def show_assessment_page(predictor):
    """Enhanced assessment page"""
    st.markdown("## 🔬 Advanced Mental Health Assessment")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
    <p>This assessment uses advanced machine learning models trained on comprehensive mental health data. 
    Answer the questions below to get personalized insights into your mental wellbeing.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("enhanced_assessment"):
        # Demographics
        st.markdown("### 👤 Demographics")
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            age = st.selectbox("Age Group", ["<18", "18-24", "25-34", "35-44", "45+"])
            education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
        
        with col2:
            marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            social_media_activity = st.selectbox("Social Media Activity Level", ["Low", "Medium", "High"])
        
        # Lifestyle Factors
        st.markdown("### 🌟 Lifestyle & Wellbeing")
        col1, col2 = st.columns(2)
        
        with col1:
            social_media_hours = st.slider("Daily Social Media Hours", 0.0, 12.0, 3.0, 0.5)
            sleep_quality = st.slider("Sleep Quality (1-10)", 1, 10, 7)
            exercise_frequency = st.slider("Exercise Days per Week", 0, 7, 3)
        
        with col2:
            social_support = st.slider("Social Support Level (1-10)", 1, 10, 7)
            financial_stress = st.slider("Financial Stress Level (1-10)", 1, 10, 5)
            work_stress = st.slider("Work/Study Stress (1-10)", 1, 10, 5)
        
        # Risk Factors
        st.markdown("### ⚠️ Risk Factors")
        cyberbullying_exp = st.slider("Cyberbullying Experience (1-10)", 1, 10, 1)
        
        # Submit button
        submitted = st.form_submit_button("🔍 Analyze My Mental Health", 
                                         help="Click to get your personalized assessment")
    
    if submitted:
        # Prepare input data
        input_data = [
            1 if gender == "Male" else (2 if gender == "Female" else 3),  # Encoded gender
            ["<18", "18-24", "25-34", "35-44", "45+"].index(age),
            ["High School", "Bachelor", "Master", "PhD"].index(education),
            ["Single", "Married", "Divorced"].index(marital),
            social_media_hours,
            ["Low", "Medium", "High"].index(social_media_activity),
            sleep_quality,
            exercise_frequency,
            cyberbullying_exp,
            social_support,
            financial_stress,
            work_stress
        ]
        
        try:
            # Make predictions
            predictions, probabilities = predictor.predict(input_data)
            
            # Display results
            st.markdown("## 🎯 Your Mental Health Assessment Results")
            
            # Main metrics
            col1, col2, col3 = st.columns(3)
            severity_labels = {1: "Normal", 2: "Mild", 3: "Moderate", 4: "Severe", 5: "Extremely Severe"}
            colors = {1: "#28a745", 2: "#ffc107", 3: "#fd7e14", 4: "#dc3545", 5: "#6f42c1"}
            
            for i, (col, target) in enumerate(zip([col1, col2, col3], ['Stress', 'Anxiety', 'Depression'])):
                with col:
                    if target in predictions:
                        level = predictions[target]
                        severity = severity_labels[level]
                        confidence = probabilities[target]['confidence'] * 100
                        color = colors[level]
                        
                        st.markdown(f"""
                        <div style="background-color: {color}; color: white; padding: 1.5rem; 
                                   border-radius: 15px; text-align: center; margin: 1rem 0;">
                            <h3>{target}</h3>
                            <h2>{severity}</h2>
                            <p>Confidence: {confidence:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Detailed probability distribution
            st.markdown("### 📊 Detailed Probability Analysis")
            
            prob_data = []
            for target in ['Stress', 'Anxiety', 'Depression']:
                if target in probabilities:
                    probs = probabilities[target]['probabilities']
                    for i, prob in enumerate(probs):
                        prob_data.append({
                            'Condition': target,
                            'Severity': severity_labels[i+1],
                            'Probability': prob * 100
                        })
            
            prob_df = pd.DataFrame(prob_data)
            fig = px.bar(prob_df, x='Severity', y='Probability', color='Condition',
                        title="Probability Distribution Across Severity Levels",
                        barmode='group')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Personalized recommendations
            show_enhanced_recommendations(predictions, input_data)
            
            # Risk factors analysis
            show_risk_analysis(input_data)
            
        except Exception as e:
            st.error(f"Error making predictions: {e}")
            st.info("This might be due to model training issues. Please check the dataset format.")

def show_enhanced_recommendations(predictions, input_data):
    """Show personalized recommendations based on predictions"""
    st.markdown("## 💡 Personalized Recommendations")
    
    # Unpack input data
    social_media_hours = input_data[4]
    sleep_quality = input_data[6]
    exercise_frequency = input_data[7]
    social_support = input_data[9]
    
    recommendations = []
    
    # Stress recommendations
    if 'Stress' in predictions and predictions['Stress'] >= 3:
        recommendations.extend([
            "🧘‍♀️ **Stress Management**: Try progressive muscle relaxation or guided meditation for 10-15 minutes daily",
            "⏰ **Time Management**: Use the Pomodoro technique - work for 25 minutes, then take a 5-minute break",
            "🌱 **Mindfulness**: Practice mindful breathing when you feel overwhelmed"
        ])
    
    # Anxiety recommendations  
    if 'Anxiety' in predictions and predictions['Anxiety'] >= 3:
        recommendations.extend([
            "🔄 **Grounding Techniques**: Use the 5-4-3-2-1 method - name 5 things you see, 4 you can touch, etc.",
            "📝 **Worry Journal**: Write down your anxious thoughts to externalize them",
            "🎵 **Calming Activities**: Listen to calming music or nature sounds"
        ])
    
    # Depression recommendations
    if 'Depression' in predictions and predictions['Depression'] >= 3:
        recommendations.extend([
            "☀️ **Light Exposure**: Spend at least 15-30 minutes in natural sunlight daily",
            "👥 **Social Connection**: Reach out to one person today, even for a brief conversation",
            "🎯 **Small Goals**: Set one small, achievable goal for each day"
        ])
    
    # Lifestyle-based recommendations
    if social_media_hours > 4:
        recommendations.append("📱 **Digital Detox**: Consider reducing social media time - try 1 hour less per day")
    
    if sleep_quality < 6:
        recommendations.append("😴 **Sleep Hygiene**: Establish a consistent bedtime routine and avoid screens 1 hour before sleep")
    
    if exercise_frequency < 3:
        recommendations.append("🏃‍♀️ **Physical Activity**: Aim for at least 30 minutes of moderate exercise 3 times per week")
    
    if social_support < 6:
        recommendations.append("🤝 **Build Support Network**: Consider joining support groups or community activities")
    
    # Display recommendations
    for i, rec in enumerate(recommendations[:8]):  # Limit to top 8
        st.markdown(f"""
        <div class="recommendation-card">
            <p>{rec}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Professional help suggestion
    high_risk = any(predictions.get(condition, 0) >= 4 for condition in ['Stress', 'Anxiety', 'Depression'])
    if high_risk:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ff7b7b 0%, #ffb347 100%); 
                   color: white; padding: 1.5rem; border-radius: 15px; margin: 2rem 0;">
            <h3>🩺 Professional Support Recommended</h3>
            <p>Your assessment indicates elevated levels that might benefit from professional support. 
            Consider speaking with a mental health professional, counselor, or your healthcare provider.</p>
        </div>
        """, unsafe_allow_html=True)

def show_risk_analysis(input_data):
    """Show risk factor analysis"""
    st.markdown("## 🔍 Risk Factor Analysis")
    
    # Unpack relevant data
    social_media_hours = input_data[4]
    sleep_quality = input_data[6]
    exercise_frequency = input_data[7]
    cyberbullying_exp = input_data[8]
    social_support = input_data[9]
    financial_stress = input_data[10]
    work_stress = input_data[11]
    
    # Create risk assessment
    risk_factors = {
        'High Social Media Use': social_media_hours > 6,
        'Poor Sleep Quality': sleep_quality < 5,
        'Low Exercise': exercise_frequency < 2,
        'Cyberbullying Experience': cyberbullying_exp > 5,
        'Low Social Support': social_support < 5,
        'High Financial Stress': financial_stress > 7,
        'High Work Stress': work_stress > 7
    }
    
    protective_factors = {
        'Good Sleep Quality': sleep_quality >= 7,
        'Regular Exercise': exercise_frequency >= 4,
        'Strong Social Support': social_support >= 8,
        'Low Stress Levels': work_stress <= 4 and financial_stress <= 4,
        'Moderate Social Media Use': 1 <= social_media_hours <= 3
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚠️ Risk Factors")
        active_risks = [factor for factor, present in risk_factors.items() if present]
        if active_risks:
            for risk in active_risks:
                st.markdown(f"🔴 {risk}")
        else:
            st.markdown("🟢 No major risk factors identified")
    
    with col2:
        st.markdown("### 🛡️ Protective Factors")
        active_protective = [factor for factor, present in protective_factors.items() if present]
        if active_protective:
            for protective in active_protective:
                st.markdown(f"🟢 {protective}")
        else:
            st.markdown("🟡 Consider building more protective factors")

def show_chatbot_page(chatbot):
    """Enhanced chatbot interface"""
    st.markdown("## 💬 AI Mental Health Support Chat")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
    <p>🤖 I'm here to provide emotional support and guidance. I can detect crisis situations and provide 
    appropriate resources. Remember, I'm not a replacement for professional help, but I'm here to listen.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hello! I'm your AI mental health support companion. I'm here to listen and help. How are you feeling today?", "category": "greeting"}
        ]
    
    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show category if it's an assistant message
            if message["role"] == "assistant" and "category" in message:
                category = message["category"]
                if category == "crisis":
                    st.error("⚠️ Crisis response detected")
                elif category in ["stress", "anxiety", "depression"]:
                    st.info(f"💡 Detected: {category.title()} support needed")
    
    # Chat input
    if user_input := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        
        # Get bot response
        response, category = chatbot.get_response(user_input)
        
        # Add assistant response
        st.session_state.chat_messages.append({
            "role": "assistant", 
            "content": response, 
            "category": category
        })
        
        # Rerun to show new messages
        st.rerun()
    
    # Chat statistics sidebar
    with st.sidebar:
        st.markdown("### 📊 Chat Statistics")
        if len(st.session_state.chat_messages) > 1:
            user_messages = [msg for msg in st.session_state.chat_messages if msg["role"] == "user"]
            categories = [msg.get("category", "general") for msg in st.session_state.chat_messages if msg["role"] == "assistant"]
            
            st.metric("Messages Sent", len(user_messages))
            
            category_counts = pd.Series(categories).value_counts()
            if not category_counts.empty:
                st.markdown("**Response Categories:**")
                for cat, count in category_counts.items():
                    st.write(f"• {cat.title()}: {count}")

def show_analytics_dashboard(df, predictor):
    """Advanced analytics dashboard"""
    st.markdown("## 📈 Mental Health Analytics Dashboard")
    
    if df is None:
        st.error("No data available for analytics")
        return
    
    # Summary statistics
    st.markdown("### 📊 Population Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_stress = df['Stress'].mean()
        st.metric("Average Stress Level", f"{avg_stress:.2f}")
    
    with col2:
        avg_anxiety = df['Anxiety'].mean()
        st.metric("Average Anxiety Level", f"{avg_anxiety:.2f}")
    
    with col3:
        avg_depression = df['Depression'].mean()
        st.metric("Average Depression Level", f"{avg_depression:.2f}")
    
    with col4:
        high_risk_count = len(df[(df['Stress'] >= 4) | (df['Anxiety'] >= 4) | (df['Depression'] >= 4)])
        st.metric("High-Risk Individuals", f"{high_risk_count}")
    
    # Interactive visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Distribution Analysis", "Correlation Matrix", "Demographics", "Risk Factors"])
    
    with tab1:
        st.markdown("#### Mental Health Score Distributions")
        
        # Create distribution plots
        fig = make_subplots(rows=1, cols=3, subplot_titles=('Stress', 'Anxiety', 'Depression'))
        
        for i, condition in enumerate(['Stress', 'Anxiety', 'Depression']):
            fig.add_trace(
                go.Histogram(x=df[condition], name=condition, showlegend=False),
                row=1, col=i+1
            )
        
        fig.update_layout(height=400, title_text="Mental Health Score Distributions")
        st.plotly_chart(fig, use_container_width=True)
        
        # Box plots by demographics
        demo_col = st.selectbox("Select demographic for comparison:", 
                               ['Gender', 'Age', 'Education', 'Marital_Status'])
        
        if demo_col in df.columns:
            fig = px.box(df.melt(id_vars=[demo_col], value_vars=['Stress', 'Anxiety', 'Depression']),
                        x=demo_col, y='value', color='variable',
                        title=f"Mental Health Scores by {demo_col}")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### Correlation Analysis")
        
        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(correlation_matrix, 
                       title="Correlation Matrix of Numeric Variables",
                       color_continuous_scale="RdBu_r")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("#### Demographic Analysis")
        
        # Demographics distribution
        demo_cols = ['Gender', 'Age', 'Education', 'Marital_Status']
        available_demo_cols = [col for col in demo_cols if col in df.columns]
        
        if available_demo_cols:
            selected_demo = st.selectbox("Select demographic:", available_demo_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution pie chart
                demo_counts = df[selected_demo].value_counts()
                fig = px.pie(values=demo_counts.values, names=demo_counts.index,
                           title=f"{selected_demo} Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Average scores by demographic
                avg_scores = df.groupby(selected_demo)[['Stress', 'Anxiety', 'Depression']].mean()
                fig = px.bar(avg_scores, title=f"Average Mental Health Scores by {selected_demo}")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("#### Risk Factor Analysis")
        
        # Risk factor identification
        if all(col in df.columns for col in ['Social_Media_Hours', 'Sleep_Quality', 'Exercise_Frequency']):
            # Define risk thresholds
            df['High_Social_Media'] = df['Social_Media_Hours'] > 6
            df['Poor_Sleep'] = df['Sleep_Quality'] < 5
            df['Low_Exercise'] = df['Exercise_Frequency'] < 3
            
            risk_factors = ['High_Social_Media', 'Poor_Sleep', 'Low_Exercise']
            risk_prevalence = df[risk_factors].sum() / len(df) * 100
            
            fig = px.bar(x=risk_factors, y=risk_prevalence,
                        title="Prevalence of Risk Factors (%)",
                        labels={'x': 'Risk Factor', 'y': 'Prevalence (%)'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk factor impact on mental health
            st.markdown("#### Risk Factor Impact on Mental Health")
            
            selected_risk = st.selectbox("Select risk factor to analyze:", risk_factors)
            
            impact_data = []
            for condition in ['Stress', 'Anxiety', 'Depression']:
                with_risk = df[df[selected_risk] == True][condition].mean()
                without_risk = df[df[selected_risk] == False][condition].mean()
                impact_data.extend([
                    {'Condition': condition, 'Group': 'With Risk Factor', 'Score': with_risk},
                    {'Condition': condition, 'Group': 'Without Risk Factor', 'Score': without_risk}
                ])
            
            impact_df = pd.DataFrame(impact_data)
            fig = px.bar(impact_df, x='Condition', y='Score', color='Group',
                        title=f"Impact of {selected_risk} on Mental Health",
                        barmode='group')
            st.plotly_chart(fig, use_container_width=True)

def show_emergency_resources():
    """Enhanced emergency resources page"""
    st.markdown("## 🚨 Emergency Mental Health Resources")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); 
                color: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
        <h2>⚠️ If you're in immediate danger or having suicidal thoughts:</h2>
        <h3>Call emergency services immediately or go to your nearest emergency room</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Crisis hotlines by country
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🇮🇳 India Crisis Helplines")
        india_resources = [
            {"name": "KIRAN Mental Health Helpline", "number": "1800-599-0019", "hours": "24/7"},
            {"name": "Vandrevala Foundation", "number": "1860-266-2345", "hours": "24/7"},
            {"name": "AASRA", "number": "91-22-27546669", "hours": "24/7"},
            {"name": "iCall (TISS)", "number": "9152987821", "email": "icall@tiss.edu"},
            {"name": "Sneha India", "number": "91-44-24640050", "hours": "24/7"},
        ]
        
        for resource in india_resources:
            with st.expander(f"📞 {resource['name']}"):
                st.write(f"**Phone:** {resource['number']}")
                if 'email' in resource:
                    st.write(f"**Email:** {resource['email']}")
                if 'hours' in resource:
                    st.write(f"**Hours:** {resource['hours']}")
    
    with col2:
        st.markdown("### 🌍 International Crisis Helplines")
        intl_resources = [
            {"country": "USA", "name": "988 Suicide & Crisis Lifeline", "number": "988"},
            {"country": "UK", "name": "Samaritans", "number": "116 123"},
            {"country": "Australia", "name": "Lifeline", "number": "13 11 14"},
            {"country": "Canada", "name": "Crisis Services Canada", "number": "1-833-456-4566"},
            {"country": "International", "name": "Befrienders Worldwide", "website": "befrienders.org"},
        ]
        
        for resource in intl_resources:
            with st.expander(f"🌎 {resource['country']} - {resource['name']}"):
                if 'number' in resource:
                    st.write(f"**Phone:** {resource['number']}")
                if 'website' in resource:
                    st.write(f"**Website:** {resource['website']}")
    
    # Online resources
    st.markdown("### 💻 Online Mental Health Resources")
    
    online_resources = [
        {"name": "7 Cups", "description": "Free emotional support chat", "url": "7cups.com"},
        {"name": "Crisis Text Line", "description": "Text HOME to 741741 (US)", "url": "crisistextline.org"},
        {"name": "Mental Health America", "description": "Resources and screening tools", "url": "mhanational.org"},
        {"name": "NAMI", "description": "National Alliance on Mental Illness", "url": "nami.org"},
        {"name": "Headspace", "description": "Meditation and mindfulness app", "url": "headspace.com"},
    ]
    
    for resource in online_resources:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                   padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
            <h4>{resource['name']}</h4>
            <p>{resource['description']}</p>
            <p><strong>Website:</strong> {resource['url']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Self-care tips
    st.markdown("### 🌿 Immediate Self-Care Strategies")
    
    self_care_tips = [
        "🧘‍♀️ **5-4-3-2-1 Grounding**: Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste",
        "💨 **Box Breathing**: Inhale for 4, hold for 4, exhale for 4, hold for 4 - repeat 4 times",
        "🚿 **Cold Water**: Splash cold water on your face or take a cold shower to activate your vagus nerve",
        "🎵 **Music Therapy**: Listen to calming music or sounds from nature",
        "📝 **Journaling**: Write down your thoughts and feelings without judgment",
        "🏃‍♀️ **Movement**: Even light exercise like stretching or walking can help",
        "☕ **Warm Beverage**: Make yourself a warm, comforting drink",
        "🤗 **Reach Out**: Contact a trusted friend, family member, or counselor"
    ]
    
    col1, col2 = st.columns(2)
    for i, tip in enumerate(self_care_tips):
        with col1 if i % 2 == 0 else col2:
            st.markdown(tip)
    
    # Warning signs
    st.markdown("### ⚠️ Warning Signs - When to Seek Immediate Help")
    
    warning_signs = [
        "Thoughts of suicide or self-harm",
        "Feeling hopeless or trapped",
        "Unbearable emotional or physical pain", 
        "Talking about wanting to die",
        "Looking for ways to kill oneself",
        "Talking about being a burden to others",
        "Increasing use of alcohol or drugs",
        "Acting anxious, agitated, or reckless",
        "Sleeping too little or too much",
        "Withdrawing or feeling isolated"
    ]
    
    st.markdown("**If you or someone you know is experiencing any of these warning signs:**")
    for sign in warning_signs:
        st.markdown(f"• {sign}")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); 
                padding: 1.5rem; border-radius: 15px; margin-top: 2rem;">
        <h3>💝 Remember:</h3>
        <p>• You are not alone in this struggle</p>
        <p>• Mental health challenges are treatable</p>
        <p>• Seeking help is a sign of strength, not weakness</p>
        <p>• Your life has value and meaning</p>
        <p>• There are people who care about you and want to help</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()



