import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import random
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# --- Enhanced Data Generation and Model Training ---

@st.cache_data
def generate_synthetic_data():
    """Generate a comprehensive synthetic mental health dataset"""
    np.random.seed(42)
    n_samples = 5000
    
    # Demographics
    genders = ['Male', 'Female', 'Non-binary']
    ages = ['<18 years old', '19-24 years old', '25-34 years old', '35-44 years old', '>45 years old']
    marital_statuses = ['Single', 'Married', 'Divorced', 'Widowed']
    education_levels = ['High School', 'Diploma/Foundation', 'Bachelor Degree', 'Master Degree', 'PhD']
    university_types = ['Public Universities', 'Private Universities', 'Community College']
    social_media_activity = ['Less active', 'Moderately active', 'Active', 'Very active']
    time_spent = ['<1 hour', '1-2 hours', '3-4 hours', '5-6 hours', '7-10 hours', '>10 hours']
    
    data = []
    for i in range(n_samples):
        # Generate correlated features
        age_category = np.random.choice(ages, p=[0.15, 0.25, 0.25, 0.20, 0.15])
        gender = np.random.choice(genders, p=[0.48, 0.50, 0.02])
        
        # Age-related correlations
        if age_category in ['<18 years old', '19-24 years old']:
            social_media_prob = [0.05, 0.15, 0.35, 0.45]
            time_prob = [0.05, 0.1, 0.15, 0.2, 0.3, 0.2]
        else:
            social_media_prob = [0.2, 0.3, 0.3, 0.2]
            time_prob = [0.15, 0.25, 0.25, 0.2, 0.1, 0.05]
        
        social_activity = np.random.choice(social_media_activity, p=social_media_prob)
        time_on_social = np.random.choice(time_spent, p=time_prob)
        
        # Cyberbullying experiences (higher for younger demographics)
        cyber_multiplier = 2 if age_category in ['<18 years old', '19-24 years old'] else 1
        cyber_exp = max(1, min(10, int(np.random.exponential(2) * cyber_multiplier)))
        public_humiliation = max(1, min(10, int(np.random.exponential(1.5) * cyber_multiplier)))
        unwanted_contact = max(1, min(10, int(np.random.exponential(1.8) * cyber_multiplier)))
        malice = max(1, min(10, int(np.random.exponential(1.6) * cyber_multiplier)))
        
        # Calculate composite scores
        cv_total = cyber_exp * 10 + np.random.randint(-20, 21)
        cv_public = public_humiliation * 5 + np.random.randint(-10, 11)
        cv_malice = malice * 5 + np.random.randint(-10, 11)
        cv_unwanted = unwanted_contact * 5 + np.random.randint(-10, 11)
        
        # Mental health outcomes (correlated with experiences)
        stress_base = (cyber_exp + public_humiliation + unwanted_contact) / 3
        anxiety_base = (cyber_exp * 1.2 + unwanted_contact * 1.1) / 2
        depression_base = (public_humiliation * 1.3 + malice * 1.1) / 2
        
        # Add some randomness and ensure realistic distributions
        stress = max(1, min(5, int(stress_base + np.random.normal(0, 0.8))))
        anxiety = max(1, min(5, int(anxiety_base + np.random.normal(0, 0.8))))
        depression = max(1, min(5, int(depression_base + np.random.normal(0, 0.8))))
        
        # Binary outcomes based on severity
        bullied = 1 if (cyber_exp > 5 or public_humiliation > 6) else 0
        insecure = 1 if (anxiety > 3 or depression > 3) else 0
        
        data.append({
            'GENDER': gender,
            'AGE': age_category,
            'MARITAL_STATUS': np.random.choice(marital_statuses, p=[0.4, 0.35, 0.15, 0.1]),
            'EDUCATION_LEVEL': np.random.choice(education_levels, p=[0.15, 0.2, 0.35, 0.25, 0.05]),
            'UNIVERSITY_STATUS': np.random.choice(university_types, p=[0.6, 0.3, 0.1]),
            'ACTIVE_SOCIAL_MEDIA': social_activity,
            'TIME_SPENT_SOCIAL_MEDIA': time_on_social,
            'CVTOTAL': cv_total,
            'CVPUBLICHUMILIATION': cv_public,
            'CVMALICE': cv_malice,
            'CVUNWANTEDCONTACT': cv_unwanted,
            'MEANPUBLICHUMILIATION': public_humiliation,
            'MEANMALICE': malice,
            'MEANDECEPTION': max(1, min(10, int(np.random.exponential(1.4) * cyber_multiplier))),
            'MEANUNWANTEDCONTACT': unwanted_contact,
            'Depression': depression,
            'Stress': stress,
            'Anxiety': anxiety,
            'Bullied': bullied,
            'Insecure': insecure
        })
    
    return pd.DataFrame(data)

@st.cache_data
def load_and_prepare_data():
    """Load or generate data and prepare it for modeling"""
    
    # Try to load existing dataset first
    if os.path.exists("mental_health_dataset_with_labels.csv"):
        try:
            df = pd.read_csv("mental_health_dataset_with_labels.csv")
            st.info("✅ Loaded existing dataset")
        except:
            df = generate_synthetic_data()
            st.info("🔄 Generated synthetic dataset (original file corrupted)")
    else:
        df = generate_synthetic_data()
        st.info("🔄 Generated synthetic dataset (no existing file found)")
    
    # Clean the data
    df = df.dropna()
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['Depression', 'Stress', 'Anxiety', 'Bullied', 'Insecure']]
    X = df[feature_cols].copy()
    
    # Encode categorical variables
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    return df, X, label_encoders

@st.cache_data
def train_models(X, y_dict):
    """Train multiple models and return their performance metrics"""
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, max_depth=6),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Support Vector Machine': SVC(random_state=42, probability=True, kernel='rbf')
    }
    
    results = {}
    model_performance = {}
    
    for target in ['Depression', 'Stress', 'Anxiety']:
        y = y_dict[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features for SVM and Logistic Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        target_results = {}
        target_performance = {}
        
        for name, model in models.items():
            # Use scaled data for SVM and Logistic Regression
            if name in ['Support Vector Machine', 'Logistic Regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train_scaled if name in ['Support Vector Machine', 'Logistic Regression'] else X_train, y_train, cv=5)
            
            target_results[name] = {
                'model': model,
                'scaler': scaler if name in ['Support Vector Machine', 'Logistic Regression'] else None,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_prob,
                'actual': y_test
            }
            
            target_performance[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
        
        results[target] = target_results
        model_performance[target] = target_performance
    
    return results, model_performance

# --- Enhanced Chatbot ---
class EnhancedMentalHealthChatbot:
    def __init__(self):
        self.responses = {
            "hello": [
                "Hi there! 😊 I'm here to support you. How are you feeling today?",
                "Hello! It's great that you're here. What's on your mind?",
                "Hey! I'm glad you decided to chat. How can I help you today?"
            ],
            "stressed": [
                "I hear that you're feeling stressed. That's completely valid. 🌱 What's been weighing on your mind?",
                "Stress can be overwhelming. Let's take it one step at a time. What's causing you the most stress right now?",
                "It sounds like you're carrying a lot right now. Remember, it's okay to take breaks. 💙"
            ],
            "anxiety": [
                "Anxiety can feel really intense. You're brave for reaching out. 🤗 What usually helps you feel calmer?",
                "I understand anxiety can be frightening. Try focusing on your breathing - in for 4, hold for 4, out for 4. 🌸",
                "Anxious feelings are valid. Remember, you've gotten through difficult times before, and you can do it again. 💪"
            ],
            "depression": [
                "Thank you for sharing this with me. Depression is real, and your feelings matter. 🤍 Have you been able to talk to someone you trust?",
                "I'm sorry you're going through this. Please remember that you matter, and there are people who want to help. 🌟",
                "Depression can make everything feel heavy. Small steps count - even just talking here is a positive step. 🌱"
            ],
            "bullied": [
                "I'm so sorry you're experiencing bullying. This is never your fault, and you don't deserve this treatment. 🛡️",
                "Bullying is serious and wrong. Have you been able to talk to a trusted adult about this situation?",
                "You're strong for enduring this, but you shouldn't have to face it alone. Please reach out to someone who can help. 💙"
            ],
            "insecure": [
                "Those feelings of insecurity are more common than you might think. You have value exactly as you are. ✨",
                "Everyone has moments of self-doubt. What's one thing you like about yourself?",
                "Insecurity is often our inner critic being too harsh. Try speaking to yourself like you would a good friend. 💕"
            ],
            "help": [
                "I'm here to listen and provide support. You can talk to me about stress, anxiety, depression, or anything else on your mind. 🤝",
                "There are many ways to get help - talking to friends, family, counselors, or calling helplines. What feels most comfortable for you?",
                "Remember, seeking help is a sign of strength, not weakness. You're taking care of yourself. 🌟"
            ],
            "default": [
                "I'm here to listen. What would you like to talk about?",
                "Tell me more about what's going on. I'm here to support you. 💙",
                "Sometimes just talking can help. What's on your mind today?"
            ]
        }
        
        self.coping_strategies = [
            "🧘 Try deep breathing exercises",
            "🚶 Take a short walk outside",
            "📝 Write down your thoughts",
            "🎵 Listen to calming music",
            "🤗 Reach out to a friend or family member",
            "🛀 Take a warm bath or shower",
            "📖 Read something positive",
            "🎨 Try a creative activity"
        ]

    def get_response(self, message):
        message_lower = message.lower()
        
        # Check for specific keywords
        for key in self.responses:
            if any(keyword in message_lower for keyword in key.split()):
                response = random.choice(self.responses[key])
                
                # Add coping strategy for stress/anxiety/depression
                if key in ["stressed", "anxiety", "depression"]:
                    strategy = random.choice(self.coping_strategies)
                    response += f"\n\nHere's a coping strategy you might try: {strategy}"
                
                return response
        
        return random.choice(self.responses["default"])

# --- Enhanced Visualization Functions ---
def create_model_comparison_chart(performance_data):
    """Create an interactive comparison chart for all models"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Model Accuracy Comparison', 'Cross-Validation Scores', 'Depression Models', 'Stress & Anxiety Models'],
        specs=[[{"colspan": 2}, None], [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Overall comparison
    models = list(performance_data['Depression'].keys())
    depression_acc = [performance_data['Depression'][model]['accuracy'] for model in models]
    stress_acc = [performance_data['Stress'][model]['accuracy'] for model in models]
    anxiety_acc = [performance_data['Anxiety'][model]['accuracy'] for model in models]
    
    fig.add_trace(go.Bar(name='Depression', x=models, y=depression_acc), row=1, col=1)
    fig.add_trace(go.Bar(name='Stress', x=models, y=stress_acc), row=1, col=1)
    fig.add_trace(go.Bar(name='Anxiety', x=models, y=anxiety_acc), row=1, col=1)
    
    # CV scores for Depression
    cv_means = [performance_data['Depression'][model]['cv_mean'] for model in models]
    cv_stds = [performance_data['Depression'][model]['cv_std'] for model in models]
    
    fig.add_trace(go.Bar(
        name='CV Score',
        x=models,
        y=cv_means,
        error_y=dict(type='data', array=cv_stds),
        showlegend=False
    ), row=2, col=1)
    
    # Combined Stress & Anxiety
    fig.add_trace(go.Bar(name='Stress', x=models, y=stress_acc, showlegend=False), row=2, col=2)
    fig.add_trace(go.Bar(name='Anxiety', x=models, y=anxiety_acc, showlegend=False), row=2, col=2)
    
    fig.update_layout(height=800, title_text="Model Performance Analytics")
    fig.update_yaxes(title_text="Accuracy", range=[0, 1])
    
    return fig

def create_prediction_distribution(predictions_dict):
    """Create distribution plots for predictions"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['Depression Distribution', 'Stress Distribution', 'Anxiety Distribution']
    )
    
    severity_labels = ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe']
    
    for i, (condition, pred_data) in enumerate(predictions_dict.items(), 1):
        # Get the best model's predictions
        best_model = max(pred_data.keys(), key=lambda x: pred_data[x]['accuracy'])
        predictions = pred_data[best_model]['predictions']
        
        # Count distribution
        unique, counts = np.unique(predictions, return_counts=True)
        labels = [severity_labels[int(val)-1] for val in unique]
        
        fig.add_trace(go.Bar(
            x=labels,
            y=counts,
            name=condition,
            showlegend=False
        ), row=1, col=i)
    
    fig.update_layout(height=400, title_text="Prediction Distributions (Best Models)")
    return fig

# --- Enhanced Recommendations ---
def get_enhanced_recommendations(stress, anxiety, depression, demographics=None):
    """Provide personalized recommendations based on assessment results and demographics"""
    recommendations = []
    severity_labels = ["Normal", "Mild", "Moderate", "Severe", "Extremely Severe"]
    
    # Adjust indices for 1-based to 0-based
    stress_level = severity_labels[min(max(stress - 1, 0), 4)]
    anxiety_level = severity_labels[min(max(anxiety - 1, 0), 4)]
    depression_level = severity_labels[min(max(depression - 1, 0), 4)]
    
    # Overall assessment
    recommendations.append(f"**📊 Your Mental Health Assessment:**")
    recommendations.append(f"• Stress: {stress_level}")
    recommendations.append(f"• Anxiety: {anxiety_level}")  
    recommendations.append(f"• Depression: {depression_level}")
    recommendations.append("")
    
    # Specific recommendations based on levels
    if stress >= 4:
        recommendations.extend([
            "**🚨 High Stress Level - Immediate Actions:**",
            "• Consider speaking with a mental health professional",
            "• Practice stress-reduction techniques daily",
            "• Identify and address major stressors in your life",
            "• Consider meditation or mindfulness apps like Headspace or Calm"
        ])
    elif stress >= 3:
        recommendations.extend([
            "**⚠️ Moderate Stress - Management Strategies:**",
            "• Implement regular exercise routine",
            "• Practice time management and prioritization",
            "• Try progressive muscle relaxation",
            "• Maintain work-life balance"
        ])
    elif stress >= 2:
        recommendations.extend([
            "**🌱 Mild Stress - Preventive Care:**",
            "• Continue current coping strategies",
            "• Build resilience through regular self-care",
            "• Practice stress-prevention techniques"
        ])
    
    if anxiety >= 4:
        recommendations.extend([
            "**🚨 High Anxiety Level - Seek Support:**",
            "• Professional counseling is strongly recommended",
            "• Consider anxiety-specific therapy (CBT)",
            "• Practice grounding techniques (5-4-3-2-1 method)",
            "• Limit caffeine and practice good sleep hygiene"
        ])
    elif anxiety >= 3:
        recommendations.extend([
            "**⚠️ Moderate Anxiety - Active Management:**",
            "• Try deep breathing exercises regularly",
            "• Challenge negative thought patterns",
            "• Consider anxiety support groups",
            "• Practice mindfulness meditation"
        ])
    
    if depression >= 4:
        recommendations.extend([
            "**🚨 Severe Depression - Professional Help Needed:**",
            "• Please reach out to a mental health professional immediately",
            "• Contact a crisis helpline if needed",
            "• Inform trusted friends or family about how you're feeling",
            "• Consider both therapy and medication consultation"
        ])
    elif depression >= 3:
        recommendations.extend([
            "**⚠️ Moderate Depression - Support Systems:**",
            "• Maintain social connections",
            "• Engage in enjoyable activities regularly",
            "• Consider counseling or therapy",
            "• Practice self-compassion and patience"
        ])
    
    # General wellness recommendations
    recommendations.extend([
        "",
        "**🌟 General Wellness Tips:**",
        "• Maintain regular sleep schedule (7-9 hours)",
        "• Exercise regularly (at least 30 minutes, 3x/week)",
        "• Eat nutritious, balanced meals",
        "• Stay connected with supportive people",
        "• Limit social media if it increases distress",
        "• Practice gratitude daily"
    ])
    
    # Emergency resources
    if any(level >= 4 for level in [stress, anxiety, depression]):
        recommendations.extend([
            "",
            "**🆘 Emergency Resources:**",
            "• National Suicide Prevention Lifeline: 988",
            "• Crisis Text Line: Text HOME to 741741",
            "• Emergency Services: 911",
            "• If you're in immediate danger, don't hesitate to call for help"
        ])
    
    return recommendations

# --- Input Preprocessing ---
def preprocess_user_input(input_data, label_encoders):
    """Preprocess user input to match model training format"""
    processed_data = input_data.copy()
    
    # Apply label encoders
    for col, encoder in label_encoders.items():
        if col in processed_data:
            try:
                processed_data[col] = encoder.transform([str(processed_data[col])])[0]
            except ValueError:
                # Handle unseen categories by using the most frequent category
                processed_data[col] = encoder.transform([encoder.classes_[0]])[0]
    
    return processed_data

# --- Main Application ---
def main():
    st.set_page_config(
        page_title="🧠 MindCare Pro - Advanced Mental Health Assessment",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4A90E2;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Load data and train models
    try:
        with st.spinner("🔄 Loading data and training models..."):
            df, X, label_encoders = load_and_prepare_data()
            y_dict = {
                'Depression': df['Depression'],
                'Stress': df['Stress'], 
                'Anxiety': df['Anxiety']
            }
            model_results, model_performance = train_models(X, y_dict)
        
        st.success("✅ Models trained successfully!")
        
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()
    
    # Initialize chatbot
    chatbot = EnhancedMentalHealthChatbot()
    
    # Sidebar navigation
    st.sidebar.title("🧠 MindCare Pro")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["🏠 Home", "📊 Assessment", "🤖 AI Chatbot", "📈 Model Analytics", "🆘 Emergency Resources"]
    )
    
    if page == "🏠 Home":
        st.markdown('<h1 class="main-header">🧠 MindCare Pro</h1>', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: center; color: #666;">Advanced Mental Health Assessment & Support System</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-container">
            <h4>🎯 Accurate Assessment</h4>
            <p>Uses 4 advanced ML models for precise mental health evaluation</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-container">
            <h4>🤖 AI Support</h4>
            <p>24/7 intelligent chatbot for emotional support and guidance</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-container">
            <h4>📊 Analytics</h4>
            <p>Comprehensive model performance and prediction analytics</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        with st.expander("🔍 How It Works", expanded=True):
            st.markdown("""
            **MindCare Pro** uses advanced machine learning to assess your mental wellbeing:
            
            1. **📝 Complete Assessment** - Answer questions about demographics, social media use, and experiences
            2. **🤖 AI Analysis** - 4 different ML models analyze your responses
            3. **📊 Get Results** - Receive detailed insights about stress, anxiety, and depression levels
            4. **💬 Chat Support** - Talk with our AI chatbot for additional support
            5. **📈 Track Progress** - View analytics and model performance metrics
            
            *This tool is for informational purposes and should not replace professional medical advice.*
            """)
        
        # Dataset overview
        st.subheader("📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Features", len(X.columns))
        with col3:
            st.metric("Avg Depression Level", f"{df['Depression'].mean():.2f}")
        with col4:
            st.metric("Model Accuracy", f"{np.mean([v['Random Forest']['accuracy'] for v in model_performance.values()]):.3f}")
    
    elif page == "📊 Assessment":
        st.title("📊 Comprehensive Mental Health Assessment")
        st.markdown("Please answer the following questions honestly for the most accurate assessment.")
        
        with st.form("mental_health_assessment"):
            st.subheader("👤 Demographics")
            col1, col2 = st.columns(2)
            
            with col1:
                gender = st.selectbox("Gender", ["Female", "Male", "Non-binary"])
                age = st.selectbox("Age Group", ["<18 years old", "19-24 years old", "25-34 years old", "35-44 years old", ">45 years old"])
                marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
                
            with col2:
                education = st.selectbox("Education Level", ["High School", "Diploma/Foundation", "Bachelor Degree", "Master Degree", "PhD"])
                university = st.selectbox("Institution Type", ["Public Universities", "Private Universities", "Community College"])
                
            st.subheader("📱 Digital Behavior")
            col3, col4 = st.columns(2)
            
            with col3:
                social_media_activity = st.selectbox("Social Media Activity Level", ["Less active", "Moderately active", "Active", "Very active"])
                time_spent = st.selectbox("Daily Social Media Time", ["<1 hour", "1-2 hours", "3-4 hours", "5-6 hours", "7-10 hours", ">10 hours"])
                
            st.subheader("🛡️ Online Experiences")
            st.markdown("*Rate your experiences on a scale of 1-10 (1 = Never/Minimal, 10 = Frequently/Severe)*")
            
            col5, col6 = st.columns(2)
            with col5:
                cyber_exp = st.slider("Overall Cyberbullying Experience", 1, 10, 3, help="General exposure to online harassment")
                public_humiliation = st.slider("Public Humiliation Online", 1, 10, 2, help="Being embarrassed or shamed publicly online")
                
            with col6:
                unwanted_contact = st.slider("Unwanted Contact", 1, 10, 2, help="Receiving unwanted messages or contact")
                malice_exp = st.slider("Malicious Behavior Exposure", 1, 10, 2, help="Encountering deliberately harmful online behavior")
            
            deception_exp = st.slider("Online Deception Experience", 1, 10, 2, help="Encountering fake profiles, scams, or lies online")
            
            submitted = st.form_submit_button("🔍 Analyze My Mental Health", use_container_width=True)
        
        if submitted:
            with st.spinner("🧠 Analyzing your responses with AI models..."):
                # Prepare input data
                input_data = {
                    'GENDER': gender,
                    'AGE': age,
                    'MARITAL_STATUS': marital_status,
                    'EDUCATION_LEVEL': education,
                    'UNIVERSITY_STATUS': university,
                    'ACTIVE_SOCIAL_MEDIA': social_media_activity,
                    'TIME_SPENT_SOCIAL_MEDIA': time_spent,
                    'CVTOTAL': cyber_exp * 10,
                    'CVPUBLICHUMILIATION': public_humiliation * 5,
                    'CVMALICE': malice_exp * 5,
                    'CVUNWANTEDCONTACT': unwanted_contact * 5,
                    'MEANPUBLICHUMILIATION': public_humiliation,
                    'MEANMALICE': malice_exp,
                    'MEANDECEPTION': deception_exp,
                    'MEANUNWANTEDCONTACT': unwanted_contact
                }
                
                # Preprocess input
                processed_input = preprocess_user_input(input_data, label_encoders)
                input_df = pd.DataFrame([processed_input])
                
                # Get predictions from all models
                predictions = {}
                for condition in ['Depression', 'Stress', 'Anxiety']:
                    condition_preds = {}
                    for model_name, model_data in model_results[condition].items():
                        model = model_data['model']
                        scaler = model_data['scaler']
                        
                        if scaler is not None:
                            input_scaled = scaler.transform(input_df)
                            pred = model.predict(input_scaled)[0]
                            prob = model.predict_proba(input_scaled)[0]
                        else:
                            pred = model.predict(input_df)[0]
                            prob = model.predict_proba(input_df)[0]
                        
                        condition_preds[model_name] = {
                            'prediction': int(pred),
                            'probability': prob,
                            'confidence': max(prob)
                        }
                    predictions[condition] = condition_preds
                
                # Display results
                st.success("✅ Analysis Complete!")
                
                # Get best model predictions (highest accuracy)
                best_predictions = {}
                for condition in ['Depression', 'Stress', 'Anxiety']:
                    best_model = max(model_performance[condition].keys(), 
                                   key=lambda x: model_performance[condition][x]['accuracy'])
                    best_predictions[condition] = predictions[condition][best_model]['prediction']
                
                # Main results display
                st.subheader("🎯 Your Mental Health Assessment Results")
                
                severity_labels = ["Normal", "Mild", "Moderate", "Severe", "Extremely Severe"]
                colors = ["#28a745", "#ffc107", "#fd7e14", "#dc3545", "#6f42c1"]
                
                col1, col2, col3 = st.columns(3)
                
                for i, (condition, pred) in enumerate(best_predictions.items()):
                    severity = severity_labels[pred - 1]
                    color = colors[pred - 1]
                    
                    if i == 0:
                        col = col1
                    elif i == 1:
                        col = col2
                    else:
                        col = col3
                    
                    with col:
                        st.markdown(f"""
                        <div style="background: linear-gradient(90deg, {color}22, {color}11); 
                                   border-left: 4px solid {color}; 
                                   padding: 1rem; 
                                   border-radius: 8px; 
                                   margin: 0.5rem 0;">
                            <h4 style="color: {color}; margin: 0;">{condition}</h4>
                            <h2 style="margin: 0.5rem 0;">{severity}</h2>
                            <small>Level {pred}/5</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Model comparison for this prediction
                st.subheader("🤖 Model Predictions Comparison")
                
                comparison_data = []
                for condition in ['Depression', 'Stress', 'Anxiety']:
                    for model_name, pred_data in predictions[condition].items():
                        comparison_data.append({
                            'Condition': condition,
                            'Model': model_name,
                            'Prediction': pred_data['prediction'],
                            'Confidence': pred_data['confidence'],
                            'Severity': severity_labels[pred_data['prediction'] - 1]
                        })
                
                comparison_df = pd.DataFrame(comparison_data)
                
                # Create interactive visualization
                fig = px.bar(comparison_df, 
                           x='Model', 
                           y='Prediction', 
                           color='Condition',
                           title='Predictions Across All Models',
                           hover_data=['Severity', 'Confidence'])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed model analysis
                with st.expander("📊 Detailed Model Analysis"):
                    for condition in ['Depression', 'Stress', 'Anxiety']:
                        st.write(f"**{condition} Analysis:**")
                        model_df = pd.DataFrame([
                            {
                                'Model': name,
                                'Prediction': data['prediction'],
                                'Severity': severity_labels[data['prediction'] - 1],
                                'Confidence': f"{data['confidence']:.3f}",
                                'Accuracy': f"{model_performance[condition][name]['accuracy']:.3f}"
                            }
                            for name, data in predictions[condition].items()
                        ])
                        st.dataframe(model_df, use_container_width=True)
                        st.write("")
                
                # Personalized recommendations
                st.subheader("💡 Personalized Recommendations")
                recommendations = get_enhanced_recommendations(
                    best_predictions['Stress'],
                    best_predictions['Anxiety'], 
                    best_predictions['Depression'],
                    demographics={'age': age, 'gender': gender}
                )
                
                for rec in recommendations:
                    if rec.strip():  # Skip empty lines
                        st.markdown(rec)
                
                # Risk assessment
                max_level = max(best_predictions.values())
                if max_level >= 4:
                    st.error("⚠️ **High Risk Detected**: Your assessment indicates concerning levels. Please consider seeking professional help immediately.")
                elif max_level >= 3:
                    st.warning("⚠️ **Moderate Risk**: Your results suggest you might benefit from professional support.")
                else:
                    st.success("✅ **Low Risk**: Your mental health appears to be in a good range. Keep up the self-care!")
    
    elif page == "🤖 AI Chatbot":
        st.title("🤖 MindCare AI Assistant")
        st.markdown("I'm here to listen and provide support. Feel free to share what's on your mind.")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm your MindCare AI assistant. 😊 I'm here to provide emotional support and listen to whatever you'd like to share. How are you feeling today?"}
            ]
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chatbot.get_response(prompt)
                st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Sidebar with helpful information
        with st.sidebar:
            st.markdown("### 💡 Chat Tips")
            st.markdown("""
            - Be honest about your feelings
            - Take your time to express yourself
            - Ask for coping strategies
            - Discuss specific concerns
            - Request resources or support
            """)
            
            st.markdown("### 🆘 Quick Help")
            if st.button("I'm feeling stressed"):
                st.session_state.messages.append({"role": "user", "content": "I'm feeling very stressed"})
                st.rerun()
            
            if st.button("I have anxiety"):
                st.session_state.messages.append({"role": "user", "content": "I'm experiencing anxiety"})
                st.rerun()
                
            if st.button("I feel sad"):
                st.session_state.messages.append({"role": "user", "content": "I'm feeling really sad and down"})
                st.rerun()
            
            if st.button("Clear Chat"):
                st.session_state.messages = [
                    {"role": "assistant", "content": "Hello! I'm your MindCare AI assistant. 😊 How can I help you today?"}
                ]
                st.rerun()
    
    elif page == "📈 Model Analytics":
        st.title("📈 Model Performance Analytics")
        st.markdown("Comprehensive analysis of all machine learning models used in the assessment.")
        
        # Overall performance metrics
        st.subheader("🎯 Model Performance Overview")
        
        # Create performance summary
        performance_summary = []
        for condition in ['Depression', 'Stress', 'Anxiety']:
            for model_name, metrics in model_performance[condition].items():
                performance_summary.append({
                    'Condition': condition,
                    'Model': model_name,
                    'Accuracy': metrics['accuracy'],
                    'CV Mean': metrics['cv_mean'],
                    'CV Std': metrics['cv_std']
                })
        
        summary_df = pd.DataFrame(performance_summary)
        
        # Display metrics table
        st.dataframe(
            summary_df.pivot(index='Model', columns='Condition', values='Accuracy').round(4),
            use_container_width=True
        )
        
        # Model comparison visualization
        st.subheader("📊 Interactive Model Comparison")
        comparison_chart = create_model_comparison_chart(model_performance)
        st.plotly_chart(comparison_chart, use_container_width=True)
        
        # Best models summary
        st.subheader("🏆 Best Performing Models")
        col1, col2, col3 = st.columns(3)
        
        for i, condition in enumerate(['Depression', 'Stress', 'Anxiety']):
            best_model = max(model_performance[condition].keys(), 
                           key=lambda x: model_performance[condition][x]['accuracy'])
            best_accuracy = model_performance[condition][best_model]['accuracy']
            
            if i == 0:
                col = col1
            elif i == 1:
                col = col2
            else:
                col = col3
            
            with col:
                st.metric(
                    f"Best {condition} Model",
                    best_model,
                    f"{best_accuracy:.4f} accuracy"
                )
        
        # Detailed model analysis
        st.subheader("🔍 Detailed Model Analysis")
        
        selected_condition = st.selectbox("Select Condition for Detailed Analysis:", 
                                        ['Depression', 'Stress', 'Anxiety'])
        
        # Cross-validation scores visualization
        cv_data = []
        for model_name, metrics in model_performance[selected_condition].items():
            cv_data.append({
                'Model': model_name,
                'CV Mean': metrics['cv_mean'],
                'CV Std': metrics['cv_std']
            })
        
        cv_df = pd.DataFrame(cv_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cv_df['Model'],
            y=cv_df['CV Mean'],
            error_y=dict(type='data', array=cv_df['CV Std']),
            name='Cross-Validation Score'
        ))
        
        fig.update_layout(
            title=f'{selected_condition} Model Cross-Validation Scores',
            xaxis_title='Model',
            yaxis_title='Accuracy Score',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model interpretability
        with st.expander("🧠 Model Interpretability & Features"):
            st.markdown("""
            **Key Features Contributing to Predictions:**
            
            1. **Cyberbullying Experiences** - Direct correlation with mental health outcomes
            2. **Social Media Usage** - Both time spent and activity level impact wellbeing  
            3. **Demographics** - Age and gender influence vulnerability patterns
            4. **Educational Status** - Correlates with coping mechanisms and resources
            5. **Online Harassment Metrics** - Various forms of digital mistreatment
            
            **Model Strengths:**
            - **Random Forest**: Best overall performance, handles feature interactions well
            - **Gradient Boosting**: Strong at capturing non-linear relationships  
            - **Logistic Regression**: Provides interpretable coefficients
            - **SVM**: Effective for complex decision boundaries
            """)
        
        # Dataset insights
        st.subheader("📊 Dataset Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution of mental health conditions
            condition_counts = {}
            for condition in ['Depression', 'Stress', 'Anxiety']:
                counts = df[condition].value_counts().sort_index()
                condition_counts[condition] = counts
            
            fig = go.Figure()
            for condition, counts in condition_counts.items():
                fig.add_trace(go.Bar(
                    x=[f"Level {i}" for i in counts.index],
                    y=counts.values,
                    name=condition
                ))
            
            fig.update_layout(
                title='Distribution of Mental Health Conditions',
                xaxis_title='Severity Level',
                yaxis_title='Count',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Correlation heatmap
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                title='Feature Correlation Matrix',
                color_continuous_scale='RdBu_r',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Model deployment info
        with st.expander("🚀 Model Deployment Information"):
            st.markdown(f"""
            **Training Dataset**: {len(df):,} samples with {len(X.columns)} features
            
            **Model Training Details**:
            - **Random Forest**: 100 trees, max depth 10
            - **Gradient Boosting**: Default parameters, max depth 6  
            - **Logistic Regression**: L2 regularization, max iter 1000
            - **SVM**: RBF kernel, probability estimates enabled
            
            **Data Preprocessing**:
            - Label encoding for categorical variables
            - Standard scaling for SVM and Logistic Regression
            - 80/20 train-test split with stratification
            - 5-fold cross-validation for model evaluation
            
            **Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """)
    
    elif page == "🆘 Emergency Resources":
        st.title("🆘 Emergency Mental Health Resources")
        
        st.error("**⚠️ If you are in immediate danger or having thoughts of self-harm, please contact emergency services immediately.**")
        
        # Crisis hotlines
        st.subheader("📞 Crisis Hotlines (24/7)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🇺🇸 United States:**
            - **988 Suicide & Crisis Lifeline**: 988
            - **Crisis Text Line**: Text HOME to 741741
            - **National Suicide Prevention Lifeline**: 1-800-273-8255
            - **SAMHSA National Helpline**: 1-800-662-4357
            
            **🇬🇧 United Kingdom:**
            - **Samaritans**: 116 123
            - **Crisis Text Line UK**: Text SHOUT to 85258
            - **NHS 111**: 111
            """)
        
        with col2:
            st.markdown("""
            **🇮🇳 India:**
            - **iCall (TISS)**: +91 9152987821
            - **AASRA**: 91-22-27546669
            - **Vandrevala Foundation**: 1860 266 2345
            - **Sneha Foundation**: +91-44-24640050
            
            **🇨🇦 Canada:**
            - **Canada Suicide Prevention Service**: 1-833-456-4566
            - **Kids Help Phone**: 1-800-668-6868
            - **Crisis Services Canada**: 1-833-456-4566
            """)
        
        # Online resources
        st.subheader("🌐 Online Mental Health Resources")
        
        resources_data = [
            {"Platform": "BetterHelp", "Type": "Online Therapy", "Description": "Professional counseling via video, phone, or text"},
            {"Platform": "Talkspace", "Type": "Online Therapy", "Description": "Text-based therapy with licensed professionals"},
            {"Platform": "7 Cups", "Type": "Peer Support", "Description": "Free emotional support from trained listeners"},
            {"Platform": "NAMI", "Type": "Education & Support", "Description": "Mental health awareness and support groups"},
            {"Platform": "Mental Health America", "Type": "Resources", "Description": "Screening tools and mental health information"},
            {"Platform": "Crisis Text Line", "Type": "Crisis Support", "Description": "24/7 text-based crisis counseling"}
        ]
        
        resources_df = pd.DataFrame(resources_data)
        st.dataframe(resources_df, use_container_width=True)
        
        # Self-help strategies
        st.subheader("🛠️ Immediate Self-Help Strategies")
        
        tab1, tab2, tab3 = st.tabs(["🧘 Anxiety", "😔 Depression", "😰 Stress"])
        
        with tab1:
            st.markdown("""
            **For Anxiety:**
            - **5-4-3-2-1 Grounding**: Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste
            - **Deep Breathing**: Breathe in for 4 counts, hold for 4, exhale for 6
            - **Progressive Muscle Relaxation**: Tense and release each muscle group
            - **Mindfulness Apps**: Headspace, Calm, Insight Timer
            - **Avoid Caffeine**: Can worsen anxiety symptoms
            """)
        
        with tab2:
            st.markdown("""
            **For Depression:**
            - **Small Steps**: Set tiny, achievable daily goals
            - **Sunlight Exposure**: Spend time outdoors when possible
            - **Physical Activity**: Even a 5-minute walk can help
            - **Social Connection**: Reach out to one person today
            - **Routine**: Maintain regular sleep and meal schedules
            """)
        
        with tab3:
            st.markdown("""
            **For Stress:**
            - **Time Management**: Break large tasks into smaller ones
            - **Prioritization**: Focus on what's most important
            - **Boundary Setting**: Learn to say no when needed
            - **Physical Release**: Exercise, stretching, or dancing
            - **Relaxation Techniques**: Meditation, yoga, or warm baths
            """)
        
        # Warning signs
        st.subheader("⚠️ Warning Signs - Seek Immediate Help")
        
        st.warning("""
        **Contact emergency services (911/999/112) or go to the nearest emergency room if you experience:**
        
        - Thoughts of harming yourself or others
        - Detailed plans for suicide
        - Hearing voices or seeing things others don't
        - Severe confusion or inability to think clearly
        - Extreme agitation or violent behavior
        - Inability to care for yourself
        - Substance abuse emergency
        """)
        
        # Professional help guidance
        st.subheader("👩‍⚕️ When to Seek Professional Help")
        
        st.info("""
        **Consider professional help if you experience:**
        
        - Persistent sadness or hopelessness for more than 2 weeks
        - Significant changes in sleep, appetite, or energy
        - Difficulty concentrating or making decisions
        - Loss of interest in activities you used to enjoy
        - Persistent anxiety or panic attacks
        - Relationship or work problems due to mental health
        - Using alcohol or drugs to cope
        - Family or friends expressing concern about your wellbeing
        """)

if __name__ == "__main__":
    main()






