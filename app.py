import streamlit as st
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge

# Hide warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- PAGE SETUP ---
st.set_page_config(page_title="IELTS Predictor Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- 1. TRAIN & CACHE THE MODEL ---
# @st.cache_resource tells Streamlit to only train the model ONCE when the app starts.
# Otherwise, it would retrain every time you move a slider!
@st.cache_resource
def load_and_train_model():
    df = pd.read_csv('expanded_ielts_dataset.csv')
    
    # Feature Engineering
    df['Score_Gain'] = df['Overall_Band'] - df['Entry_Overall']
    df['Study_x_Motivation'] = df['Study_Hours_Per_Week'] * df['Motivation_Score']
    df['MockTest_x_Attendance'] = df['Mock_Tests_Count'] * df['Attendance_Rate']
    df['Low_Entry'] = (df['Entry_Overall'] < 5.5).astype(int)
    
    features = ['Major', 'Entry_Overall', 'Study_Hours_Per_Week', 'Mock_Tests_Count', 'Anxiety_Level', 'Motivation_Score', 'Attendance_Rate', 'Study_x_Motivation', 'MockTest_x_Attendance', 'Low_Entry']
    X = df[features]
    y = df['Score_Gain']
    
    num_cols = ['Entry_Overall', 'Study_Hours_Per_Week', 'Mock_Tests_Count', 'Anxiety_Level', 'Motivation_Score', 'Attendance_Rate', 'Study_x_Motivation', 'MockTest_x_Attendance', 'Low_Entry']
    preprocessor = ColumnTransformer(transformers=[
        ('num', 'passthrough', num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Major'])
    ])
    
    gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, min_samples_leaf=3, random_state=42)
    rf = RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=3, random_state=42)
    ridge = Ridge(alpha=1.0)
    
    ensemble = VotingRegressor([('gbr', gbr), ('rf', rf), ('ridge', ridge)])
    model = Pipeline(steps=[('preprocessor', preprocessor), ('model', ensemble)])
    
    model.fit(X, y)
    return model

model = load_and_train_model()

# --- 2. FRONTEND LAYOUT ---
st.title("📊 University Student IELTS Performance Predictor")
st.markdown("Adjust the student's profile on the left to see real-time score predictions and AI-driven factor analysis.")
st.markdown("---")

# Create two columns for our dashboard layout
col1, col2 = st.columns([1, 2])

# --- LEFT COLUMN: INPUT SLIDERS ---
with col1:
    st.subheader("Student Profile")
    major = st.selectbox("University Major", ["Business", "Engineering", "Linguistics", "Information Technology", "Social Sciences"])
    entry_band = st.slider("Entry Band", 4.0, 8.5, 5.0, 0.5)
    study_hours = st.slider("Study Hours / Week", 0, 30, 8)
    mock_tests = st.slider("Mock Tests Completed", 0, 12, 5)
    anxiety = st.slider("Anxiety Level (1=Low, 10=High)", 1, 10, 4)
    motivation = st.slider("Motivation Score (1=Low, 10=High)", 1, 10, 8)
    attendance_pct = st.slider("Class Attendance Rate (%)", 0, 100, 85)
    attendance_rate = attendance_pct / 100.0

# --- PREDICTION LOGIC ---
# Package the inputs exactly how the model expects them
new_student = pd.DataFrame([{
    'Major': major,
    'Entry_Overall': entry_band,
    'Study_Hours_Per_Week': study_hours,
    'Mock_Tests_Count': mock_tests,
    'Anxiety_Level': anxiety,
    'Motivation_Score': motivation,
    'Attendance_Rate': attendance_rate,
    'Study_x_Motivation': study_hours * motivation,
    'MockTest_x_Attendance': mock_tests * attendance_rate,
    'Low_Entry': 1 if entry_band < 5.5 else 0
}])

# Make the prediction
predicted_gain = model.predict(new_student)[0]
final_score = entry_band + predicted_gain

# --- RIGHT COLUMN: DASHBOARD OUTPUTS ---
with col2:
    st.subheader("Model Predictions")
    
    # 1. The Score Boxes
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric(label="Entry Band", value=f"{entry_band:.1f}")
    metric_col2.metric(label="Predicted Gain", value=f"{predicted_gain:+.2f}")
    metric_col3.metric(label="Final Target Band", value=f"{final_score:.1f}")
    
    st.markdown("---")
    
    # 2. Factor Analysis (Color-coded logic)
    st.subheader("Factor Analysis")
    
    def get_status(value, good_threshold, bad_threshold, is_reverse=False):
        if not is_reverse:
            if value >= good_threshold: return "🟢 Strong"
            elif value <= bad_threshold: return "🔴 Risk Factor"
            else: return "🟡 Neutral"
        else:
            if value <= good_threshold: return "🟢 Strong"
            elif value >= bad_threshold: return "🔴 Risk Factor"
            else: return "🟡 Neutral"

    st.write(f"**Study Hours:** {get_status(study_hours, 10, 4)}")
    st.write(f"**Mock Tests:** {get_status(mock_tests, 6, 2)}")
    st.write(f"**Attendance:** {get_status(attendance_pct, 85, 70)}")
    st.write(f"**Motivation:** {get_status(motivation, 8, 4)}")
    st.write(f"**Anxiety Level:** {get_status(anxiety, 3, 7, is_reverse=True)}")

    st.markdown("---")
    
    # 3. AI Analysis & Recommendation
    st.subheader("Advisory Analysis")
    
    # Dynamic text generation based on the worst factor
    risk_factors = []
    if anxiety >= 7: risk_factors.append("high exam anxiety")
    if attendance_pct <= 70: risk_factors.append("poor attendance")
    if mock_tests <= 2: risk_factors.append("a lack of mock test practice")
    if study_hours <= 4: risk_factors.append("insufficient study hours")
    
    if len(risk_factors) == 0:
        st.success("This student shows a highly optimal academic profile. The model predicts steady improvement. **Recommendation:** Continue current study habits and maintain momentum.")
    else:
        risks_joined = " and ".join(risk_factors[:2])
        st.warning(f"The model has identified a risk to the student's performance improvement, primarily driven by {risks_joined}.")
        
        # Give a specific recommendation based on the top risk
        st.write("**Targeted Recommendation for University Advisors:**")
        if "anxiety" in risk_factors[0]:
            st.write("Recommend the student visit the university counseling center for test-anxiety management techniques before scheduling their official exam.")
        elif "mock test" in risk_factors[0]:
            st.write("Require the student to complete at least two full-length, timed mock exams under simulated conditions this week.")
        elif "attendance" in risk_factors[0]:
            st.write("Intervene with an academic warning regarding attendance; the model heavily penalizes missed contact hours.")
        else:
            st.write("Help the student draft a weekly schedule to block out dedicated, uninterrupted study hours.")
