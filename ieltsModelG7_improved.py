import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Hide yellow warnings for a clean report
warnings.filterwarnings("ignore", category=UserWarning)

# 2. Load Data
df = pd.read_csv('expanded_ielts_dataset.csv')

# 3. Feature Engineering
df['Score_Gain'] = df['Overall_Band'] - df['Entry_Overall']

# NEW: Extra engineered features to help the model find patterns
df['Study_x_Motivation']    = df['Study_Hours_Per_Week'] * df['Motivation_Score']
df['MockTest_x_Attendance'] = df['Mock_Tests_Count'] * df['Attendance_Rate']
df['Low_Entry']             = (df['Entry_Overall'] < 5.5).astype(int)

# 4. Define Features (Inputs) and Target (Output)
features = [
    'Major', 'Entry_Overall', 'Study_Hours_Per_Week',
    'Mock_Tests_Count', 'Anxiety_Level', 'Motivation_Score', 'Attendance_Rate',
    'Study_x_Motivation', 'MockTest_x_Attendance', 'Low_Entry'   # NEW
]
X = df[features]
y = df['Score_Gain']

# 5. Preprocessing (Turning text into numbers)
num_cols = [
    'Entry_Overall', 'Study_Hours_Per_Week', 'Mock_Tests_Count',
    'Anxiety_Level', 'Motivation_Score', 'Attendance_Rate',
    'Study_x_Motivation', 'MockTest_x_Attendance', 'Low_Entry'
]
preprocessor = ColumnTransformer(transformers=[
    ('num', 'passthrough', num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Major'])
])

# 6. Build the Model — Ensemble of 3 models (more robust than one alone)
gbr   = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=3, min_samples_leaf=3,
            subsample=0.8, random_state=42)
rf    = RandomForestRegressor(
            n_estimators=200, max_depth=4,
            min_samples_leaf=3, random_state=42)
ridge = Ridge(alpha=1.0)

ensemble = VotingRegressor([('gbr', gbr), ('rf', rf), ('ridge', ridge)])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', ensemble)
])

# 7. Split and Train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# 8. Print Results to Terminal
print("\n" + "="*50)
print("   IELTS PERFORMANCE ANALYSIS RESULTS ")
print("="*50)
predictions = model.predict(X_test)
r2  = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
print(f"Model Accuracy (R2):       {r2:.4f}")
print(f"Average Prediction Error:  {mae:.4f} Bands")
print("="*50)

# 9. Generate Feature Importance Chart
# Average importances across the GBR and RF sub-models
cat_encoder = (model.named_steps['preprocessor']
               .named_transformers_['cat'])
major_names = cat_encoder.get_feature_names_out(['Major']).tolist()

num_labels = [
    'Entry Score', 'Study Hours', 'Mock Tests',
    'Anxiety', 'Motivation', 'Attendance',
    'Study x Motivation', 'MockTest x Attendance', 'Low Entry'
]
all_feature_names = num_labels + major_names

gbr_imp = model.named_steps['model'].estimators_[0].feature_importances_
rf_imp  = model.named_steps['model'].estimators_[1].feature_importances_
avg_imp = (gbr_imp + rf_imp) / 2

plt.figure(figsize=(12, 8))
sns.barplot(x=avg_imp, y=all_feature_names, palette='viridis')
plt.title('Analysis of Factors Improving IELTS Performance ',
          fontsize=15)
plt.xlabel('Impact Level (Feature Importance)', fontsize=12)
plt.ylabel('Factors & Student Backgrounds', fontsize=12)
plt.tight_layout()
plt.show()
new_student = pd.DataFrame([{
    'Major':                'Business',
    'Entry_Overall':         5.0,
    'Study_Hours_Per_Week':  8,
    'Mock_Tests_Count':      5,
    'Anxiety_Level':         4,
    'Motivation_Score':      8,
    'Attendance_Rate':       0.85,
    'Study_x_Motivation':    8 * 8,     
    'MockTest_x_Attendance': 5 * 0.85,  
    'Low_Entry':             1          
}])

# Ask the trained model to predict
predicted_gain = model.predict(new_student)
final_score    = 5.0 + predicted_gain[0]

print(f"Predicted score gain : +{predicted_gain[0]:.2f} bands")
print(f"Predicted final score: {final_score:.1f}")

