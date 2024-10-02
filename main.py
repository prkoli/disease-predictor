import streamlit as st
import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Database setup
conn = sqlite3.connect('disease_data.db')
c = conn.cursor()

# Create a table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS patients (
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             fever INTEGER,
             cough INTEGER,
             fatigue INTEGER,
             difficulty_breathing INTEGER,
             age INTEGER,
             gender INTEGER,
             blood_pressure INTEGER,
             cholesterol_level INTEGER,
             disease TEXT
             )''')

# Load dataset
df = pd.read_csv('Disease_symptom_and_patient_profile_dataset.csv')

# Encode binary values to 1 and 0
binary_columns = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']
for col in binary_columns:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

label_encoder = LabelEncoder()
df['Blood Pressure'] = label_encoder.fit_transform(df['Blood Pressure'])
df['Cholesterol Level'] = label_encoder.fit_transform(df['Cholesterol Level'])

# Encode gender as a binary variable
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Define features and target variable
X = df.drop(columns=['Disease', 'Outcome Variable'])
y = df['Disease']

# Define feature names
feature_names = X.columns

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit UI
st.title("Disease Predictor System")

st.write("Please answer the following questions to predict the disease:")

fever = st.radio("Do you have fever?", ("Yes", "No"))
cough = st.radio("Do you have a cough?", ("Yes", "No"))
fatigue = st.radio("Do you experience fatigue?", ("Yes", "No"))
difficulty_breathing = st.radio("Do you have difficulty breathing?", ("Yes", "No"))

age = st.number_input("Please enter your age:")
gender = st.radio("Please enter your gender:", ("Male", "Female"))
blood_pressure = st.selectbox("Please enter your blood pressure level:", ("Low", "Normal", "High"))
cholesterol_level = st.selectbox("Please enter your cholesterol level:", ("Normal", "High"))

if st.button("Predict"):
    # Encode binary symptoms to 0 and 1
    fever_encoded = 1 if fever == 'Yes' else 0
    cough_encoded = 1 if cough == 'Yes' else 0
    fatigue_encoded = 1 if fatigue == 'Yes' else 0
    difficulty_breathing_encoded = 1 if difficulty_breathing == 'Yes' else 0

    # Encode gender, blood pressure, and cholesterol level
    gender_encoded = 1 if gender == 'Male' else 0
    blood_pressure_encoded = label_encoder.transform([blood_pressure])[0]
    cholesterol_level_encoded = label_encoder.transform([cholesterol_level])[0]

    input_data = [fever_encoded, cough_encoded, fatigue_encoded, difficulty_breathing_encoded, age, gender_encoded, blood_pressure_encoded, cholesterol_level_encoded]

    # Predict disease
    disease_prediction = model.predict([input_data])[0]
    st.write(f"Predicted Disease: {disease_prediction}")

    # Insert patient data into the database
    c.execute('INSERT INTO patients (fever, cough, fatigue, difficulty_breathing, age, gender, blood_pressure, cholesterol_level, disease) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
              (fever_encoded, cough_encoded, fatigue_encoded, difficulty_breathing_encoded, age, gender_encoded, blood_pressure_encoded, cholesterol_level_encoded, disease_prediction))
    conn.commit()
    st.write("Patient data has been saved to the database.")

# Query database
if st.button("Show Database Records"):
    c.execute("SELECT * FROM patients")
    rows = c.fetchall()
    st.write(pd.DataFrame(rows, columns=["ID", "Fever", "Cough", "Fatigue", "Difficulty Breathing", "Age", "Gender", "Blood Pressure", "Cholesterol Level", "Disease"]))
