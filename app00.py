import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -------------------------
# Streamlit Page Settings
# -------------------------
st.set_page_config(page_title="AI Health Recommender", layout="wide")
st.title("🌍 SDG 3: Good Health & Well-Being — AI Recommender System")
st.markdown("""
This AI-powered app promotes **Good Health and Well-Being (UN SDG 3)**  
by predicting a user’s **Health Risk Level** and providing **personalized recommendations**.
""")

# -------------------------
# Internal Dataset (Training Data)
# -------------------------
# You can replace this with your own dataset of 250+ entries
data = pd.DataFrame({
    "Age": np.random.randint(18, 60, 250),
    "BMI": np.random.uniform(16, 35, 250),
    "ActivityLevel": np.random.choice(["Low", "Moderate", "High"], 250),
    "SleepHours": np.random.randint(4, 10, 250),
    "DietType": np.random.choice(["Vegetarian", "Non-Vegetarian", "Mixed"], 250),
    "HealthRisk": np.random.choice(["Low", "Medium", "High"], 250)
})

# Encode categorical features
df = data.copy()
le = LabelEncoder()
df["ActivityLevel"] = le.fit_transform(df["ActivityLevel"])
df["DietType"] = le.fit_transform(df["DietType"])
df["HealthRisk"] = le.fit_transform(df["HealthRisk"])

# -------------------------
# Model Training
# -------------------------
X = df[["Age", "BMI", "ActivityLevel", "SleepHours", "DietType"]]
y = df["HealthRisk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

st.sidebar.success(f"✅ Model trained successfully! Accuracy: {accuracy * 100:.2f}%")

# -------------------------
# User Input Section
# -------------------------
st.header("🧠 Enter Your Health Information")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Enter your Age", 10, 80, 25)
    bmi = st.number_input("Enter your BMI (Body Mass Index)", 10.0, 40.0, 22.5)
    activity = st.selectbox("Your Physical Activity Level", ["Low", "Moderate", "High"])
with col2:
    sleep = st.slider("Average Sleep (hours per night)", 3, 10, 7)
    diet = st.selectbox("Your Diet Type", ["Vegetarian", "Non-Vegetarian", "Mixed"])

# -------------------------
# Prediction Section
# -------------------------
if st.button("🔍 Predict My Health Risk"):
    input_data = pd.DataFrame({
        "Age": [age],
        "BMI": [bmi],
        "ActivityLevel": [le.transform([activity])[0]],
        "SleepHours": [sleep],
        "DietType": [le.transform([diet])[0]]
    })

    prediction = model.predict(input_data)[0]
    risk_label = le.inverse_transform([prediction])[0]

    st.subheader(f"🩺 Predicted Health Risk Level: **{risk_label}**")

    # -------------------------
    # Recommendations
    # -------------------------
    st.markdown("### 💡 Personalized Recommendations")

    if risk_label == "Low":
        st.success("✅ You are in good health! Maintain your balanced lifestyle.")
        st.write("""
        **Recommendations:**
        - Keep exercising 4–5 times per week 🏃‍♂️  
        - Eat balanced meals with fruits and vegetables 🥗  
        - Continue regular sleep pattern 😴  
        - Stay hydrated 💧  
        """)
    elif risk_label == "Medium":
        st.warning("⚠️ Moderate risk — Consider improving daily habits.")
        st.write("""
        **Recommendations:**
        - Exercise at least 30 mins daily 🏋️‍♀️  
        - Include more whole grains and proteins 🍎  
        - Reduce sugar and junk food ❌  
        - Improve sleep consistency 🕒  
        """)
    else:
        st.error("🚨 High health risk — You should consult a doctor soon.")
        st.write("""
        **Recommendations:**
        - Consult a healthcare provider 👩‍⚕️  
        - Start light exercise or walking 🚶‍♂️  
        - Focus on low-fat, high-fiber foods 🍲  
        - Maintain a fixed sleep schedule 💤  
        """)

# -------------------------
# Footer
# -------------------------
st.markdown("""
---
**Made for College AI Project — Supporting UN Sustainable Development Goal 3**  
*Ensure healthy lives and promote well-being for all at all ages.*
""")