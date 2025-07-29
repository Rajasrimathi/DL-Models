
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import streamlit as st

st.title("☕ Coffee Buying Prediction using Decision Tree (ID3 Algorithm)")

# Dataset
data = {
    'Weather': ['Sunny', 'Rainy', 'Overcast', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Rainy', 'Sunny', 'Rainy'],
    'TimeOfDay': ['Morning', 'Morning', 'Afternoon', 'Afternoon', 'Evening', 'Morning', 'Morning', 'Afternoon', 'Evening', 'Morning'],
    'SleepQuality': ['Poor', 'Good', 'Poor', 'Good', 'Poor', 'Good', 'Poor', 'Good', 'Good', 'Poor'],
    'Mood': ['Tired', 'Fresh', 'Tired', 'Energetic', 'Tired', 'Fresh', 'Tired', 'Tired', 'Energetic', 'Tired'],
    'BuyCoffee': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

# Encoding categorical data
encoders = {}
for col in df.columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoders[col] = encoder

X = df[['Weather', 'TimeOfDay', 'SleepQuality', 'Mood']]
y = df['BuyCoffee']

# Train model
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

# Show decision tree
st.subheader("Decision Tree Visualization")
fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(model, feature_names=X.columns, class_names=encoders['BuyCoffee'].classes_, filled=True)
st.pyplot(fig)

# Input for prediction
st.subheader("Make a Prediction")
weather = st.selectbox("Weather", encoders['Weather'].classes_)
timeofday = st.selectbox("Time of Day", encoders['TimeOfDay'].classes_)
sleep = st.selectbox("Sleep Quality", encoders['SleepQuality'].classes_)
mood = st.selectbox("Mood", encoders['Mood'].classes_)

if st.button("Predict"):
    input_data = {
        'Weather': encoders['Weather'].transform([weather])[0],
        'TimeOfDay': encoders['TimeOfDay'].transform([timeofday])[0],
        'SleepQuality': encoders['SleepQuality'].transform([sleep])[0],
        'Mood': encoders['Mood'].transform([mood])[0]
    }
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    result = encoders['BuyCoffee'].inverse_transform([prediction])[0]
    st.success(f"☕ Prediction: BuyCoffee = {result}")

# Show rules
st.subheader("Decision Tree Rules")
rules = export_text(model, feature_names=list(X.columns))
st.text(rules)