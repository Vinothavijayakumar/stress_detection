import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# UCI Dataset Placeholder (replace with real one if available)
# Example link: https://archive.ics.uci.edu/ml/datasets/Bio+Stress+Data

def create_fake_physiological_data(samples=500):
    np.random.seed(42)
    X = np.random.rand(samples, 6)  # Features: heart rate, temp, etc.
    y = np.random.randint(0, 3, size=samples)  # 0=Low, 1=Medium, 2=High
    return X, y

# Data
X, y = create_fake_physiological_data()
y_cat = to_categorical(y, num_classes=3)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_cat, test_size=0.2, stratify=y)

# CNN Model
model = Sequential([
    Conv1D(32, 2, activation='relu', input_shape=(X.shape[1], 1)),
    MaxPooling1D(2),
    Dropout(0.3),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

# Plot Validation Accuracy (for internal use)
def show_accuracy_chart():
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(val_accuracy) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(epochs, val_accuracy, color='skyblue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracy per Epoch')
    ax.set_ylim(0, 1)
    st.pyplot(fig)

# Streamlit Interface
st.title("🧠 Stress Level Predictor using CNN")
st.markdown("Built with Keras + Streamlit | Dataset: Simulated (UCI Placeholder)")

st.sidebar.header("Enter Physiological Inputs")

input_features = []
feature_names = ['Heart Rate', 'Temperature', 'EDA', 'Respiration', 'O2 Level', 'Blood Pressure']
for name in feature_names:
    val = st.sidebar.slider(f"{name}", 0.0, 1.0, 0.5)
    input_features.append(val)

if st.sidebar.button("Predict Stress Level"):
    input_array = np.array(input_features).reshape(1, -1)
    input_scaled = scaler.transform(input_array).reshape(1, 6, 1)
    prediction = model.predict(input_scaled)[0]
    predicted_class = np.argmax(prediction)
    class_labels = ['Low Stress', 'Medium Stress', 'High Stress']
    st.success(f"🧾 Predicted Stress Level: **{class_labels[predicted_class]}**")

    st.subheader("Prediction Probabilities")
    st.bar_chart(prediction)

st.subheader("📊 Model Validation Accuracy")
show_accuracy_chart()

# Final Evaluation
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
st.write(f"✅ Final Test Accuracy: **{accuracy:.2f}**")  
