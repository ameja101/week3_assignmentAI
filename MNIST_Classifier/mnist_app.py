# === Step : Streamlit App for Deployment ===
"""
To deploy the model with Streamlit, create a new file named 'mnist_app.py' and add this code:
----------------------------------------------------------

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🧠 MNIST Digit Classifier")
st.write("Upload a 28x28 grayscale iSmage of a handwritten digit.")

# Load trained model
model = tf.keras.models.load_model('mnist_cnn_model')

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L').resize((28, 28))
    img_array = np.array(image).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    pred = model.predict(img_array)
    pred_label = np.argmax(pred, axis=1)[0]

    st.image(image, caption=f"Predicted Digit: {pred_label}", use_container_width=True)
    st.success(f"The model predicts this digit as: {pred_label}")

-------------------------------------fi---------------------
Run the Streamlit app:
    streamlit run mnist_app.py
"""
