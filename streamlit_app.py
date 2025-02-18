import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained binary classification model
model = load_model('fruit_classifier_binary.h5')

st.title('Fruit Fresh or Rotten Classification')

# Upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Load and preprocess the image
    img = image.load_img(uploaded_image, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    # Predict
    prediction = model.predict(img_array)[0][0]  # Get the scalar output

    # Display the result
    if prediction > 0.5:
        label = "Rotten Fruit 🍂"
    else:
        label = "Fresh Fruit 🍏"

    st.image(uploaded_image, caption=f"Prediction: {label}", use_container_width=True)
    st.write(f"Prediction: **{label}**")