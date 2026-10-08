import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# Page settings
# =========================

st.set_page_config(
    page_title="Fruit & Vegetable Classifier",
    page_icon="🍎"
)

st.title("🍎 Fruit & Vegetable Classifier")
st.write("Upload an image and the AI model will predict the fruit or vegetable.")

# =========================
# Model
# =========================

model_path = r"D:\desktop\jupyter notebook\fruit_classifier_mobilenet.keras"

model = tf.keras.models.load_model(model_path)

# =========================
# Class names
# =========================

class_names = [
    'apple',
    'banana',
    'beetroot',
    'bell pepper',
    'cabbage',
    'capsicum',
    'carrot',
    'cauliflower',
    'chilli pepper',
    'corn',
    'cucumber',
    'eggplant',
    'garlic',
    'ginger',
    'grapes',
    'jalepeno',
    'kiwi',
    'lemon',
    'lettuce',
    'mango',
    'onion',
    'orange',
    'paprika',
    'pear',
    'peas',
    'pineapple',
    'pomegranate',
    'potato',
    'raddish',
    'soy beans',
    'spinach',
    'sweetcorn',
    'sweetpotato',
    'tomato',
    'turnip',
    'watermelon'
]

# =========================
# Upload image
# =========================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize
    img = image.resize((160, 160))

    # Convert to array
    img_array = np.array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # =========================
    # Prediction
    # =========================

    prediction = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction[0])

    predicted_class = class_names[predicted_index]

    confidence = prediction[0][predicted_index] * 100

    # =========================
    # Result
    # =========================

    st.success(f"Prediction: {predicted_class}")

    st.info(f"Confidence: {confidence:.2f}%")

    # Show top 3 predictions
    st.subheader("Top 3 Predictions")

    top_indices = np.argsort(prediction[0])[-3:][::-1]

    for index in top_indices:
        st.write(
            f"**{class_names[index]}:** "
            f"{prediction[0][index] * 100:.2f}%"
        )