import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
# Dictionary to map model names to their corresponding file paths
model_paths = {
    "Hand": "C:\\Users\\sahaj\\Downloads\\hand_model.h5",
    "Finger": "C:\\Users\sahaj\\Downloads\\finger_model.h5",
    "Elbow": "C:\\Users\\sahaj\\Downloads\\elbow_model.h5",
    "Shoulder": "C:\\Users\\sahaj\\Downloads\\shoulder_model.h5",
   
    "Forearm": "C:\\Users\\sahaj\\Downloads\\forearm_model.h5"
    
}

# Define a function to load the selected model
def load_model(model_name):
    model_path = model_paths.get(model_name)
    if model_path:
        return tf.keras.models.load_model(model_path)
    else:
        return None

def predict_fracture(model, image):
    img = np.array(image)
    img_pil = Image.fromarray(img)
    img_gray = img_pil.convert('L')
    img_gray = img_gray.resize((224, 224))
    img_gray = np.array(img_gray)
    img_gray = img_gray / 255.0  # Normalize
    img_gray = np.expand_dims(img_gray, axis=0)  # Add batch dimension
    img_gray = np.expand_dims(img_gray, axis=-1)  # Add channel dimension
    prediction = model.predict(img_gray)
    return prediction



# Streamlit app
def main():
    st.title("Fracture Detection from X-ray Images")
    st.write("Select the body part and upload an X-ray image to detect fracture")

    # Dropdown to select the body part
    selected_part = st.selectbox("Select Body Part:", list(model_paths.keys()))

    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Load the selected model
        model = load_model(selected_part)

        if model is not None:
            # Make prediction on the uploaded image
            prediction = predict_fracture(model, image)
            st.write(prediction)
            predict_max = np.argmax(prediction)
            st.write(predict_max)
            if predict_max >= 0.5:
                st.write("Prediction: Fracture Detected with Probability:", predict_max)
            else:
                st.write("Prediction: No Fracture Detected with Probability:", 1 - predict_max)
        else:
            st.write("Error: Model not found for the selected body part")

if __name__ == "__main__":





























































    





    main()
