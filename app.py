import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from googletrans import Translator
import time

# Load your model from the JSON file
def load_model():
    try:
        # Load model architecture from JSON file
        with open('model-bw.json', 'r') as json_file:
            model_json = json_file.read()
        
        # Deserialize model from JSON
        model = tf.keras.models.model_from_json(model_json)
        
        # Load model weights
        model.load_weights('model-bw.h5')
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

minValue = 60
target_size = (256, 256)

# Preprocess the image and visualize ROI
def preprocess_image(frame):
    # Crop the frame to the desired ROI size (256x256)
    roi = frame[100:356, 100:356]  # Adjust the coordinates as needed

    # Convert ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 2)

    # Apply adaptive thresholding
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Apply global thresholding
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Resize the image to target size
    resized_image = cv2.resize(res, target_size)

    # Draw a rectangle on the original frame to visualize the ROI
    frame_with_roi = frame.copy()
    cv2.rectangle(frame_with_roi, (100, 100), (356, 356), (0, 255, 0), 2)  # Adjust the coordinates as needed

    return resized_image, frame_with_roi

# Make predictions using the model
def predict_model(image):
    model = load_model()
    prediction = model.predict(image)
    return prediction

# Function to map prediction indices to characters
def map_to_char(prediction):
    if prediction >= 0 and prediction <= 25:
        return chr(prediction + ord('a'))
    else:
        return "Unknown"


# Translate text to Indian languages
def translate_to_indian(text, dest):
    if text is None:
        return ""
    translator = Translator()
    result = translator.translate(text=text, dest=dest)
    if result is not None:
        return result.text
    else:
        return "Translation failed"

def main():
    st.title("Indian Sign Language Image Classification")

    # Language selection dropdown
    selected_language = st.selectbox("Select Language", ["Hindi", "Marathi", "Gujarati"])

    # Start the webcam feed
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    word = ""

    # Placeholder for displaying live feed
    feed_placeholder = st.empty()

    # Placeholder for displaying the accumulated word
    word_placeholder = st.empty()

    translation_interval = 40  # Translate every 30 seconds
    last_translation_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to retrieve frame from webcam.")
            break

        # Preprocess the frame and get the processed image and frame with ROI
        processed_image, frame_with_roi = preprocess_image(frame)

        # Display the frame with ROI in the app
        feed_placeholder.image(frame_with_roi, channels="BGR", use_column_width=True)

        if time.time() - start_time >= 8:  # Capture frame every 5 seconds
            input_image = np.expand_dims(processed_image, axis=0)
            prediction = predict_model(input_image)
            predicted_label = map_to_char(np.argmax(prediction))
            word += predicted_label
            start_time = time.time()  # Reset the timer
        
        # Update the accumulated word
        word_placeholder.text("Accumulated Word: " + word)

        # Translate the accumulated word every translation_interval seconds
        if time.time() - last_translation_time >= translation_interval:
            translation = translate_to_indian(word, selected_language)
            st.write("Translated Text:", translation)
            last_translation_time = time.time()
            word = ""  # Reset the accumulated word

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
