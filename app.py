import os
import tempfile
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# Load the YOLO model
MODEL_PATH = "best_model.pt"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found! Please make sure 'best_model.pt' is in the project directory.")
else:
    model = YOLO(MODEL_PATH)

# Function to process image and draw bounding boxes
def predict_and_draw_boxes(image: Image.Image):
    # Convert PIL Image to OpenCV format
    open_cv_image = np.array(image)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

    # Save image to a temporary file for YOLO processing
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name, format="JPEG")
        results = model.predict(source=tmp.name, verbose=False)

    # Process YOLO results and draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = float(box.conf[0])  # Confidence score
            label = result.names[int(box.cls[0])]  # Class label

            # Draw bounding box
            cv2.rectangle(open_cv_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Add label and confidence score
            text = f"{label} {conf:.2f}"
            cv2.putText(open_cv_image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Convert OpenCV image back to PIL format
    result_pil = Image.fromarray(cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB))
    return result_pil

# Streamlit UI
st.title("YOLO Object Detection")
st.write("Upload an image to detect defects:")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict"):
        with st.spinner("Running inference..."):
            result_image = predict_and_draw_boxes(image)
        
        st.image(result_image, caption="Detected Defects", use_column_width=True)