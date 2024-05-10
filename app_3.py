import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np

# Function for automatic thresholding
def automatic_threshold(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #_, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray_image

# Function for noise removal
def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image

# Function for deskewing
def getSkewAngle(cvImage) -> float:
    gray = cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

def rotateImage(cvImage, angle: float):
    (h, w) = cvImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(cvImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# Function to preprocess and process the image using OCR
def process_image(image, lang):
    image = np.array(image)
    thresholded = automatic_threshold(image)
    denoised = noise_removal(thresholded)
    #angle = getSkewAngle(denoised)
    #deskewed = rotateImage(denoised, angle)
    processed_text = pytesseract.image_to_string(denoised, lang=lang)
    return processed_text

# Streamlit app layout
st.title("📃OCR (Optical Character Recognition) App")
st.write("Upload an image for OCR processing.")

# Radio button for language selection
lang_selection = st.radio("Select OCR Language", options=["English", "Hindi"])

# Map language selection to Tesseract language codes
lang_map = {"English": "eng", "Hindi": "hin"}
selected_lang = lang_map[lang_selection]

# File upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Display processed text
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process the image and get text
    processed_text = process_image(image, selected_lang)

    # Display processed text
    st.header("Processed Text:")
    st.text(processed_text)
