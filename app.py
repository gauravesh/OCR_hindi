import streamlit as st
from PIL import Image
import pytesseract

# Function to process uploaded image
def process_image(image):
    # Add your OCR processing logic here
    processed_text = pytesseract.image_to_string(image,lang='hin')
    #processed_text = "This is a placeholder for processed text."
    return processed_text

# Streamlit app layout
st.title("📃OCR (Optical Character Recognition) App")
st.write("Upload an image for OCR processing.")

# File upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Display processed text
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process the image and get text
    processed_text = process_image(image)

    # Display processed text
    st.header("Processed Text:")
    st.text(processed_text)