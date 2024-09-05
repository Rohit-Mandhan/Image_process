import streamlit as st
import cv2 as cv
from PIL import Image, ImageEnhance
import numpy as np

# Function to rescale the image
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Function to convert image to grayscale
def to_grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Function to invert colors
def invert_colors(image):
    return cv.bitwise_not(image)

# Function to apply histogram equalization
def histogram_equalization(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return cv.equalizeHist(gray)

# Function to apply a pencil sketch effect
def pencil_sketch(image):
    gray, sketch = cv.pencilSketch(image, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    return sketch

# Function to apply cartoon effect
def cartoon_effect(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    edges = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 10)
    color = cv.bilateralFilter(image, 9, 300, 300)
    cartoon = cv.bitwise_and(color, color, mask=edges)
    return cartoon

# Function to apply an emboss effect
def emboss_effect(image):
    kernel = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
    return cv.filter2D(image, -1, kernel)

# Function to apply dilation
def dilation(image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv.dilate(image, kernel, iterations=1)

# Function to apply erosion
def erosion(image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv.erode(image, kernel, iterations=1)

# Function to apply bilateral filtering
def bilateral_filter(image):
    return cv.bilateralFilter(image, 9, 75, 75)

# Function to apply CLAHE
def apply_clahe(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

# Function to apply thresholding
def thresholding(image, thresh_value=128):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, thresh_value, 255, cv.THRESH_BINARY)
    return thresh

# Streamlit app
st.title("Advanced Image Processing with OpenCV and Streamlit")

# Uploading the image
uploaded_file = st.file_uploader("Upload any image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    image = Image.open(uploaded_file)
    image = np.array(image)  # Convert to numpy array for OpenCV compatibility

    # Display the original image
    st.image(image, caption='Original Image', use_column_width=True)

    # Dropdown for selecting image processing operation
    operation = st.selectbox("Choose an image processing operation", [
        'Rescale Image',
        'Convert to Grayscale',
        'Invert Colors',
        'Histogram Equalization',
        'Pencil Sketch',
        'Cartoon Effect',
        'Emboss Effect',
        'Edge Detection (Canny)',
        'Dilation',
        'Erosion',
        'Bilateral Filtering',
        'CLAHE (Adaptive Histogram Equalization)',
        'Thresholding'
    ])

    # Additional input sliders for certain operations
    if operation == 'Rescale Image':
        scale = st.slider('Scale', 0.1, 2.0, 0.75)
    elif operation == 'Dilation' or operation == 'Erosion':
        kernel_size = st.slider('Kernel Size', 1, 30, 5)
    elif operation == 'Thresholding':
        thresh_value = st.slider('Threshold Value', 0, 255, 128)

    # Process the image based on selected operation
    if operation == 'Rescale Image':
        processed_image = rescaleFrame(image, scale)
    elif operation == 'Convert to Grayscale':
        processed_image = to_grayscale(image)
    elif operation == 'Invert Colors':
        processed_image = invert_colors(image)
    elif operation == 'Histogram Equalization':
        processed_image = histogram_equalization(image)
    elif operation == 'Pencil Sketch':
        processed_image = pencil_sketch(image)
    elif operation == 'Cartoon Effect':
        processed_image = cartoon_effect(image)
    elif operation == 'Emboss Effect':
        processed_image = emboss_effect(image)
    elif operation == 'Edge Detection (Canny)':
        processed_image = cv.Canny(image, 100, 200)
    elif operation == 'Dilation':
        processed_image = dilation(image, kernel_size)
    elif operation == 'Erosion':
        processed_image = erosion(image, kernel_size)
    elif operation == 'Bilateral Filtering':
        processed_image = bilateral_filter(image)
    elif operation == 'CLAHE (Adaptive Histogram Equalization)':
        processed_image = apply_clahe(image)
    elif operation == 'Thresholding':
        processed_image = thresholding(image, thresh_value)
    else:
        processed_image = image

    # Display the processed image
    st.image(processed_image, caption=f'Processed Image ({operation})', use_column_width=True)
