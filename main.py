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

# Function to blur the image
def blur_image(image):
    return cv.GaussianBlur(image, (15, 15), cv.BORDER_DEFAULT)

# Function for edge detection using Canny
def canny_edges(image):
    return cv.Canny(image, 100, 200)

# Function to adjust brightness
def adjust_brightness(image, factor):
    pil_img = Image.fromarray(image)
    enhancer = ImageEnhance.Brightness(pil_img)
    return np.array(enhancer.enhance(factor))

# Function to adjust contrast
def adjust_contrast(image, factor):
    pil_img = Image.fromarray(image)
    enhancer = ImageEnhance.Contrast(pil_img)
    return np.array(enhancer.enhance(factor))

# Function to flip the image horizontally
def flip_image(image):
    return cv.flip(image, 1)

# Function to rotate the image by 90 degrees
def rotate_image(image):
    return cv.rotate(image, cv.ROTATE_90_CLOCKWISE)

# Function to convert image to sepia
def apply_sepia(image):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_img = cv.transform(image, sepia_filter)
    return np.clip(sepia_img, 0, 255).astype(np.uint8)

# Function to apply sharpening
def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv.filter2D(image, -1, kernel)

# Add more image processing functions as needed...

# Streamlit app
st.title("Image Processing with OpenCV and Streamlit")

# Uploading the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

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
        'Blur Image',
        'Edge Detection (Canny)',
        'Brightness Adjustment',
        'Contrast Adjustment',
        'Flip Image Horizontally',
        'Rotate Image 90°',
        'Apply Sepia Effect',
        'Sharpen Image',
        # Add more operations here...
    ])

    # Parameters for certain operations
    if operation == 'Brightness Adjustment':
        brightness_factor = st.slider('Brightness Factor', 0.5, 3.0, 1.0)
    if operation == 'Contrast Adjustment':
        contrast_factor = st.slider('Contrast Factor', 0.5, 3.0, 1.0)
    if operation == 'Rescale Image':
        scale = st.slider('Scale', 0.1, 2.0, 0.75)

    # Process the image based on selected operation
    if operation == 'Rescale Image':
        processed_image = rescaleFrame(image, scale)
    elif operation == 'Convert to Grayscale':
        processed_image = to_grayscale(image)
    elif operation == 'Blur Image':
        processed_image = blur_image(image)
    elif operation == 'Edge Detection (Canny)':
        processed_image = canny_edges(image)
    elif operation == 'Brightness Adjustment':
        processed_image = adjust_brightness(image, brightness_factor)
    elif operation == 'Contrast Adjustment':
        processed_image = adjust_contrast(image, contrast_factor)
    elif operation == 'Flip Image Horizontally':
        processed_image = flip_image(image)
    elif operation == 'Rotate Image 90°':
        processed_image = rotate_image(image)
    elif operation == 'Apply Sepia Effect':
        processed_image = apply_sepia(image)
    elif operation == 'Sharpen Image':
        processed_image = sharpen_image(image)
    else:
        processed_image = image

    # Display the processed image
    st.image(processed_image, caption=f'Processed Image ({operation})', use_column_width=True)
