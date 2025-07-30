import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os

# Import the XFeat model from the modules folder
from modules.xfeat import XFeat

# Use Streamlit's caching to load the model only once
@st.cache_resource
def load_model():
    """Loads the XFeat model and caches it."""
    model = XFeat()
    print("XFeat model loaded.")
    return model

# Re-use the visualization function from your previous script
def plot_matches(img0, img1, kps0, kps1, matches, color=(0,255,0), radius=3):
    if isinstance(img0, np.ndarray):
        img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
    if isinstance(img1, np.ndarray):
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    canvas = np.zeros((max(h0,h1), w0+w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0:] = img1

    if kps0 is not None:
        for p in kps0:
            cv2.circle(canvas, tuple(p.astype(int)), radius, color, -1)
    if kps1 is not None:
        for p in kps1:
            cv2.circle(canvas, (p[0].astype(int) + w0, p[1].astype(int)), radius, color, -1)

    if matches is not None:
        for m in matches:
            idx0, idx1 = m
            pt0 = kps0[idx0].astype(int)
            pt1 = kps1[idx1].astype(int)
            cv2.line(canvas, tuple(pt0), (pt1[0]+w0, pt1[1]), color, 1)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB) # Return RGB for Streamlit

# --- Streamlit App ---

st.set_page_config(layout="wide")
st.title("XFeat: Accelerated Feature Matcher 🚀")

# Load the model
model = load_model()

# Create two columns for file uploaders
col1, col2 = st.columns(2)

with col1:
    st.header("Image 1")
    uploaded_file1 = st.file_uploader("Upload the first image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file1 is not None:
        image1 = Image.open(uploaded_file1)
        st.image(image1, caption='Uploaded Image 1', use_column_width=True)

with col2:
    st.header("Image 2")
    uploaded_file2 = st.file_uploader("Upload the second image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file2 is not None:
        image2 = Image.open(uploaded_file2)
        st.image(image2, caption='Uploaded Image 2', use_column_width=True)

# "Match Features" button
if uploaded_file1 and uploaded_file2:
    if st.button("Match Features", use_container_width=True):
        # Convert PIL images to NumPy arrays for OpenCV/XFeat
        img1_np = np.array(image1)
        img2_np = np.array(image2)

        # Show a spinner while processing
        with st.spinner('Finding matches... this may take a moment.'):
            # The model's match function takes numpy arrays in BGR format
            img1_bgr = cv2.cvtColor(img1_np, cv2.COLOR_RGB2BGR)
            img2_bgr = cv2.cvtColor(img2_np, cv2.COLOR_RGB2BGR)
            
            mkpts0, mkpts1 = model.match_xfeat(img1_bgr, img2_bgr)
            
            st.success(f"Found {len(mkpts0)} matches!")
            
            # Create match indices for visualization
            matches_indices = np.arange(len(mkpts0)).reshape(-1,1)
            matches_indices = np.hstack((matches_indices, matches_indices))
            
            # Plot the matches
            output_image = plot_matches(img1_np, img2_np, mkpts0, mkpts1, matches_indices)
            
            st.image(output_image, caption='Matching Result', use_column_width=True)