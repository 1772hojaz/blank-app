# app.py
import streamlit as st
import numpy as np
from PIL import Image
import rasterio
from rasterio.plot import reshape_as_image
import cv2
import matplotlib.pyplot as plt
from deepforest import main
from deepforest.utilities import plot_predictions
import tempfile
import os

# Set Streamlit configuration
st.set_page_config(page_title="Tree Detection with DeepForest", layout="wide")

# Load DeepForest model and cache it
@st.cache_resource
def load_model():
    model = main.deepforest()
    model.use_release()
    return model

# Process the uploaded image and return predictions
def process_image(uploaded_file, model):
    # Save uploaded file to a temporary path
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        if uploaded_file.name.lower().endswith(('.tif', '.tiff')):
            with rasterio.open(tmp_path) as src:
                img_array = src.read()
                img_array = reshape_as_image(img_array)

                if img_array.shape[2] > 3:
                    img_array = img_array[:, :, :3]
                elif img_array.shape[2] == 1:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

                img_array = img_array.astype("uint8")
        else:
            img_array = np.array(Image.open(tmp_path))
            if len(img_array.shape) == 2:  # grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    finally:
        os.unlink(tmp_path)

    # Convert to BGR as expected by DeepForest
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Run prediction
    boxes = model.predict_image(image=img_array)

    return boxes, img_array

# Main application
def main():
    # Sidebar info
    st.sidebar.title("🌲 Tree Detector")
    st.sidebar.markdown("""
    Upload aerial or satellite images to detect and count trees using DeepForest.

    **Supported formats:** TIFF, PNG, JPG, JPEG
    **Max size:** 10MB
    """)

    # Main content
    st.title("🌳 Tree Counting with DeepForest")
    st.write("Upload a geospatial image to detect trees.")

    uploaded_file = st.file_uploader("📤 Upload Image", type=["tif", "tiff", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("🚫 File too large. Please upload an image smaller than 10MB.")
            return

        model = load_model()

        with st.spinner("🧠 Processing image..."):
            boxes, img_array = process_image(uploaded_file, model)

        # Tree count
        tree_count = len(boxes)
        st.success(f"✅ **Detected Trees: {tree_count}**")

        # Image columns
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_array, caption="Original Image", channels="BGR", use_column_width=True)

        with col2:
            fig, ax = plt.subplots(figsize=(10, 10))
            plot_predictions(img_array, boxes, ax=ax)
            ax.set_axis_off()
            st.pyplot(fig, use_container_width=True)

        # Download results
        csv = boxes.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Detection Results", csv, file_name="tree_detections.csv", mime="text/csv")

        # Show prediction data
        with st.expander("📋 Detection Details"):
            st.dataframe(boxes)

if __name__ == "__main__":
    main()
