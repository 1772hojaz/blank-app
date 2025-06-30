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

# Initialize DeepForest model
@st.cache_resource
def load_model():
    model = main.deepforest()
    model.use_release()
    return model

def process_image(uploaded_file, model):
    """Process uploaded image and return predictions"""
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    
    try:
        # Handle TIFF separately
        if uploaded_file.name.lower().endswith(('.tif', '.tiff')):
            with rasterio.open(tmp_path) as src:
                img_array = src.read()
                img_array = reshape_as_image(img_array)
                
                # Handle different band counts
                if img_array.shape[2] > 3:
                    img_array = img_array[:, :, :3]  # Use first 3 bands
                elif img_array.shape[2] == 1:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                
                img_array = img_array.astype("uint8")
        else:
            # Handle other image formats
            img_array = np.array(Image.open(tmp_path))
            if len(img_array.shape) == 2:  # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    finally:
        os.unlink(tmp_path)
    
    # Convert to BGR for DeepForest compatibility
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Run prediction
    boxes = model.predict_image(image=img_array)
    
    return boxes, img_array

def main():
    st.title("🌳 Tree Counting with DeepForest")
    st.write("Upload an image to count trees (supports TIFF, PNG, JPG)")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image", 
        type=["tif", "tiff", "png", "jpg", "jpeg"]
    )
    
    if uploaded_file is not None:
        model = load_model()
        
        with st.spinner("Processing image..."):
            boxes, img_array = process_image(uploaded_file, model)
            
        # Count trees
        tree_count = len(boxes)
        st.success(f"**Detected Trees: {tree_count}**")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        plot_predictions(img_array, boxes, ax=ax)
        ax.set_axis_off()
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_array, 
                    caption="Original Image", 
                    channels="BGR",
                    use_column_width=True)
        
        with col2:
            st.pyplot(fig, use_container_width=True)
        
        # Show prediction data
        st.subheader("Detection Details")
        st.dataframe(boxes)

if __name__ == "__main__":
    main()
