import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from pathlib import Path
import zipfile

st.set_page_config(page_title="Change Detection Algorithm", layout="wide")

st.title("🔍 Change Detection Algorithm")
st.markdown("**Detect and highlight differences between 'before' and 'after' images**")

def detect_changes(before_img, after_img, threshold=30, min_area=500):
    """Detect changes between before and after images."""
    # Convert to grayscale
    gray_before = cv2.cvtColor(before_img, cv2.COLOR_RGB2GRAY)
    gray_after = cv2.cvtColor(after_img, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur
    blur_before = cv2.GaussianBlur(gray_before, (5, 5), 0)
    blur_after = cv2.GaussianBlur(gray_after, (5, 5), 0)
    
    # Compute absolute difference
    diff = cv2.absdiff(blur_before, blur_after)
    
    # Apply threshold
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=3)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area
    significant_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    
    return significant_contours, diff, thresh

def draw_detections(image, contours, draw_bbox=True, draw_polygon=True, draw_segment=True):
    """Draw detections on image."""
    result = image.copy()
    
    for contour in contours:
        if draw_bbox:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(result, "CHANGE", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        if draw_polygon:
            cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
        
        if draw_segment:
            overlay = result.copy()
            cv2.drawContours(overlay, [contour], -1, (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
    
    return result

# Sidebar settings
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Detection Threshold", 1, 100, 30, 
                              help="Lower = more sensitive")
min_area = st.sidebar.slider("Minimum Area", 100, 5000, 500,
                             help="Minimum pixel area to detect")

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Visualization")
draw_bbox = st.sidebar.checkbox("Bounding Box", value=True)
draw_polygon = st.sidebar.checkbox("Polygon Contour", value=True)
draw_segment = st.sidebar.checkbox("Filled Segment", value=True)

# Mode selection
mode = st.radio("Select Mode:", ["Upload Single Pair", "Upload Folder (ZIP)"], horizontal=True)

if mode == "Upload Single Pair":
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Before Image (X.jpg)")
        before_file = st.file_uploader("Upload before image", type=['jpg', 'jpeg', 'png'], key="before")
    
    with col2:
        st.subheader("📤 After Image (X~2.jpg)")
        after_file = st.file_uploader("Upload after image", type=['jpg', 'jpeg', 'png'], key="after")
    
    if before_file and after_file:
        # Load images
        before_img = np.array(Image.open(before_file).convert('RGB'))
        after_img = np.array(Image.open(after_file).convert('RGB'))
        
        # Resize if needed
        if before_img.shape != after_img.shape:
            after_img = cv2.resize(after_img, (before_img.shape[1], before_img.shape[0]))
        
        # Detect changes
        contours, diff, thresh = detect_changes(before_img, after_img, threshold, min_area)
        
        # Draw results
        result = draw_detections(after_img, contours, draw_bbox, draw_polygon, draw_segment)
        
        # Display results
        st.markdown("---")
        st.success(f"✅ Detected **{len(contours)}** changes")
        
        # Show images
        col1, col2 = st.columns(2)
        with col1:
            st.image(before_img, caption="BEFORE", use_container_width=True)
        with col2:
            st.image(after_img, caption="AFTER", use_container_width=True)
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.image(diff, caption="DIFFERENCE MAP", use_container_width=True)
        with col2:
            st.image(result, caption="DETECTED CHANGES", use_container_width=True)
        
        # Download result
        st.markdown("---")
        buf = io.BytesIO()
        Image.fromarray(result).save(buf, format='PNG')
        buf.seek(0)
        st.download_button("📥 Download Result", buf, "change_detection_result.png", "image/png")

else:
    st.markdown("---")
    st.subheader("📁 Upload ZIP Folder")
    st.markdown("""
    **Expected structure:**
    ```
    folder.zip
    ├── image1.jpg      (before)
    ├── image1~2.jpg    (after)
    ├── image2.jpg      (before)
    ├── image2~2.jpg    (after)
    └── ...
    ```
    """)
    
    zip_file = st.file_uploader("Upload ZIP file", type=['zip'])
    
    if zip_file:
        with zipfile.ZipFile(zip_file, 'r') as z:
            file_list = z.namelist()
            
            # Find pairs
            pairs = []
            for f in file_list:
                if f.endswith(('.jpg', '.jpeg', '.png')) and '~2' not in f:
                    name = Path(f).stem
                    ext = Path(f).suffix
                    after_name = f"{name}~2{ext}"
                    
                    if after_name in file_list or any(after_name in x for x in file_list):
                        pairs.append({'before': f, 'after': after_name, 'name': name})
            
            st.info(f"Found **{len(pairs)}** image pairs")
            
            if pairs and st.button("🚀 Process All Pairs"):
                results_zip = io.BytesIO()
                
                with zipfile.ZipFile(results_zip, 'w') as out_zip:
                    progress = st.progress(0)
                    
                    for i, pair in enumerate(pairs):
                        # Load images
                        before_data = z.read(pair['before'])
                        after_data = z.read(pair['after'])
                        
                        before_img = np.array(Image.open(io.BytesIO(before_data)).convert('RGB'))
                        after_img = np.array(Image.open(io.BytesIO(after_data)).convert('RGB'))
                        
                        if before_img.shape != after_img.shape:
                            after_img = cv2.resize(after_img, (before_img.shape[1], before_img.shape[0]))
                        
                        # Detect and draw
                        contours, _, _ = detect_changes(before_img, after_img, threshold, min_area)
                        result = draw_detections(after_img, contours, draw_bbox, draw_polygon, draw_segment)
                        
                        # Save to zip
                        buf = io.BytesIO()
                        Image.fromarray(result).save(buf, format='PNG')
                        out_zip.writestr(f"{pair['name']}_detected.png", buf.getvalue())
                        
                        progress.progress((i + 1) / len(pairs))
                        st.write(f"✅ {pair['name']}: {len(contours)} changes")
                    
                results_zip.seek(0)
                st.download_button("📥 Download All Results (ZIP)", results_zip, "change_detection_results.zip", "application/zip")

# Instructions
st.markdown("---")
st.markdown("""
### 📖 How It Works:
1. **Upload** before image (X.jpg) and after image (X~2.jpg)
2. Algorithm **computes pixel differences** between images
3. **Detects contours** around changed/missing areas
4. **Draws bounding boxes, polygons, or segments** around changes

### 🎯 Output Types:
- **Bounding Box**: Red rectangle around changed area
- **Polygon**: Green contour following exact shape
- **Segment**: Blue filled area highlighting change
""")
