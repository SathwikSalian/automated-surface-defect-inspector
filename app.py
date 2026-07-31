import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os

st.set_page_config(page_title="Quality Control Inspector", layout="centered")
st.title("Automated Surface Defect Inspector")
st.write("Upload a surface image to detect, localize, and classify defect regions.")

@st.cache_resource
def load_custom_model():
    model_path = os.path.join("runs", "detect", "defect_model", "weights", "best.pt")
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        st.error("Could not find trained weights at 'runs/detect/defect_model/weights/best.pt'")
        return None

model = load_custom_model()

uploaded_file = st.file_uploader("Upload Product Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Surface", use_container_width=True)
    
    if st.button("Inspect Product"):
        with st.spinner("Analyzing image..."):
            results = model(image)
            annotated_frame = results[0].plot()
            
            st.subheader("Inspection Result:")
            st.image(annotated_frame, caption="Detected Defects", use_container_width=True)
            
            boxes = results[0].boxes
            if len(boxes) > 0:
                st.error(f"❌ DEFECT DETECTED: Found {len(boxes)} anomaly region(s)!")
                
                # Show detected defect classes and confidence scores
                st.write("### Detected Anomalies:")
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0]) * 100
                    st.write(f"- **{class_name.capitalize()}** (Confidence: {confidence:.1f}%)")
            else:
                st.success("✅ PASSED: No defects detected on this product.")