import streamlit as st
from model import LayoutDetector
from main import save_uploaded_image, bgr_to_rgb
import pandas as pd

st.set_page_config(
    page_title="Document Layout Detection",
    layout="wide"
)

st.title("Document Layout Detection System.")
st.write("Upload a document image and detect layout regions.")

@st.cache_resource
def load_model():
    return LayoutDetector("models/best.pt")
# loading our model.
detector = load_model()

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(uploaded, use_column_width=True)

    image_path = save_uploaded_image(uploaded)

    with st.spinner("Running detection..."):
        annotated, results = detector.predict(image_path)

    with col2:
        st.subheader("Detected Layout")
        st.image(bgr_to_rgb(annotated), use_column_width=True)

    # Show detection details
    if results.boxes is not None:
        data = []
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = results.names[cls]
            data.append({
                "Class": name,
                "Confidence": round(conf, 3)
            })

        st.subheader("Detection Summary")
        st.dataframe(pd.DataFrame(data))


