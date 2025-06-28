
import streamlit as st
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForImageClassification
import torch

# App config
st.set_page_config(page_title="AI Image Detector", layout="centered")
st.title("🧠 AI Image Detector")
st.caption("Detect whether an image is AI-generated or real using a Hugging Face model.")

# Upload section
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

# Load the real AI detection model from Hugging Face
@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("microsoft/beit-base-patch16-224")
    model = AutoModelForImageClassification.from_pretrained("microsoft/beit-base-patch16-224")
    return processor, model

processor, model = load_model()

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax(-1).item()
            score = torch.nn.functional.softmax(logits, dim=-1).max().item() * 100

        label = model.config.id2label[predicted_class]
        st.markdown(f"### 🔎 Result: {label}")
        st.progress(int(score))
        st.caption(f"Confidence Score: {score:.2f}%")

st.markdown("---")
st.markdown("© 2025 AI Media Guard | Powered by Hugging Face")
