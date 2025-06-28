
import os
import streamlit as st
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# App config
st.set_page_config(page_title="AI Image Detector", layout="centered")
st.title("🧠 AI Image Detector")
st.caption("Detect whether an image is AI-generated or real using a fine-tuned CLIP model.")

# Upload section
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

# Load the fake-vs-real detection model from Hugging Face
@st.cache_resource
def load_model():
    hf_token = st.secrets["HF_TOKEN"]

processor = CLIPProcessor.from_pretrained(
    "nateraw/clip-vit-base-patch32-finetuned-fake-vs-real",
    use_auth_token=hf_token
)
model = CLIPModel.from_pretrained(
    "nateraw/clip-vit-base-patch32-finetuned-fake-vs-real",
    use_auth_token=hf_token
)
    return processor, model

processor, model = load_model()
labels = ["real", "fake"]

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).squeeze()

        predicted_idx = torch.argmax(probs).item()
        label = labels[predicted_idx]
        confidence = probs[predicted_idx].item() * 100

        emoji = "🟢" if label == "real" else "🔴"
        st.markdown(f"### {emoji} Result: {label.upper()}")
        st.progress(int(confidence))
        st.caption(f"Confidence Score: {confidence:.2f}%")

st.markdown("---")
st.markdown("© 2025 AI Media Guard | Powered by Hugging Face")
