import streamlit as st
from PIL import Image
import requests
import io
import torch # We need torch to use the model directly for the explanation

# --- Make sure Streamlit can find your other scripts ---
import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from pipeline.predict_and_explain import load_full_model_for_inference, get_transforms, generate_attention_map

# -----------------------------
# --- PAGE CONFIGURATION ---
# -----------------------------
st.set_page_config(
    page_title="AI X-Ray Analysis",
    page_icon="🤖",
    layout="wide" # Use the full page width
)

# -----------------------------
# --- MODEL LOADING (Cached) ---
# -----------------------------
# Cache the model so it doesn't reload on every interaction
@st.cache_resource
def load_model():
    chexpert_path = PROJECT_ROOT / "checkpoints/best_model_epoch9_auc0.7613.pth"
    lora_path = PROJECT_ROOT / "checkpoints_mura_lora/best_model_adapters/"
    model = load_full_model_for_inference(chexpert_path, lora_path)
    return model

model = load_model()
_, val_transform = get_transforms()


# -----------------------------
# --- UI LAYOUT ---
# -----------------------------
st.title("🤖 AI-Powered X-Ray Anomaly Detection")
st.sidebar.title("About")
st.sidebar.info(
    "This is a demonstration of a medical imaging foundation model. "
    "It was pre-trained on the CheXpert dataset and then adapted for the MURA "
    "musculoskeletal dataset using LoRA."
    "\n\n**Disclaimer:** This tool is for educational purposes only and is not a "
    "substitute for professional medical advice."
)

st.write("Upload a musculoskeletal X-ray to classify it as **Normal** or **Abnormal** and see where the AI is looking.")

uploaded_file = st.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Create two columns for a side-by-side layout
    col1, col2 = st.columns(2)
    
    image_pil = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.image(image_pil, caption="Uploaded X-Ray", use_container_width=True)

    with col2:
        with st.spinner("🧠 Analyzing..."):
            # --- API Call for Prediction ---
            image_bytes = io.BytesIO()
            image_pil.save(image_bytes, format='PNG')
            image_bytes.seek(0)
            api_url = "http://127.0.0.1:8000/predict" # Using the simple predict endpoint
            files = {'file': ('image.png', image_bytes, 'image/png')}
            
            try:
                response = requests.post(api_url, files=files)
                if response.status_code == 200:
                    results = response.json()
                    
                    # --- Generate Explanation ---
                    # We do this in the UI app now, which is simpler
                    image_tensor = val_transform(image_pil).unsqueeze(0)
                    attention_map_img = generate_attention_map(model, image_tensor.to("cuda" if torch.cuda.is_available() else "cpu"), image_pil)

                    # --- Display Results ---
                    st.subheader("📊 Analysis Results")
                    
                    st.metric(
                        label="Prediction",
                        value=results['predicted_class'],
                        delta=f"Confidence: {results['confidence']:.2%}",
                        delta_color=("normal" if results['predicted_class'] == "Normal (Negative)" else "inverse")
                    )

                    st.image(attention_map_img, caption="🧠 AI Attention Map", use_container_width=True)
                    
                    with st.expander("See Detailed Probabilities"):
                        st.write(results['probabilities'])

                else:
                    st.error(f"Error from API: {response.status_code} - {response.text}")
            
            except requests.exceptions.ConnectionError:
                st.error("Connection Error: Could not connect to the API. Is the `app.py` server running?")