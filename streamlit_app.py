import streamlit as st
from PIL import Image
import torch
import sys
from pathlib import Path

# --- ROBUST PATHING ---
# This makes sure the script knows where it is and can find all other files
SCRIPT_DIR = Path(__file__).resolve().parent
# We add the 'pipeline' folder to the path so imports work
sys.path.append(str(SCRIPT_DIR / "pipeline"))

# --- Import your custom functions ---
# We can now be sure this import will work
from predict_and_explain import load_full_model_for_inference, get_transforms, generate_attention_map

# -----------------------------
# --- PAGE CONFIGURATION ---
# -----------------------------
st.set_page_config(page_title="AI X-Ray Analysis", page_icon="🤖", layout="wide")

# -----------------------------
# --- MODEL LOADING (Cached & with Correct Paths) ---
# -----------------------------
@st.cache_resource
def load_model():
    # --- THIS IS THE FIX ---
    # We build absolute paths from the script's location to ensure they are always correct.
    # Make sure these folder and file names EXACTLY match your repository.
    project_root = Path(__file__).resolve().parent
    chexpert_path = project_root / "checkpoints_full_dataset" / "best_model_epoch9_auc0.7952.pth"
    lora_path = project_root / "checkpoints_mura_lora" / "best_model_adapters"
    
    model = load_full_model_for_inference(chexpert_path, lora_path)
    return model

try:
    model = load_model()
    _, val_transform = get_transforms()
    model_loaded = True
except FileNotFoundError as e:
    st.error(f"Model files not found. The script looked for a file at: {e.filename}. Please ensure this path is correct in your GitHub repository.")
    model_loaded = False

# -----------------------------
# --- UI LAYOUT (No changes needed here) ---
# -----------------------------
st.title("🤖 Detection of Anomaly in human body through Xray")
st.sidebar.title("About")
st.sidebar.info(
    "This is a demonstration of a medical imaging foundation model, pre-trained on CheXpert "
    "and adapted for the MURA dataset using LoRA."
    "\n\n**Disclaimer:** This tool is for educational purposes only."
)

st.write("Upload a musculoskeletal X-ray to classify it and see where the AI is looking.")
uploaded_file = st.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None and model_loaded:
    col1, col2 = st.columns(2)
    image_pil = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.image(image_pil, caption="Uploaded X-Ray", use_container_width=True)

    with col2:
        with st.spinner("🧠 Analyzing..."):
            image_tensor = val_transform(image_pil).unsqueeze(0)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            image_tensor = image_tensor.to(device)
            
            with torch.no_grad():
                logits = model(image_tensor)
                probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]

            attention_map_img = generate_attention_map(model, image_tensor, image_pil)

            st.subheader("📊 Analysis Results")
            class_names = ["Normal (Negative)", "Abnormal (Positive)"]
            pred_idx = probabilities.argmax()
            pred_name = class_names[pred_idx]
            confidence = probabilities[pred_idx]
            
            st.metric(
                label="Prediction",
                value=pred_name,
                delta=f"Confidence: {confidence:.2%}",
                delta_color=("normal" if pred_name == "Normal (Negative)" else "inverse")
            )

            st.image(attention_map_img, caption="🧠 AI Attention Map", use_container_width=True)
            
            with st.expander("See Detailed Probabilities"):
                st.write({name: f"{prob:.4f}" for name, prob in zip(class_names, probabilities)})