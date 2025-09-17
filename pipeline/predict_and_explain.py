import streamlit as st
from PIL import Image
import torch
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

# --- ROBUST PATHING & IMPORTS ---
# This ensures the script can always find the 'pipeline' folder
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR / "pipeline"))

# Now we can reliably import our custom functions
from predict_and_explain import create_foundation_model, get_transforms, generate_attention_map
from peft import PeftModel

# -----------------------------
# --- PAGE CONFIGURATION ---
# -----------------------------
st.set_page_config(page_title="AI X-Ray Analysis", page_icon="🤖", layout="wide")

# -----------------------------
# --- Redefined Loading Function (for HF Hub) ---
# -----------------------------
# This is a modified version of your original loading function
def load_full_model_for_inference(chexpert_path, lora_path_dir):
    model = create_foundation_model(num_classes_chexpert=14)
    checkpoint = torch.load(chexpert_path, map_location='cpu', weights_only=False) 
    model.load_state_dict(checkpoint['model_state'])
    model.head = torch.nn.Linear(model.head.in_features, 2)
    model = PeftModel.from_pretrained(model, lora_path_dir)
    model = model.merge_and_unload()
    model.eval()
    print("Model ready for inference.")
    return model

# -----------------------------
# --- MODEL LOADING (From Hugging Face Hub) ---
# -----------------------------
@st.cache_resource
def load_model_from_hub():
    # --- IMPORTANT: UPDATE THIS WITH YOUR DETAILS ---
    HF_REPO_ID = "akshatgupta-dev/Foundation-Model-for-Medical-Image-Classification-with-Low-Resource-Adaptation"
    
    st.info(f"Downloading model files from Hugging Face Hub: {HF_REPO_ID}. This may take a moment on first run.")
    
    # Download each file from the Hub. It will be cached locally.
    chexpert_path = hf_hub_download(repo_id=HF_REPO_ID, filename="checkpoints_full_dataset/best_model_epoch9_auc0.7952.pth")
    # Note: peft requires the directory, so we download one file to find the directory
    adapter_file_path = hf_hub_download(repo_id=HF_REPO_ID, filename="checkpoints_mura_lora/best_model_adapters/adapter_model.safetensors")
    lora_path_dir = Path(adapter_file_path).parent

    model = load_full_model_for_inference(chexpert_path, lora_path_dir)
    return model

try:
    model = load_model_from_hub()
    _, val_transform = get_transforms()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model from Hugging Face Hub. Please ensure the HF_REPO_ID is correct and files exist. Error: {e}")
    model_loaded = False

# -----------------------------
# --- UI LAYOUT ---
# -----------------------------
st.title("🤖 AI-Powered X-Ray Anomaly Detection")

# ... (The rest of your UI code remains exactly the same) ...
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