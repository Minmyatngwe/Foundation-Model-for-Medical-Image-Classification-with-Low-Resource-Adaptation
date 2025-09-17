import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
import timm
from peft import PeftModel
from pathlib import Path
from huggingface_hub import hf_hub_download

# -----------------------------
# --- PAGE CONFIGURATION ---
# -----------------------------
st.set_page_config(page_title="AI X-Ray Analysis", page_icon="🤖", layout="wide")

# -------------------------------------------------------------------
# --- ALL HELPER FUNCTIONS ARE NOW INSIDE THIS SINGLE FILE ---
# -------------------------------------------------------------------
def create_foundation_model(num_classes_chexpert=14, model_name="vit_base_patch16_224"):
    model = timm.create_model(model_name, pretrained=False)
    model.head = nn.Linear(model.head.in_features, num_classes_chexpert)
    return model

def get_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_full_model_for_inference(chexpert_path, lora_path_dir):
    model = create_foundation_model()
    checkpoint = torch.load(chexpert_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.head = nn.Linear(model.head.in_features, 2)
    model = PeftModel.from_pretrained(model, lora_path_dir)
    model = model.merge_and_unload()
    model.eval()
    return model

def generate_attention_map(model, input_tensor, original_image_pil, img_size=224):
    attention_maps = []
    hooks = []
    device = input_tensor.device
    for block in model.blocks:
        def hook_fn(module, args, output):
            attention_maps.append(args[0])
        hooks.append(block.attn.attn_drop.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(input_tensor)

    for hook in hooks:
        hook.remove()

    num_patches = (img_size // model.patch_embed.patch_size[0]) ** 2
    residual_attn = torch.eye(num_patches + 1, device=device)
    attn_mats = [attn.mean(dim=1) for attn in attention_maps]
    for attn_mat in attn_mats:
        aug_attn_mat = attn_mat + torch.eye(attn_mat.size(1), device=device)
        aug_attn_mat = aug_attn_mat / aug_attn_mat.sum(dim=-1).unsqueeze(-1)
        residual_attn = torch.matmul(aug_attn_mat, residual_attn)

    grid_attn = residual_attn[0, 1:]
    grid_size = int(np.sqrt(num_patches))
    attention_grid = grid_attn.reshape(grid_size, grid_size).cpu().numpy()

    mask = cv2.resize(attention_grid, (img_size, img_size))
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    original_img_np = np.array(original_image_pil.resize((img_size, img_size)))
    superimposed_img = np.uint8(0.6 * original_img_np + 0.4 * heatmap)
    return Image.fromarray(superimposed_img)

# -----------------------------
# --- MODEL LOADING (From Hugging Face Hub) ---
# -----------------------------
@st.cache_resource
def load_model_from_hub():
    HF_REPO_ID = "akshatgg/CXR-MURA-Foundation-Model"
    
    with st.spinner(f"Downloading model files from Hugging Face Hub... This may take a moment."):
        # --- THIS IS THE FIX ---
        # Provide the FULL path of the files within the repository.
        chexpert_path = hf_hub_download(
            repo_id=HF_REPO_ID, 
            filename="best_model_epoch18_auc0.7952.pth"
        )
        adapter_file_path = hf_hub_download(
            repo_id=HF_REPO_ID, 
            filename="adapter_model.safetensors"
        )
        # We also need the config file from the same directory
        hf_hub_download(
            repo_id=HF_REPO_ID, 
            filename="adapter_config.json"
        )
        
        lora_path_dir = Path(adapter_file_path).parent

    model = load_full_model_for_inference(chexpert_path, lora_path_dir)
    return model

# -----------------------------
# --- MAIN SCRIPT LOGIC ---
# -----------------------------
st.title("🤖 AI-Powered X-Ray Anomaly Detection")

model_loaded = False
try:
    model = load_model_from_hub()
    st.success("Model loaded successfully!")
    val_transform = get_transforms()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model from Hugging Face Hub. Please double-check your repository name and that all files are present. Error: {e}")

# ... (The rest of your UI code is identical to the last version) ...
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
                probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
            attention_map_img = generate_attention_map(model, image_tensor, image_pil)
            st.subheader("📊 Analysis Results")
            class_names = ["Normal (Negative)", "Abnormal (Positive)"]
            pred_idx = probabilities.argmax()
            pred_name = class_names[pred_idx]
            confidence = probabilities[pred_idx]
            st.metric(label="Prediction", value=pred_name, delta=f"Confidence: {confidence:.2%}", delta_color=("normal" if pred_name == "Normal (Negative)" else "inverse"))
            st.image(attention_map_img, caption="🧠 AI Attention Map", use_container_width=True)