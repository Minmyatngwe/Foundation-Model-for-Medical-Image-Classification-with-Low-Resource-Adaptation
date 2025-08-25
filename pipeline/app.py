import torch
import torch.nn.functional as F
from PIL import Image
import io
from fastapi import FastAPI, File, UploadFile
import uvicorn

# --- ROBUST PATHING AND IMPORTS ---
from pathlib import Path
import sys

# Get the absolute path of the script's directory (pipeline/)
SCRIPT_DIR = Path(__file__).resolve().parent
# Get the absolute path of the project root (the parent of pipeline/)
PROJECT_ROOT = SCRIPT_DIR.parent
# Add the project root to Python's path so it can find the other scripts
sys.path.append(str(PROJECT_ROOT))

# Now we can import from the other files in the pipeline directory
from pipeline.predict_and_explain import load_full_model_for_inference, get_transforms

# -----------------------------
# --- CONFIGURATION (using absolute paths) ---
# -----------------------------
# Use the absolute paths to ensure files are always found
CHEXPERT_MODEL_PATH = PROJECT_ROOT / "checkpoints/best_model_epoch9_auc0.7613.pth"
LORA_ADAPTER_PATH = PROJECT_ROOT / "checkpoints_mura_lora/best_model_adapters/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# -----------------------------
# --- MODEL LOADING & APP SETUP ---
# -----------------------------
print("Loading model using absolute paths...")
model = load_full_model_for_inference(CHEXPERT_MODEL_PATH, LORA_ADAPTER_PATH)
print("Model loaded successfully.")

app = FastAPI(title="X-Ray Classifier API")
_, val_transform = get_transforms(img_size=IMG_SIZE)


# -----------------------------
# --- API ENDPOINT ---
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    image_tensor = val_transform(image_pil).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]

    class_names = ["Normal (Negative)", "Abnormal (Positive)"]
    predicted_class_index = probabilities.argmax()
    predicted_class_name = class_names[predicted_class_index]
    confidence = float(probabilities[predicted_class_index])
    
    return {
        "predicted_class": predicted_class_name,
        "predicted_class_index": int(predicted_class_index),
        "confidence": confidence,
        "probabilities": {name: float(prob) for name, prob in zip(class_names, probabilities)}
    }

# -----------------------------
# --- MAIN EXECUTION ---
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    