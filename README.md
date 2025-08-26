# Medical Imaging Foundation Model for X-Ray Classification and Low-Resource Adaptation

**Author:** Akshat Gupta

**Date:** August 25, 2025

You can try the model here:
[Live Demo](https://murachxpertfoundationmodel.streamlit.app/)

## Executive Summary

This project develops a deep learning foundation model for medical X-ray analysis. The model was pre-trained on the large-scale CheXpert dataset and adapted to a specialized musculoskeletal classification task (MURA dataset) using parameter-efficient fine-tuning (LoRA).

Key results:

* **Foundation Model**: Vision Transformer (ViT) pre-trained on CheXpert.
* **Adaptation**: Efficiently fine-tuned on MURA using LoRA.
* **Performance**: Achieved peak validation **Mean AUC = 0.795** after 100 epochs.
* **Deployment**: Includes a FastAPI backend and Streamlit-based web UI for real-time inference with explainability via Attention Rollout.



## 1. Project Overview and Objectives

### Goal

Develop a reusable foundation model for medical image classification that addresses the scarcity of large labeled datasets in specialized medical domains.

### Objectives

1. **Foundation Model Pre-training**: Train ViT on CheXpert dataset.
2. **Low-Resource Adaptation**: Apply LoRA for efficient adaptation to MURA dataset.
3. **Model Explainability**: Implement Attention Rollout to visualize decision-making.
4. **Deployment**: Deliver model with API + web UI for predictions and explainability.



## 2. Methodology and Workflow

### 2.1 Data Preparation

* **Datasets**: CheXpert (\~224k chest X-rays), MURA (\~40k musculoskeletal images).
* **Transfer**: Used `rsync` for robust dataset transfer.
* **Cleaning**: Applied “U-Ones” strategy to handle uncertain labels in CheXpert.

### 2.2 Foundation Model Training

* **Architecture**: `timm` ViT (`vit_base_patch16_224`, ImageNet pre-trained).
* **Loss**: `BCEWithLogitsLoss` with `pos_weight` for class imbalance.
* **Training**: 100 epochs on CheXpert.
* **Server Management**: Long runs handled with `screen` sessions.

### 2.3 Low-Resource Adaptation with LoRA

* Loaded best foundation checkpoint.
* Replaced classification head for binary Normal/Abnormal MURA task.
* Applied **LoRA adapters** (Hugging Face PEFT library), training \~1.8% parameters.
* Fine-tuned efficiently with minimal epochs.

### 2.4 Explainability

* Implemented **Attention Rollout**: aggregated ViT attention maps into heatmaps showing model focus on X-ray regions.

### 2.5 Deployment

* **Backend**: FastAPI with `/predict` endpoint.
* **Frontend**: Streamlit app with file uploader, prediction output, and heatmap visualization.



## 3. Challenges and Solutions

1. **Data Transfer**

   * *Problem*: Unreliable transfers via SCP.
   * *Solution*: Used `rsync` for resumable, checksum-verified transfers.

2. **Python Environment Issues**

   * *Problem*: `ModuleNotFoundError` despite installed libraries.
   * *Solution*: Corrected use of virtual environment (`python3` vs `/usr/bin/python3`).

3. **Model Loading**

   * *Problem*: `pickle.UnpicklingError` due to PyTorch 2.6 `weights_only=True`.
   * *Solution*: Used `weights_only=False` for trusted checkpoints.

4. **Explainability Hooks**

   * *Problem*: `timm` ViT attention not returning scores.
   * *Solution*: Hooked into `block.attn.attn_drop` to capture raw attention.



## 4. Key Learning Outcomes

* **Technical Skills**

  * PyTorch: advanced training & custom datasets.
  * LoRA & PEFT: efficient fine-tuning methods.
  * XAI: implemented Attention Rollout for ViTs.
  * Deployment: integrated FastAPI + Streamlit.
  * Server Management: used `screen` and `rsync`.

* **Insights**

  * Foundation models can adapt rapidly across domains.
  * Debugging skills are as critical as modeling.
  * Importance of full-stack project structure (data → model → deployment).



## 5. Future Work

### Model Accuracy

* Test alternative backbones (ConvNeXt, newer ViTs).
* Use advanced loss functions (e.g., Focal Loss).
* Hyperparameter search with Optuna/Ray Tune.

### Data & Robustness

* Multi-dataset pretraining (CheXpert + ChestX-ray14 + mammography).
* Apply k-fold cross-validation.

### Deployment & Usability

* Containerize with Docker (FastAPI + Streamlit).
* Add user feedback loop for error correction and continuous learning.


```
{
"Normal (Negative)":0.40870174765586853
"Abnormal (Positive)":0.5912982821464539
}
```


## 6. Installation & Requirements

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/medical-xray-foundation-model.git
cd medical-xray-foundation-model
pip install -r requirements.txt
```

### requirements.txt

```text
torch>=2.0
torchvision>=0.15
timm>=0.9
transformers>=4.40
peft>=0.10
numpy
pandas
scikit-learn
matplotlib
fastapi
uvicorn
streamlit
Pillow
```

---

## 7. How to Run

1. **Pre-train**: Train foundation model on CheXpert.

2. **Fine-tune**: Adapt model to MURA with LoRA.

3. **Deploy**:

   ```bash
   # Start FastAPI backend
   uvicorn app:app --reload  

   # Start Streamlit frontend
   streamlit run ui.py        
   ```

4. **Predict**: Upload an X-ray in the UI → Get Normal/Abnormal prediction + attention map.

---

## 8. API Usage

### Endpoint

`POST /predict`

### Request

* Input: image file (`.png`, `.jpg`, `.jpeg`)
* Example (cURL):

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@sample_xray.png"
```

### Response

```json
{
  "prediction": "Abnormal",
  "probability": 0.873,
  "attention_map": "base64_encoded_image_string"
}
```

---

## 9. License & Citation

### License

This project is released under the **MIT License**.

### Citations

If you use this work in research, please cite the datasets:

* **CheXpert**:
  Irvin J, Rajpurkar P, Ko M, et al. *CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.* AAAI 2019.

* **MURA**:
  Rajpurkar P, Irvin J, Bagul A, et al. *MURA: Large Dataset for Abnormality Detection in Musculoskeletal Radiographs.* arXiv:1712.06957.

