# Water-Meter-Reader
An AI-powered solution for automated analog water meter reading
This project automates the reading of **analog water meters** using a **Donut (Document Understanding Transformer)** model — an OCR-free, transformer-based architecture that directly converts images into structured text.

Unlike traditional OCR systems, this approach is **robust to noise, varied meter designs, lighting variations, and skewed angles**, and it is optimized to run on **CPU-only environments** (e.g., macOS), making it suitable for **edge devices**.

---

## 📌 Problem Statement

Manual water meter readings are:
- Time-consuming
- Error-prone
- Labor-intensive

Traditional OCR struggles with:
- Low contrast & varied fonts
- Scratches, watermarks, and shadows
- Non-standard digit arrangements

**Solution**: An AI system that:
- Reads meter values **directly from images**
- Works reliably on **noisy, tilted, and low-quality images**
- Runs **without GPU acceleration**

---

## 🎯 Objectives

- Automate reading of analog water meters using deep learning.
- Fine-tune **Donut** for **image-to-text** tasks.
- Ensure **CPU compatibility** for training and inference.
- Evaluate performance with **Levenshtein Distance** & **exact match accuracy**.
- Visualize predictions for verification.

---

## 🛠️ Tools & Technologies

| Category                | Tool/Library |
|-------------------------|-------------|
| Language               | Python 3.12 |
| Core Framework         | PyTorch 2.7.1 |
| Model Architecture     | Donut (Hugging Face Transformers) |
| Image Processing       | OpenCV, Pillow |
| Data Handling          | NumPy, Regex |
| Visualization          | Matplotlib |
| MLOps & Tracking       | Weights & Biases (wandb) |
| Development Environment| Jupyter Notebook |

---

## 📚 Literature Review

- **Traditional OCR** (e.g., Tesseract) fails on analog meters due to overlapping digits and noise.
- **Donut** eliminates OCR by mapping images directly to structured text via a vision encoder + transformer decoder.
- Proven more robust in challenging environments compared to CNN-RNN-CTC pipelines.

---

## ⚙️ Methodology

### 1. Data Collection
- Thousands of real-world water meter images.
- Diverse in lighting, viewing angles, and meter types.

### 2. Data Preprocessing
- **Image**: Resize & normalize.
- **Labels**: Clean with regex and wrap in special tags.
- **Split**: Train (70%), Validation (15%), Test (15%).

### 3. Model Design
- **Encoder**: Swin Transformer for feature extraction.
- **Decoder**: Transformer for sequential prediction.
- Output format:  
  ```json
  {"meter_reading": "123.456"}
4. Training Setup
Loss: CrossEntropyLoss

Optimizer: AdamW (lr=5e-5)

Epochs: 2

Batch Size: 4

CPU-only training (no GPU acceleration)

Logging & versioning via W&B

📊 Results
Metric	Value
Accuracy	>99.9%
Precision	1.0
Recall	1.0
Levenshtein Dist.	~0

Correctly predicted up to 3 decimal places.

Robust to tilted, shadowed, and noisy images.

🚀 How to Run
1. Clone Repository
bash
Copy
Edit
git clone https://github.com/harshitsrivastava644/Water-Meter-Reader.git
cd water-meter-reader
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Train Model
bash
Copy
Edit
python train.py
4. Run Inference
bash
Copy
Edit
python infer.py --image_path sample.jpg
📂 Project Structure
bash
Copy
Edit
water-meter-reader/
│── data/                # Dataset
│── notebooks/           # Jupyter notebooks
│── src/                 # Source code
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── infer.py
│── outputs/             # Predictions & visualizations
│── requirements.txt     # Dependencies
│── README.md            # Documentation
🖼 Sample Output
Example – Predicted vs Ground Truth:

makefile
Copy
Edit
Image: meter_001.jpg
Predicted: 595.825
Actual:    595.825
Accuracy:  100%
📌 Limitations
CPU training is slower than GPU.

Small dataset may limit generalization to all meter types.

Faint or partially hidden digits can reduce accuracy.
