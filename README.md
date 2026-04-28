# 🫁 PneumoScan AI — Chest X-Ray Pneumonia Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?style=flat-square&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Model](https://img.shields.io/badge/Model-ResNet50-blue?style=flat-square)
![Explainability](https://img.shields.io/badge/Explainability-Grad--CAM-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**AI-powered chest X-ray analysis for pneumonia detection with explainable heatmaps.**  
Early screening · Clinical assistance · Visual explainability · Real-time inference

</div>

---

## 🩺 What It Does

PneumoScan AI analyzes chest X-rays using deep learning and classifies them into:

- **Normal**
- **Pneumonia**

It also generates **Grad-CAM heatmaps** to highlight important regions.

---

## ✨ Features

- Deep learning model (ResNet50)
- Grad-CAM explainability
- FastAPI backend
- Image upload UI
- Confidence scoring
- Patient metadata support
- Clean dashboard UI

---

## 🚀 Quickstart

### Install

```bash
git clone https://github.com/YOUR_USERNAME/pneumonia-detection.git
cd pneumonia-detection

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
uvicorn app.main:app --reload --port 8000
```

---

## 📖 API

### POST /predict

```bash
curl -X POST http://localhost:8000/predict -F "file=@xray.jpg"
```

Response:
```json
{
  "prediction": "Pneumonia",
  "confidence": 0.93
}
```

---

## ⚠️ Disclaimer

This project is for educational purposes only.  
Not a medical diagnostic tool.

---

## 📄 License

MIT License
