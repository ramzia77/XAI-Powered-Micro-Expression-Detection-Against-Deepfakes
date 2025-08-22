# MED – Micro-Expression Detector

**MED (Micro-Expression Detector)** is a conceptual web-based application designed to identify and classify human facial micro-expressions in real time. It aims to enhance **deepfake-resistant biometric authentication** by combining live video processing, emotion recognition, and planned integrations for explainability and blockchain-based audit trails.

---

## 🚀 Features

- **Real-time webcam streaming** and frame capture with OpenCV.
- **Face detection** using Haar Cascade Classifier.
- **Emotion recognition** with a pre-trained CNN model (Emo0.1 trained on FER2013).
- **Seven emotion classes:** Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised.
- **Prototype deepfake detection** using Celeb-DF dataset (temporal micro-expressions).
- **Low-latency inference:** ~137–230 ms per frame.
- **Planned integrations:**
  - **Grad-CAM / SHAP** for explainable AI (visual heatmaps and transparency).
  - **Blockchain logging** for tamper-proof liveness and deepfake verification.

---

## 🧩 System Overview

**Pipeline:**
1. Capture frames from webcam (OpenCV).
2. Detect and preprocess faces (grayscale, ROI crop, resize 48×48, normalize).
3. Classify expressions using Emo0.1 CNN.
4. Log inference time per frame (~185 ms average).
5. Prototype deepfake detection using Celeb-DF videos.
6. (Future) Visualize influential regions via Grad-CAM.
7. (Future) Log hashed outputs on blockchain for verifiable audits.


---

## 📊 Key Results (Prototype)

- **FER2013 Emotion Classification Accuracy:** ~62.5%
- **Celeb-DF Deepfake Detection Accuracy:** ~78.3% (custom CNN classifier)
- **Latency:** Near real-time; stable performance under typical conditions
- **Limitations:** Lower accuracy in poor lighting or low-quality cameras; Neutral/Sad harder to differentiate

---

## 🛠 Key Technologies

- **Languages/Frameworks:** Python 3.10, TensorFlow 2.11, Keras, OpenCV
- **Models:** CNN (Emo0.1), planned LSTM/3D CNN for temporal analysis
- **Datasets:** FER2013 (emotions), Celeb-DF v2 (deepfake detection)
- **Explainability (planned):** Grad-CAM, SHAP
- **Blockchain (conceptual):** Hyperledger Fabric/Ethereum, SHA-256 hashing, smart contracts

---

## ⚠️ Current Limitations & Future Work

- Grad-CAM explainability not yet integrated (planned for future).
- Lighting and camera quality affect prediction accuracy.
- Integration of sequence-based models (CNN-LSTM) to better capture temporal micro-expressions.
- Blockchain layer for decentralized, verifiable audit logs (conceptual design complete).

---

## 🧪 Testing Environment

- Python 3.10  
- TensorFlow 2.11  
- Average inference time: ~185 ms per frame  
- Tested on FER2013 and Celeb-DF datasets

---

## 📷 Demo (Planned)

- **Live webcam feed:** Detect faces and classify emotions with confidence scores.
- **Image upload mode:** Predict emotion from static images.
- **Visualizations:** Planned Grad-CAM heatmaps and deepfake confidence scores.

_Add sample screenshots or GIFs to `/docs` and link them here once available._

---

## 📚 References

- FER2013 Dataset  
- Celeb-DF v2 Dataset  
- Selvaraju et al., 2017 – Grad-CAM paper

---

## 🤝 Contributing

Contributions are welcome!  
Ideas for improvement include:
- Integrating Grad-CAM explainability
- Adding LSTM/3D CNN for temporal modeling
- Blockchain proof-of-concept for audit logs

Fork the repo and submit a PR.

---

**Note:** MED is currently a conceptual and prototype-level system for research and educational purposes.
