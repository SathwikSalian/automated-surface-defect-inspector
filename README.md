# 🔍 Automated Surface Defect Inspection System

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8n-brightgreen)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow)

An end-to-end Computer Vision application designed for industrial quality control. The system identifies, localizes, and classifies surface anomalies (scratches, inclusions, patches, crazing, pitted surface, rolled-in scale) in real-time using a fine-tuned **YOLOv8** model and an interactive **Streamlit** web application.

---

## 🌟 Key Features

* **Real-time Anomaly Detection:** Identifies defect locations and bounding boxes with millisecond inference times (~70ms on CPU).
* **Defect Classification:** Categorizes surface flaws into specific industrial classes with confidence scoring.
* **Interactive Web Interface:** Drag-and-drop file uploader for quality control operators with instant Pass/Fail alerts.
* **Low System Overhead:** Optimized using a lightweight `YOLOv8n` backbone suitable for edge deployments.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.10+
* **Computer Vision & Deep Learning:** Ultralytics (YOLOv8), PyTorch, OpenCV, Pillow
* **Frontend Web Framework:** Streamlit
* **Dataset Format:** Roboflow / YOLOv8 Annotation Format

---

## 📂 Project Structure

```text
Quality_Control_Project/
│
├── dataset/                     # Local dataset directory (train/valid splits)
│   ├── data.yaml                # Class mapping & dataset paths
│   ├── train/                   # Training images & YOLO annotation text files
│   └── valid/                   # Validation images & text files
│
├── runs/
│   └── detect/
│       └── defect_model/        # Trained weights (best.pt)
│
├── train.py                     # Training script for model fine-tuning
├── app.py                       # Streamlit web dashboard
├── .gitignore                   # Ignored files (large datasets, .pt weights)
└── README.md                    # Project documentation

How to run:
1.python app.py in  vscode terminal
2.python -m streamlit run app.py