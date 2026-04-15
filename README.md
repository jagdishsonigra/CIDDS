# 🚢 Container Identification and Damage Detection System (CIDDS)

> AI-powered system for automated container inspection using YOLOv8n and OCR.

---
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-success)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-Academic%20Use-orange.svg)](LICENSE)

## 📌 Overview

The **Container Identification and Damage Detection System (CIDDS)** is an AI-based solution designed to automate the inspection of shipping containers. It detects **physical damages** (rust, dents, cracks, etc.) and extracts **container identification codes (BIC/ISO 6346)**.

Manual inspection is slow and error-prone. CIDDS provides a **fast, accurate, and scalable software-only solution**.

📄 Based on project proposal (DES Pune University)

---

## 🎯 Objectives

- 🔍 Detect container damages (rust, dents, cracks, punctures)
- 🔢 Extract BIC container ID using OCR
- ✅ Ensure accuracy using validation logic
- ⚡ Fast inference (<50 ms target)
- 📊 Generate inspection-ready outputs

---

## 🧠 Model Used

- ✅ **YOLOv8n (Nano version)**
  - Lightweight  
  - Fast inference  
  - Suitable for low-resource systems  

⚠️ Experimental models like ResNet are NOT part of final implementation.

---

## 📂 Dataset

Datasets used:
- Roboflow (Container Damage Detection)
- Container Orientation Dataset (OCR region detection)

### 📁 Structure
dataset/
│
├── train/
│ ├── images/
│ └── labels/
│
├── valid/
│ ├── images/
│ └── labels/
│
├── test/
│ ├── images/
│ └── labels/

---

## ⚙️ Tech Stack

* **Core AI:** YOLOv8 (Ultralytics)
* **Computer Vision:** OpenCV
* **Language:** Python
* **Data Handling:** Pandas, NumPy
* **OCR (Optional):** PaddleOCR / EasyOCR

---

## 🧩 Pipeline Architecture

1.  **Input Image:** High-resolution capture of shipping containers.
2.  **Damage Detection:** YOLOv8 identifies dents, rust, or structural breaches.
3.  **Region Extraction:** Crops damaged areas for granular analysis.
4.  **OCR Processing:** Extracts container IDs/serial numbers.
5.  **Validation & Output:** Cross-references data and generates inspection reports.

---

## 🚀 Installation

Clone the repository and navigate to the project directory:

bash
git clone [https://github.com/jagdishsonigra/CIDDS.git](https://github.com/jagdishsonigra/CIDDS.git)
cd CIDDS

