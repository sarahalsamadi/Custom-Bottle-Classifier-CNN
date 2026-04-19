# 🧴 Custom Water Bottle Classifier (Deep Learning)

An end-to-end Computer Vision pipeline designed to recognize and classify specific water bottle brands (e.g., HAYAWIYA, SHAMLAN). This project demonstrates a full machine learning lifecycle, from manual data acquisition to model deployment.


## 🚀 Project Overview
Unlike projects using pre-made datasets, this system was built using a **handcrafted dataset** to simulate real-world AI challenges, including:
1. **Manual Data Collection:** Captured raw footage via smartphone cameras using IP streaming scripts.
2. **Data Engineering:** Automated cleaning, rotation correction (EXIF-based), and duplicate removal using SHA-1 hashing.
3. **Transfer Learning:** Fine-tuning the **MobileNetV2** architecture for high-speed, lightweight inference.

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Deep Learning:** TensorFlow, Keras
- **Computer Vision:** OpenCV, YOLOv8 (for segmentation support)
- **Data Processing:** Pillow (Image transformations), Scikit-learn (Dataset splitting)

## 📁 Repository Structure
- `image_captured.py`: Script to stream video from a phone/IP camera and save labeled frames.
- `prossesing.py`: Advanced preprocessing tool for center-cropping, resizing, and filtering corrupt/duplicate images.
- `main.ipynb`: The primary training notebook containing data augmentation, training loops, and performance metrics.
- `best_model.keras`: The final trained model weights optimized for accuracy.

## ⚙️ How to Run
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

2. **Preprocess Raw Data:**
Standardize your manually captured images by running:
   ```bash
   python prossesing.py --in_dir ./raw_data --out_dir ./dataset

3. **Train & Evaluate:**

Open main.ipynb in Jupyter Lab or Google Colab to run the training pipeline and test the model on external images.

## 📊 Key Features
- **Lightweight Architecture:** Uses MobileNetV2, making it suitable for mobile or edge device integration.
- **Robustness:** Trained on images with varying lighting and backgrounds, reflecting real-world deployment conditions.
- **Automated Pipeline:** Scripts handle the heavy lifting of data preparation, ensuring high-quality input for the neural network.

### Developed by: *Sarah Ammar*

- *Artificial Intelligence Engineering Student*
