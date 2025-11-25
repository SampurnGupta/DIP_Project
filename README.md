# 🌞 Solar Imaging & Digital Image Processing Toolkit

A lightweight toolkit for exploring **solar images**, visualizing **active regions**, and applying **digital image processing techniques** for feature extraction and visualization.  
This project includes:

- **Solar flare / sunspot annotation viewer**
- **Bounding box selection + region cropping**
- **ROI DIP operations** (Equalization, CLAHE, Gaussian Blur, Sharpen, LBP, HOG, Gabor, Edges, etc.)
- **1600 Å style solar UV simulator**

## DataSet used:
https://zenodo.org/records/4435219
now the images are enhanced via a DIP Pipeline then the Rectangle Sections from the labels are extracted and used to train ML Model. (the entire image is not used ot train the ML Model)

## 📁 Project Overview

### 1. Annotation Viewer  
Loads images + labels, draws bounding boxes, allows cropping + downloading + DIP processing.

### 2. DIP Processing on ROI  
Includes:
- Equalization  
- CLAHE  
- Gaussian Blur  
- Sharpening  
- Canny Edges  
- LBP  
- Gabor  
- HOG  
- More...

### 3. 1600 Å Style Simulator  
Transforms RGB solar images into pseudo-ultraviolet AIA 1600 Å style imagery.

## 🛠️ Installation
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Running
```bash
streamlit run app.py
streamlit run app_1600_style.py
```

## 📦 Recommended Structure
```
project/
├── app.py
├── app_1600_style.py
├── README.md
├── requirements.txt
├── Folder1/ containes the images of all the classes (b,c,m,x: refer original Dataset) Combined!
├── models/  contains all the trained ML Models
└── Folder2/ contains the labels corresponding to the images in folder1
```
