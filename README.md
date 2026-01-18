# RandFourBoost

RandFourBoost is a thesis-based machine learning system that integrates Random Forest, Fourier-based feature analysis, and boosting techniques to enhance classification accuracy, robustness, and model stability in data mining applications.

---

## 📌 Abstract
This study proposes RandFourBoost, a hybrid ensemble learning approach that combines Random Forest for feature diversity, Fourier Transform for frequency-domain feature enhancement, and boosting for iterative error reduction. The system aims to improve classification performance compared to traditional ensemble models.

---

## 🎯 Objectives
- To design a hybrid ensemble learning model using Random Forest, Fourier Transform, and Boosting
- To enhance classification accuracy and robustness
- To evaluate the performance of the proposed method using standard metrics
- To compare RandFourBoost with existing classification models

---

## 🧠 Methodology

### 1. Data Collection
This study utilized the **GTZAN Music Genre Classification Dataset**, obtained from Kaggle. The dataset consists of **1,000 audio tracks**, each with a duration of **30 seconds**, evenly distributed across **10 music genres**: Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, and Rock.

The GTZAN dataset is widely used as a benchmark in music genre classification research and was selected to ensure consistency, reproducibility, and comparability with existing studies.

Dataset source:  
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

---

### 2. Data Preprocessing
- Audio signal loading and resampling
- Removal of corrupted or duplicate audio files
- Feature normalization and scaling
- Dataset partitioning into training and testing sets

---

### 3. Feature Transformation (Fourier Analysis)
A Fourier Transform was applied to the audio signals to extract frequency-domain features, capturing spectral characteristics that enhance the representation of musical patterns for genre classification.

---

### 4. Random Forest Modeling
Random Forest was employed to:
- Generate diverse decision trees from extracted features
- Reduce overfitting through ensemble averaging
- Improve generalization across unseen audio samples

---

### 5. Boosting Integration
Boosting techniques were applied to:
- Emphasize misclassified audio samples
- Iteratively strengthen weak learners
- Improve overall classification accuracy and robustness

---

### 6. RandFourBoost Framework
The RandFourBoost framework integrates Fourier-based audio features with Random Forest outputs, enhanced through boosting iterations to form a robust hybrid ensemble model for music genre classification.

---

## 🏗️ System Architecture
<p align="center">
  <img src="https://github.com/user-attachments/assets/f1141d78-3d07-4bce-9ee1-42b13910e54e" width="650">
</p>

<p align="center">
  <em>Figure 1. Overall System Architecture of the RandFourBoost Model</em>
</p>

---

## 📊 Evaluation Metrics
The performance of RandFourBoost was evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 📈 Results and Discussion
Experimental results show that RandFourBoost outperforms traditional ensemble models in terms of accuracy and stability, demonstrating its effectiveness in complex classification tasks.

---

## 🛠️ Technologies Used
- Python
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

---

## 📁 Project Structure
```text
RandFourBoost/
├── data/
├── images/
├── src/
├── README.md
└── LICENSE
```

# Copyright Notice

© 2025 Cyrus Manipon Cavero, Joey Castro Gannaban, Ricardo Querubin Camungao.  
**Title:** RandFourBoost: An Enhanced Classification Model for Music Production  
**AI Generated:** Yes, ChatGPT  
**All rights reserved.**  

This work is registered with the Intellectual Property Office of the Philippines (Certificate No. 2025-06674-A).  
Validate at [www.ipophil.gov.ph](https://www.ipophil.gov.ph)


