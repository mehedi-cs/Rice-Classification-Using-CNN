# 🌾 Rice Classification Using Deep Learning

## 📘 Project Overview
This project focuses on **analyzing and classifying different types of rice** — one of the world’s most essential grain products. Due to their genetic diversity, rice varieties show unique characteristics in **texture, shape, and color**, making them ideal for computational classification.

This system automates the classification of rice into **five major types cultivated in Turkey**:
- Arborio  
- Basmati  
- Ipsala  
- Jasmine  
- Karacadag  

A dataset of **75,000 images** was used (15,000 per class).  
Additionally, a **feature-based dataset** was created with **106 extracted features**, including:
- 12 morphological features  
- 4 shape-related features  
- 90 color-related features  

---

## 🧠 Models Used
Three different architectures were implemented to compare performance:

| Dataset Type | Model Used | Description |
|---------------|-------------|--------------|
| Image Dataset | **CNN (Convolutional Neural Network)** | Automatically extracts spatial and texture-based features from rice images. |
| Feature Dataset | **ANN (Artificial Neural Network)** | Classifies rice based on pre-extracted numerical features. |
| Feature Dataset | **DNN (Deep Neural Network)** | Captures deeper feature representations for higher accuracy. |

---

## 🧩 Project Structure
