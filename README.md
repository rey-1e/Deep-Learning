# Deep Learning Laboratory

**Vishwakarma Institute of Technology, Pune**  
*(An Autonomous Institute Affiliated to Savitribai Phule Pune University)*  
**Department:** Computer Science and Engineering (Artificial Intelligence)  
**Academic Year:** 2026-27 | **Semester:** V  

---

### Student Information
- **Name:** Raj Kanade
- **Roll No:** 26
- **PRN:** 12413760
- **Batch:** B3
- **Subject:** Deep Learning

---

## Practical Assignments Overview

| Practical No. | Problem Statement / Topic | Implementation | Google Colab | Report |
| :---: | :--- | :---: | :---: | :---: |
| **Practical 1** | TensorFlow/Keras Setup, Data Preprocessing, Normalization, Fashion MNIST | [Notebook](assignment_1/Assignment-1%20on%20fashion%20mnist.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rey-1e/Deep-Learning/blob/main/assignment_1/Assignment-1%20on%20fashion%20mnist.ipynb) | [PDF Report](assignment_1/Assignment-1.pdf) |
| **Practical 2** | Multilayer Perceptron (MLP) for Iris Dataset Classification | [Notebook](assignment_2/Iris_Assign_2.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rey-1e/Deep-Learning/blob/main/assignment_2/Iris_Assign_2.ipynb) | [PDF Report](assignment_2/Assignment%202.pdf) |
| **Practical 3** | Forward & Backpropagation in ANN, Learning Rates & Epochs Analysis | [Notebook](assignment_3/Assignment_3%20fashion%20mnist.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rey-1e/Deep-Learning/blob/main/assignment_3/Assignment_3%20fashion%20mnist.ipynb) | [PDF Report](assignment_3/assignment-3.pdf) |
| **Practical 4** | Time-Series Weather Forecasting with LSTM on Jena Climate Dataset | [Notebook](Assignment-4/Assignment_4_JENA_CLIMATE_DATASET.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rey-1e/Deep-Learning/blob/main/Assignment-4/Assignment_4_JENA_CLIMATE_DATASET.ipynb) | [PDF Report](Assignment-4/Assignment-4.pdf) |
| **Practical 5** | Sequence Classification Comparison: Simple RNN vs LSTM vs GRU (Reuters) | [Notebook](Assignment-5/RNN%20vs%20LSTM%20vs%20GRU.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rey-1e/Deep-Learning/blob/main/Assignment-5/RNN%20vs%20LSTM%20vs%20GRU.ipynb) | [PDF Report](Assignment-5/Assignment-5.pdf) |
| **Practical 6** | Convolutional Neural Network (CNN) for Maize/Corn Leaf Disease Classification | [Notebook](assignment_6/CNN_Assign_6_Maize_leaf_disease.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rey-1e/Deep-Learning/blob/main/assignment_6/CNN_Assign_6_Maize_leaf_disease.ipynb) | [PDF Report](assignment_6/Assignment-6.pdf) |
| **Practical 7** | Transfer Learning (AlexNet, VGG16, ResNet50, EfficientNetB0) on CIFAR-10 | [Notebook](Assignment-7/Assignment_7_CIFAR_10.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rey-1e/Deep-Learning/blob/main/Assignment-7/Assignment_7_CIFAR_10.ipynb) | [PDF Report](Assignment-7/Assignment%20-7.pdf) |
| **Practical 8** | Sentiment Analysis using Pre-trained BERT on Rotten Tomatoes | [Notebook](Assignment-8/BERT.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rey-1e/Deep-Learning/blob/main/Assignment-8/BERT.ipynb) | [PDF Report](Assignment-8/Assignment-8.pdf) |

---

## 🚀 Running on Google Colab

Every practical assignment is pre-configured to run directly in [Google Colab](https://colab.research.google.com/) with zero manual file setup. Datasets are automatically loaded or downloaded within the notebook runtime.

### Recommended Colab Hardware Accelerators

| Practical | Topic | Recommended Runtime | Notes |
| :---: | :--- | :---: | :--- |
| **Practical 1** | Fashion MNIST ANN Preprocessing | **CPU** | Runs in seconds |
| **Practical 2** | Iris Dataset MLP Classification | **CPU** | Lightweight tabular dataset |
| **Practical 3** | Learning Rates & Epochs Analysis | **CPU** | Quick iterative training |
| **Practical 4** | Weather Forecasting with LSTM | **CPU / T4 GPU** | Resampled hourly time-series |
| **Practical 5** | RNN vs LSTM vs GRU Comparison | **CPU / T4 GPU** | 10 epochs per model on Reuters |
| **Practical 6** | Maize Leaf Disease CNN | **T4 GPU** | Faster image dataset training |
| **Practical 7** | Transfer Learning on CIFAR-10 | **T4 GPU** | Required for deep CNN backbones (ResNet, VGG) |
| **Practical 8** | BERT Sentiment Classification | **T4 GPU** | Required for Transformer fine-tuning |

### How to Open & Run in Google Colab

#### Option 1: 1-Click Launch from GitHub
Click any of the **Open In Colab** badges in the table above or directly at the top of any `.ipynb` notebook file in this repository.

#### Option 2: Open from Google Colab Interface
1. Go to [colab.research.google.com](https://colab.research.google.com/).
2. In the dialog, select the **GitHub** tab.
3. Paste repository URL: `https://github.com/rey-1e/Deep-Learning` (or search `rey-1e/Deep-Learning`).
4. Select branch `main` and choose any assignment notebook.

#### Enabling GPU Acceleration in Colab
For Practicals 6, 7, and 8, enable the GPU accelerator:
1. In the Colab top menu, click **Runtime** > **Change runtime type**.
2. Select **T4 GPU** under *Hardware accelerator*.
3. Click **Save**.

#### Saving Changes
To save your experiments, go to **File** > **Save a copy in Drive** or **File** > **Save a copy in GitHub**.

---

## Directory Structure

```text
Deep-Learning/
├── README.md
├── assignment_1/
│   ├── Assignment-1 on fashion mnist.ipynb
│   └── Assignment-1.pdf
├── assignment_2/
│   ├── Iris.csv
│   ├── Iris_Assign_2.ipynb
│   └── Assignment 2.pdf
├── assignment_3/
│   ├── Assignment_3 fashion mnist.ipynb
│   └── assignment-3.pdf
├── Assignment-4/
│   ├── Assignment_4_JENA_CLIMATE_DATASET.ipynb
│   └── Assignment-4.pdf
├── Assignment-5/
│   ├── RNN vs LSTM vs GRU.ipynb
│   ├── README_ASSIGN_5.md
│   └── Assignment-5.pdf
├── assignment_6/
│   ├── CNN_Assign_6_Maize_leaf_disease.ipynb
│   └── Assignment-6.pdf
├── Assignment-7/
│   ├── Assignment_7_CIFAR_10.ipynb
│   └── Assignment -7.pdf
└── Assignment-8/
    ├── BERT.ipynb
    ├── bert_sentiment_analysis.py
    ├── requirements.txt
    ├── README.md
    └── Assignment-8.pdf
```
