# RNN vs LSTM vs GRU for Sequence Classification

**Name:** Raj Kanade | **Roll No:** 26 | **PRN:** 12413760 | **Batch:** B3

## 1. Assignment Title

**Implement and Compare RNN, LSTM, and GRU Models for Sequence
Classification**

## 2. Problem Statement

Implement and compare **Simple RNN, LSTM, and GRU** models for sequence
classification. Evaluate their performance using suitable metrics such
as accuracy, precision, recall, F1-score, training time, and number of
parameters.

## 3. Objective

The main objectives of this assignment are:

-   To understand sequence classification using recurrent neural
    networks.
-   To implement a Simple RNN model.
-   To implement an LSTM model.
-   To implement a GRU model.
-   To compare the performance of the three architectures.
-   To analyze accuracy, precision, recall, F1-score, training time, and
    model parameters.

## 4. Dataset

The **Reuters Newswire dataset** provided by Keras is used.

The dataset contains news articles represented as sequences of word
indices and belongs to **46 different topic classes**.

Dataset information:

-   Training samples: 8,982
-   Testing samples: 2,246
-   Number of classes: 46
-   Maximum sequence length used: 300
-   Vocabulary size used: 10,000

The sequences are padded/truncated to a fixed length before being given
to the neural networks.

## 5. Technologies Used

-   Python
-   TensorFlow
-   Keras
-   NumPy
-   Pandas
-   Matplotlib
-   Seaborn
-   Scikit-learn

## 6. Models Used

### Simple RNN

A Simple RNN processes sequence elements one at a time and maintains a
hidden state that carries information from previous time steps.

**Advantages:** - Simple architecture - Fewer parameters - Faster
training than LSTM/GRU in this experiment

**Limitation:** - Can have difficulty learning long-term dependencies.

### LSTM

Long Short-Term Memory networks use memory cells and gates to retain
useful information for longer sequences.

The main gates are:

-   Forget gate
-   Input gate
-   Output gate

LSTM is designed to reduce the vanishing-gradient problem found in
conventional RNNs.

### GRU

Gated Recurrent Unit is a simplified gated recurrent architecture.

It mainly uses:

-   Update gate
-   Reset gate

GRU generally has fewer parameters than LSTM while still being able to
learn long-term dependencies.

## 7. Model Architecture

The models use the following general architecture:

``` text
Reuters Input Sequence
        ↓
Padding / Truncation
        ↓
Embedding Layer
        ↓
RNN / LSTM / GRU
        ↓
Dropout
        ↓
Dense Layer (64 neurons)
        ↓
Dropout
        ↓
Softmax Output Layer
        ↓
46 Topic Classes
```

### Common configuration

-   Vocabulary size: 10,000
-   Embedding dimension: 128
-   Recurrent units: 128
-   Dense layer: 64 neurons
-   Output classes: 46
-   Optimizer: Adam
-   Learning rate: 0.001
-   Loss function: Sparse Categorical Crossentropy
-   Batch size: 128
-   Maximum epochs: 10
-   Early stopping is used to avoid unnecessary training.

The embedding layer uses `mask_zero=True` so that padded positions can
be ignored by the recurrent layers.

## 8. Data Preprocessing

The following preprocessing steps are performed:

1.  Load the Reuters dataset.
2.  Restrict the vocabulary to the 10,000 most frequent words.
3.  Examine sequence-length statistics.
4.  Pad or truncate sequences to 300 tokens.
5.  Prepare training and testing data.
6.  Train the three recurrent models using the same general dataset and
    training setup.

## 9. Training

Each model is trained using:

``` text
Epochs       : 10 maximum
Batch size   : 128
Optimizer    : Adam
Loss         : Sparse Categorical Crossentropy
Validation   : 20% of training data
Early Stop   : Enabled
```

Early stopping monitors validation loss and restores the best model
weights.

## 10. Evaluation Metrics

### Accuracy

Measures the proportion of correctly classified samples.

``` text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Measures how many samples predicted as a particular class are actually
correct.

### Recall

Measures how many samples belonging to a class are correctly identified.

### F1 Score

The F1 score is the harmonic mean of precision and recall.

``` text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### Training Time

The time required to train each model is recorded and compared.

### Number of Parameters

The total number of trainable model parameters is recorded to compare
model complexity.

## 11. Visualizations

The program generates:

-   Reuters sequence-length distribution
-   Reuters class distribution
-   Training vs validation accuracy
-   Training vs validation loss
-   Accuracy comparison
-   Precision comparison
-   Recall comparison
-   F1-score comparison
-   Training-time comparison
-   Parameter-count comparison
-   Combined performance comparison

**Confusion matrix visualization is intentionally not included.**


## 12. How to Run

### Step 1: Install required libraries

``` bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

### Step 2: Open the notebook

Open the Python/Google Colab notebook containing the assignment code.

### Step 3: Run the cells

Run the cells in order:

``` text
1. Import libraries
2. Set parameters
3. Load Reuters dataset
4. Analyze dataset
5. Preprocess sequences
6. Build models
7. Train RNN
8. Train LSTM
9. Train GRU
10. Evaluate models
11. Generate comparison graphs
```

### Step 4: Analyze the results

Compare the three models based on:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Training time
-   Number of parameters

## 13. Result Analysis

The earlier experiment showed that all three models were able to learn
the Reuters classification task. In the previous run, the test
accuracies were approximately:

-   RNN: 50.76%
-   LSTM: 55.08%
-   GRU: 56.14%

These values are from an earlier run and may change when the corrected
configuration is executed because neural-network training can vary
between runs.

The earlier experiment also showed that the recurrent models performed
much better on classes with more training examples than on very small
classes. Therefore, accuracy should be interpreted together with
precision, recall, and F1-score.

## 14. Conclusion

This assignment demonstrates the use of recurrent neural networks for
multi-class sequence classification using the Reuters dataset.

Simple RNN, LSTM, and GRU models are implemented using the same general
preprocessing and evaluation procedure. Their performance is compared
using multiple metrics rather than accuracy alone.

The experiment helps demonstrate the differences between conventional
RNNs and gated recurrent architectures such as LSTM and GRU, as well as
the trade-offs between model performance, complexity, and training time.

## 15. References

-   TensorFlow/Keras Reuters dataset
-   TensorFlow/Keras documentation
-   Scikit-learn metrics documentation
