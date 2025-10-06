# Multimodal Emotion Analysis with Gated Cross-Modal Attention

## Overview
This project presents a predictive model designed for **Multimodal Emotion Analysis (MEA)**, classifying human emotions by effectively combining information from three distinct modalities: **Textual, Visual, and Acoustic**.

Leveraging advanced deep learning techniques, the core innovation lies in a novel **Gated Cross-Modal Attention Fusion** architecture that ensures robust and context-aware feature integration, leading to high-accuracy emotion classification.

The model was developed and rigorously tested on the well-established **CMU-MOSI (Multimodal Opinion Sentiment and Emotion) dataset**.

## 🚀 Key Features and Innovations

* **Multimodal Fusion:** Successfully processes and aligns three asynchronous modalities (Text, Visual, Acoustic) to build a holistic representation of human emotion.
* **Gated Fusion Mechanism:** Implements a proprietary **Modality Gated Fusion** layer that uses visual features to selectively amplify or suppress corresponding textual features, preventing noise and maximizing inter-modal synergy.
* **Attention-Driven Classification:** Utilizes a **Self-Attention** mechanism on the fused features to assign importance scores dynamically, ensuring the final classification decision is based on the most salient segments of the conversational sequence.
* **Performance:** Achieved strong classification performance with a **70.41% Accuracy** and **72.31% F1 score** on the CMU-MOSI dataset.
* **Code Quality:** Custom **PyTorch** implementations for dataset loading, sequence padding (`multi_collate`), and model architecture, emphasizing performance tuning and code clarity.

## 🧠 Model Architecture (GatedCrossModalAttentionFusion)

The model follows a structured approach to feature processing and fusion:

1.  **Unimodal Feature Extraction:**
    * Separate Bi-directional LSTM (Bi-LSTM) layers are used to extract temporal dependencies from the Text (using GloVe embeddings) and Visual (using FACET features) modalities.
    * Includes Batch Normalization and Dropout for regularization.
2.  **Modality Gating Fusion:**
    * The **ModalityGatedFusion** layer projects the Visual LSTM output to the Text feature dimension.
    * This projection is passed through a Sigmoid gate, which then element-wise multiplies the Text LSTM output, allowing visual context to "gate" (modulate) the textual understanding.
3.  **Self-Attention:**
    * A Multihead Attention layer is applied to the resultant fused sequence to capture global relationships and aggregate sequence-level context before final prediction.
4.  **Classification:**
    * The sequence is reduced to a single vector via sum-pooling and passed through two fully connected (FC) layers with ReLU activation and Dropout for final emotion classification (regression with `nn.MSELoss`).

## 🛠️ Technical Implementation Details

| Component | Technology / Technique | Purpose |
| :--- | :--- | :--- |
| **Framework** | PyTorch, NumPy | Deep learning implementation and numerical operations. |
| **Dataset** | CMU-MOSI (Multimodal Opinion Sentiment and Emotion) | Standard dataset for multimodal analysis. |
| **Text Features** | GloVe Embeddings | Word-level representation of textual data. |
| **Fusion Logic** | Modality Gated Fusion | Selective, context-aware combination of visual and text features. |
| **Optimization** | Adam Optimizer, nn.MSELoss | Training and minimizing regression loss for emotion classification. |
| **Feature Extraction** | Bi-directional LSTM | Capturing sequential dependencies in text and visual data. |
| **Alignment** | CMU-MultimodalSDK | Used for word-level alignment and standardization of high-frequency features. |

## ⚙️ Setup and Usage

### Prerequisites

* Python (3.7+)
* PyTorch
* NumPy, Pandas, Scikit-learn
* CMU-MultimodalSDK (for dataset handling)

