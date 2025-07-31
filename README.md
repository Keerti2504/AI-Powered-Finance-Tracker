# AI-Powered-Finance-Tracker
Smart, secure, and self-learning — your money’s new best friend.

# Transformer-Based Diffusion Anomaly Detection for Personal Finance

A hybrid anomaly detection pipeline that combines **Graph Neural Networks**, **Transformer Encoders**, and **Diffusion Models** to identify abnormal financial transactions from tabular time-series data. Built using **PyTorch**, **PyTorch Geometric**, and **scikit-learn**.

---

## 📦 Installation

```bash
# Install PyTorch Geometric dependencies (CPU version)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# Install PyTorch Geometric core
pip install torch-geometric

# Other dependencies
pip install pandas scikit-learn matplotlib seaborn
```

---

## 📂 Dataset

- **File:** `dataset_2years.csv`
- **Columns:**
  - `Date`, `Category`, `Payment Method`, `Transaction Type`, `Income`, `Expense`, `Balance`
- **Processing:**
  - Categorical features are label-encoded
  - `Amount = Income + Expense`
  - All numerical features normalized via MinMaxScaler

---

## 🔧 Model Architecture

### 1. **Graph Construction**
- Nodes: Individual transactions
- Edges: Based on shared `Category`, `Payment Method`, etc.

### 2. **Transformer Encoder**
- Learns contextual embeddings across graph nodes

### 3. **Diffusion Denoising**
- Injects Gaussian noise into embeddings
- Reconstructs clean embeddings
- Anomaly score = reconstruction error (MSE)

---

## 📈 Training

- Loss Function: **Mean Squared Error (MSE)**
- Optimizer: **Adam**
- Epochs: 100 (early stopping after 10 patience)
- Anomalies are detected by thresholding top 5% MSE scores

---

## 📊 Evaluation Results

```
✅ Precision:     0.6977
✅ Recall:        1.0000
✅ F1 Score:      0.8219
```

**Confusion Matrix:**

```
[[811  13]
 [  0  30]]
```

|        | Predicted Normal | Predicted Anomaly |
|--------|------------------|-------------------|
| Actual Normal  | 811              | 13                |
| Actual Anomaly | 0                | 30                |

---

## 📝 Output

- Results stored in: `T-Diff_anomaly_results.csv`
- Sample output:

```csv
| Date       | Category | Expense | Anomaly Score |
|------------|----------|---------|----------------|
| 2022-01-12 | Travel   | 0.976   | 0.1781         |
| 2023-04-08 | Shopping | 0.842   | 0.1935         |
```

---

## 🚧 Future Enhancements

- Add **temporal GNNs** or **time-aware edges**
- Use **GAT (Graph Attention Network)** instead of GCN
- Visualize transaction graphs using **PyVis** or **NetworkX**
- Extend to real-time detection pipelines
- Adaptive thresholding for real-time anomaly sensitivity
- Integrate Explainable AI (SHAP) to justify flagged anomalies







