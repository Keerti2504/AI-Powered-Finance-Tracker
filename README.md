# AI-Powered-Finance-Tracker
Smart, secure, and self-learning — your money’s new best friend.

# Transformer-Based Diffusion Anomaly Detection for Personal Finance

A hybrid anomaly detection pipeline combining **Graph Neural Networks**, **Transformer Encoders**, and **Diffusion Models** to flag abnormal financial transactions from tabular time-series data. Built with **PyTorch**, **PyTorch Geometric**, and **scikit-learn**, wrapped in an interactive **Streamlit app** for seamless anomaly detection on your own data.

---
1. **Clone the repo:**

```bash
git clone https://github.com/Keerti2504/AI-Powered-Finance-Tracker.git
cd AI-Powered-Finance-Tracker
```
## 📦 Installation

```bash
# Install PyTorch Geometric dependencies (CPU version)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# Install PyTorch Geometric core
pip install torch-geometric

# Other dependencies
pip install streamlit pandas scikit-learn matplotlib seaborn

#run the app
streamlit run app.py

#Open the URL shown in your terminal (usually http://localhost:8501) to interact with the anomaly detection dashboard.
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



|                | Predicted Normal | Predicted Anomaly |
|----------------|------------------|-------------------|
| Actual Normal  | 811              | 13                |
| Actual Anomaly | 0                | 30                |

---

## 📝 Output

- Results stored in: `T-Diff_anomaly_results.csv`
- Sample output:


| Date       | Category | Expense | Anomaly Score  |
|------------|----------|---------|----------------|
| 2022-01-12 | Travel   | 0.976   | 0.1781         |
| 2023-04-08 | Shopping | 0.842   | 0.1935         |


---

## 🚧 Future Enhancements

- Add **temporal GNNs** or **time-aware edges**
- Use **GAT (Graph Attention Network)** instead of GCN
- Visualize transaction graphs using **PyVis** or **NetworkX**
- Extend to real-time detection pipelines
- Adaptive thresholding for real-time anomaly sensitivity
- Integrate Explainable AI (SHAP) to justify flagged anomalies

## Output 
<img width="994" height="822" alt="image" src="https://github.com/user-attachments/assets/979ffd69-d9b3-4365-abc0-c7a621145f43" />

<img width="922" height="773" alt="image" src="https://github.com/user-attachments/assets/c16376f0-9a95-4e3a-8477-806572ecbb45" />










