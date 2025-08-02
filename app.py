import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch_geometric.data import Data

# --- Transformer Encoder Model ---
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, ff_dim=128):
        super(TransformerEncoder, self).__init__()
        # Choose number of heads (max divisor of input_dim <= 8)
        num_heads = max([h for h in range(1, input_dim + 1) if input_dim % h == 0 and h <= 8])
        self.self_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.ff = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, input_dim)
        )

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.ff(x))
        return x

# --- Diffusion Model ---
class DiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DiffusionModel, self).__init__()
        self.denoiser = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, noise_level=0.1):
        noise = torch.randn_like(x) * noise_level
        return self.denoiser(x + noise)

# Preprocess dataframe to numerical features & scale
def preprocess(df):
    df = df.copy()
    df.fillna(0, inplace=True)

    label_encoder = LabelEncoder()
    if "Payment Method" in df.columns:
        df["Payment Method"] = label_encoder.fit_transform(df["Payment Method"])
    if "DayOfWeek" in df.columns:
        df["DayOfWeek"] = label_encoder.fit_transform(df["DayOfWeek"])
    if "Transaction Type" in df.columns:
        df["Transaction Type"] = df["Transaction Type"].map({"Income": 1, "Expense": 0})

    # Create Amount column (Income + Expense)
    df["Income"] = df.get("Income", 0)
    df["Expense"] = df.get("Expense", 0)
    df["Amount"] = df["Income"] + df["Expense"]
    df["Balance"] = df.get("Balance", 0)

    scaler = MinMaxScaler()
    features_to_scale = ["Income", "Expense", "Amount", "Balance"]
    for col in features_to_scale:
        if col not in df.columns:
            df[col] = 0
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    # Select features for node features
    feature_cols = ["Income", "Expense", "Amount", "Transaction Type", "Payment Method", "DayOfWeek", "Balance"]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    node_features = torch.tensor(df[feature_cols].values, dtype=torch.float)
    return df, node_features

# Create edges based on matching 'Category' or 'Payment Method'
def create_graph_edges(df):
    edges = []
    cat_available = "Category" in df.columns
    pm_available = "Payment Method" in df.columns

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            same_cat = cat_available and df.iloc[i]["Category"] == df.iloc[j]["Category"]
            same_pm = pm_available and df.iloc[i]["Payment Method"] == df.iloc[j]["Payment Method"]
            if same_cat or same_pm:
                edges.append((i, j))
                edges.append((j, i))
    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

# Training and anomaly detection
def anomaly_detection(df, node_features, edge_index):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer = TransformerEncoder(input_dim=node_features.shape[1]).to(device)
    diffusion = DiffusionModel(input_dim=node_features.shape[1], hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(list(transformer.parameters()) + list(diffusion.parameters()), lr=0.001)

    data = Data(x=node_features.to(device), edge_index=edge_index.to(device))

    def train():
        transformer.train()
        diffusion.train()
        optimizer.zero_grad()
        embeddings = transformer(data.x.unsqueeze(0)).squeeze(0)
        reconstructed = diffusion(embeddings)
        loss = F.mse_loss(reconstructed, embeddings)
        loss.backward()
        optimizer.step()
        return loss.item()

    best_loss = float("inf")
    patience, counter = 10, 0
    for epoch in range(100):
        loss = train()
        if loss < best_loss:
            best_loss = loss
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            break

    transformer.eval()
    diffusion.eval()
    with torch.no_grad():
        embeddings = transformer(data.x.unsqueeze(0)).squeeze(0)
        reconstructed = diffusion(embeddings)
        anomaly_scores = torch.mean((embeddings - reconstructed) ** 2, dim=1).cpu().numpy()

    threshold = np.percentile(anomaly_scores, 95)
    df["Anomaly Score"] = anomaly_scores
    df["Anomaly"] = df["Anomaly Score"] > threshold
    return df

# --- Streamlit UI ---
def main():
    st.title("🚨 T-Diff Anomaly Detection")

    uploaded_file = st.file_uploader("Upload transaction CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df.head())

        if st.button("Run Anomaly Detection"):
            with st.spinner("Processing and training model... This may take a few minutes."):
                df_processed, node_features = preprocess(df)
                edge_index = create_graph_edges(df_processed)
                result_df = anomaly_detection(df_processed, node_features, edge_index)

            st.success("Anomaly Detection Completed!")
            anomalies = result_df[result_df["Anomaly"] == True]
            if anomalies.empty:
                st.info("No anomalies detected.")
            else:
                st.write("### Detected Anomalies")
                display_cols = ["Date", "Category", "Expense", "Anomaly Score"]
                available_cols = [col for col in display_cols if col in anomalies.columns]
                st.dataframe(anomalies[available_cols].sort_values("Anomaly Score", ascending=False))

            # Download button
            csv = result_df.to_csv(index=False).encode()
            st.download_button(
                label="Download full results CSV",
                data=csv,
                file_name="T-Diff_anomaly_results.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
