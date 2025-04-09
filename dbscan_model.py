import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# -------------------- Load & Preprocess Data --------------------
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("All.csv", encoding="ISO-8859-1")
    
    df.rename(columns={
        "Agriculture_SOL": "agricultural_SOL",
        "Urban_SOL": "urban_SOL",
        "Urban_SOL_Change": "changeOfSOL",
        "Area (km²)": "area_km2",
        "GSDP(INR billions)": "gdp"
    }, inplace=True)

    numeric_cols = ["agricultural_SOL", "urban_SOL", "changeOfSOL", "area_km2", "gdp"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")

    df.dropna(inplace=True)

    if "state_name" in df.columns:
        df = df.groupby("state_name", as_index=False)[numeric_cols].mean()

    return df


# -------------------- DBSCAN Clustering --------------------
def run_dbscan_model():
    st.subheader("🔍 DBSCAN Clustering on Light Pollution intensity")

    df = load_and_preprocess_data()

    features = ["agricultural_SOL", "urban_SOL", "area_km2"]
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # -------------------- Smart DBSCAN Tuning --------------------
    found_valid_clusters = False
    max_loops = 20
    eps_value = 0.5
    min_samples_value = 4
    loop_count = 0

    while not found_valid_clusters and loop_count < max_loops:
        dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
        labels = dbscan.fit_predict(X_pca)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters >= 4:
            found_valid_clusters = True
        else:
            eps_value *= 1.05
            min_samples_value = max(min_samples_value - 1, 2)
            loop_count += 1

    df["Cluster"] = labels

    # -------------------- Re-map clusters 0-3 only --------------------
    valid_clusters = df[df["Cluster"] != -1]
    top_clusters = valid_clusters.groupby("Cluster")["urban_SOL"].mean().sort_values().head(4).index.tolist()
    cluster_remap = {old: new for new, old in enumerate(top_clusters)}
    df["Cluster"] = df["Cluster"].apply(lambda x: cluster_remap.get(x, -1))

    # -------------------- Interpretation --------------------
    cluster_labels = {
        -1: "🚨 Outliers (Noise Points)",
         0: "🌿 Low Light Pollution (Mostly Agricultural States)",
         1: "🌅 Moderate Light Pollution (Balanced Economy)",
         2: "🌆 High Light Pollution (Urbanizing States)",
         3: "🌃 Very High Light Pollution (Highly Urbanized States)"
    }

    df["Cluster Label"] = df["Cluster"].map(cluster_labels)

    # -------------------- Display Summaries --------------------
    st.subheader("📊 Cluster Summary")
    st.dataframe(df.groupby("Cluster")[features].mean().style.format("{:.2f}"))

    st.write("🔢 Cluster Distribution")
    st.dataframe(df["Cluster"].value_counts())

    # -------------------- Plots --------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=df["Cluster"].value_counts().index, y=df["Cluster"].value_counts().values, palette="viridis", ax=ax)
    ax.set_title("Number of States in Each Cluster (DBSCAN)")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of States")
    st.pyplot(fig)

    # Scatter plot
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    scatter = ax2.scatter(df["urban_SOL"], df["gdp"], c=df["Cluster"], cmap="viridis", edgecolors="k", s=100)
    ax2.set_xlabel("Urban SOL")
    ax2.set_ylabel("GSDP (INR Billions)")
    ax2.set_title("DBSCAN Cluster Visualization (Urban SOL vs GSDP)")
    plt.colorbar(scatter, ax=ax2, label="Cluster")
    st.pyplot(fig2)

    # -------------------- Silhouette Score --------------------
    st.subheader("📈 Model Performance")
    valid_for_silhouette = df[df["Cluster"] != -1]
    if len(valid_for_silhouette["Cluster"].unique()) > 1:
        sil_score = silhouette_score(X_pca[valid_for_silhouette.index], valid_for_silhouette["Cluster"])
        st.markdown(f"**Silhouette Score:** `{sil_score:.3f}`")
        st.caption("A higher Silhouette Score(>0.5) indicates better-defined clusters.")
    else:
        st.warning("⚠️ Not enough clusters for Silhouette Score.")

    # -------------------- Final Info --------------------
    st.subheader("🌎 Cluster Interpretation")
    for cid, label in cluster_labels.items():
        st.markdown(f"**Cluster {cid}: {label}**")

    st.subheader("🗺️ State-wise Clusters")
    st.dataframe(df[["state_name", "Cluster"]].sort_values(by="Cluster"))
