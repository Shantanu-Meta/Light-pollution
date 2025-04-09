import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# -------------------- Load & Preprocess Data --------------------
@st.cache_data
def load_and_preprocess_data():
    file_path = "All.csv"
    df = pd.read_csv(file_path, encoding="ISO-8859-1")

    # Convert relevant columns to numeric (removing commas)
    numeric_columns = ["Urban_SOL", "Agriculture_SOL", "Urban_SOL_Change", "Area (km²)", "GSDP(INR billions)"]
    for col in numeric_columns:
        df[col] = df[col].astype(str).str.replace(",", "").astype(float)

    # Drop missing GSDP values
    df.dropna(subset=["GSDP(INR billions)"], inplace=True)

    # Keep only the latest entry for each state
    df_sorted = df.sort_values(by=["state_name", "system:time_start"], ascending=[True, False])
    df_latest = df_sorted.drop_duplicates(subset=["state_name"], keep="first")

    return df_latest

# -------------------- KMeans Clustering --------------------
def run_kmeans_model():
    st.subheader("🧩 K-Means Clustering of Indian States by Light Pollution")

    df = load_and_preprocess_data()

    # Features for clustering
    features = ["Urban_SOL", "Agriculture_SOL", "GSDP(INR billions)"]
    X = df[features]

    # Normalize data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow Method for Optimal K
    inertia = []
    k_values = range(1, 11)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    # Plot Elbow Method
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_values, inertia, marker="o", linestyle="-", color="b")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method for Optimal K")
    ax.grid()
    st.pyplot(fig)

    # Apply KMeans with optimal K (assume K=4)
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    # Sort clusters by light pollution level
    cluster_means = df.groupby("Cluster")["Urban_SOL"].mean().sort_values()
    cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(cluster_means.index)}
    df["Cluster"] = df["Cluster"].map(cluster_mapping)

    # Cluster Labels (0 = Low Pollution, 3 = High Pollution)
    cluster_labels = {
        0: "🌿 Low Light Pollution",
        1: "🌅 Moderate Light Pollution",
        2: "🌆 High Light Pollution",
        3: "🌃 Very High Light Pollution"
    }

    # Show cluster summary
    cluster_summary = df.groupby("Cluster")[features].mean()
    st.subheader("📊 Cluster Summary (Mean Values)")
    st.dataframe(cluster_summary.style.format("{:.2f}"))
    st.caption("In nw/m²/sr, the higher the value, the more light pollution.")

    # Show cluster count distribution
    cluster_counts = df["Cluster"].value_counts().sort_index()

    # Plot cluster distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="viridis", ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of States")
    ax.set_title("Number of States in Each Cluster")
    st.pyplot(fig)

    # Scatter plot to visualize clusters
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(df["Urban_SOL"], df["GSDP(INR billions)"], c=df["Cluster"], cmap="viridis", edgecolors="k")
    ax.set_xlabel("Urban SOL")
    ax.set_ylabel("GSDP (INR Billions)")
    ax.set_title("Cluster Visualization (Urban SOL vs GSDP)")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    st.pyplot(fig)

    # -------------------- Model Performance Metrics --------------------
    silhouette_avg = silhouette_score(X_scaled, df["Cluster"])


    st.subheader("📈 Model Performance & Evaluation")

    # Silhouette Score
    st.markdown(f"**🔹 Silhouette Score:** `{silhouette_avg:.3f}` ")
    st.caption("Higher values indicate better-defined clusters.")
    if silhouette_avg > 0.5:
        st.success("Good clustering performance!")
    elif silhouette_avg > 0.2:
        st.warning("moderate to good clustering performance!")
    else:
        st.error("Poor clustering, data separation might not be effective.")

    # -------------------- Cluster Interpretation --------------------
    st.subheader("🌎 Cluster Interpretation")
    for cluster, label in cluster_labels.items():
        st.markdown(f"**Cluster {cluster}: {label}**")

    # Display clustered states
    st.subheader("🗺️ State-wise Clusters")
    df_sorted = df[["state_name", "Cluster"]].sort_values(by="Cluster")
    st.dataframe(df_sorted)

