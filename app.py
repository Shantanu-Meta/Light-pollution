import streamlit as st
from streamlit_option_menu import option_menu
import time

# Custom Home page screen
def show_home():
    st.title("👋 Hello, welcome to Light Pollution Analyzer")
    st.markdown("""
        This app helps analyze and forecast light pollution data using Machine Learning models.
        
        🔹 **Forecast (state)** – Predict light pollution levels using LSTM  
        🔹 **Key Factor** – Analyze mean SOL values (Urban & Agriculture)  
        🔹 **Clusters** – Group states using KMeans Clustering
    """)

# Load model functions
def run_forecast(target_state):
    from forecast_model import run_forecast_model
    run_forecast_model(target_state)

def run_key_factor(target_state):
    from keyfactor_model import run_keyfactor_model
    run_keyfactor_model(target_state)

def run_clusters():
    from cluster_model import run_kmeans_model
    run_kmeans_model()

# -------------------- Streamlit Layout --------------------

# Sidebar Menu
with st.sidebar:
    selected = option_menu(
        "Light Pollution Analyzer",  # App name as header
        ["🏠 Home", "📈 Forecast (state wise)", "🔍 Key Factor", "🧩 Clusters(state wise)"],
        icons=["house", "bar-chart", "graph-up", "diagram-3"],
        menu_icon="cloud-sun",
        default_index=0,
    )

# Page Routing
if selected == "🏠 Home":
    show_home()

elif selected == "📈 Forecast (state wise)":
    st.header("📈 Forecast Light Pollution for a State")

    target_state = st.text_input("Enter state name (e.g., West Bengal)")

    # Show the button AFTER user has typed something
    if target_state:
        if st.button("Start Forecasting"):
            run_forecast(target_state)

elif selected == "🔍 Key Factor":
    st.header("🔍 Analyze Key Pollution Factors")

    target_state = st.text_input("Enter state name (e.g., Goa)")

    # Show the button AFTER user has typed something
    if target_state:
        if st.button("Find key factor"):
            run_key_factor(target_state)

elif selected == "🧩 Clusters(state wise)":
    st.header("🧩 Clustering Indian States by Light Pollution Intensity")
    if st.button("Run Model"):
        run_clusters()
