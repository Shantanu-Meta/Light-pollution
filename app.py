import streamlit as st
from streamlit_option_menu import option_menu
from dotenv import load_dotenv
load_dotenv()

# 🔧 Set page config (title + favicon)
st.set_page_config(
    page_title="Light Pollution Analyzer",   # Title in browser tab
    page_icon="💡",                          # Favicon - emoji or image like 'favicon.png'
    layout="wide",                          # 'centered' or 'wide'
)

# Custom Home page screen
def show_home():
    st.title("Welcome to Light Pollution Analyzer")
    st.markdown("""
    Helps analyze and forecast light pollution data of Indian states using Machine Learning models.

    ### 🔍 Features:
    - **📈 Forecast (state wise):** Predict light pollution levels using the **LSTM model**
    - **🔍 Find Key Factor:** Analyze **key factors** (SOL values – Urban & Agriculture) that impact light pollution
    - **🧩 Clusters (state wise):** Group states using the **KMeans model** into low, moderate, and high light pollution zones
    - **💾 Finding Outliers:** Detect abnormal light pollution trends using the **DBSCAN model**
    - **💬 Ask Bot:** Chatbot to answer queries about light pollution data
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

def run_dbscan():
    from dbscan_model import run_dbscan_model
    run_dbscan_model()

# -------------------- Streamlit Layout --------------------

# Sidebar Menu
with st.sidebar:
    selected = option_menu(
        "Light Pollution Analyzer",  # App name as header
        ["🏠 Home", "📈 Forecast (state wise)", "🔍 Key Factor", "🧩 Clusters(state wise)", "💾 DB-scan", "💬 Ask Bot"],
        icons=["house", "bar-chart", "graph-up", "diagram-3", "database", "robot"],
        menu_icon="cloud-sun",
        default_index=0,
    )

# Page Routing
if selected == "🏠 Home":
    show_home()

elif selected == "📈 Forecast (state wise)":
    st.header("📈 Forecast Light Pollution for a State in India")
    target_state = st.text_input("Enter state name (e.g., Delhi)")
    if target_state:
        if st.button("Start Forecasting"):
            run_forecast(target_state)

elif selected == "🔍 Key Factor":
    st.header("🔍 Analyze Key Pollution Factors")
    target_state = st.text_input("Enter state name (e.g., Goa)")
    if target_state:
        if st.button("Find key factor"):
            run_key_factor(target_state)

elif selected == "🧩 Clusters(state wise)":
    st.header("🧩 Clustering Indian States by Light Pollution Intensity")
    if st.button("Run Model"):
        run_clusters()

elif selected == "💾 DB-scan":
    st.header("💾 DB-scan to find outliers of Indian States by Light Pollution Intensity")
    if st.button("Run Model"):
        run_dbscan()

elif selected == "💬 Ask Bot":
    from doc_chat import run_doc_chat
    run_doc_chat()
