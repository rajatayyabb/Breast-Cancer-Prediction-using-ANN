import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import load_model

# Force CPU usage
tf.config.set_visible_devices([], 'GPU')

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF1744;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1976D2;
        margin-top: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .benign {
        background-color: #C8E6C9;
        color: #1B5E20;
    }
    .malignant {
        background-color: #FFCDD2;
        color: #B71C1C;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_models():
    try:
        # Check if files exist
        if not os.path.exists('breast_cancer_ann_model.keras'):
            st.error("❌ Model file 'breast_cancer_ann_model.keras' not found!")
            st.info("Please run 'python train_model.py' first to train the model.")
            return None, None
        
        if not os.path.exists('scaler.pkl'):
            st.error("❌ Scaler file 'scaler.pkl' not found!")
            st.info("Please run 'python train_model.py' first to train the model.")
            return None, None
        
        # Load model
        model = load_model('breast_cancer_ann_model.keras')
        
        # Load scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        return model, scaler
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info("Please ensure you have run 'python train_model.py' to generate the model files.")
        return None, None

# Title
st.markdown('<h1 class="main-header">🏥 Breast Cancer Prediction System</h1>', unsafe_allow_html=True)
st.markdown("### Powered by Artificial Neural Networks")

# Sidebar
st.sidebar.header("📊 Model Information")
st.sidebar.info("""
**Model Architecture:**
- Input Layer: 30 features
- Hidden Layer 1: 16 neurons (ReLU)
- Hidden Layer 2: 8 neurons (ReLU)
- Output Layer: 1 neuron (Sigmoid)

**Hyperparameters:**
- Learning Rate: 0.01
- Batch Size: 16
- Epochs: 50
- Loss Function: MSE
- Optimizer: SGD
""")

# Load models
model, scaler = load_models()

if model is None or scaler is None:
    st.error("⚠️ Models not loaded! Please follow these steps:")
    st.markdown