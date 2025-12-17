import streamlit as st
import numpy as np
import pandas as pd
import pickle
from keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px

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
        model = load_model('breast_cancer_ann_model.keras')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
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
    st.error("⚠️ Models not found! Please train the model first using train_model.py")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["🔬 Single Prediction", "📁 Batch Prediction", "ℹ️ About"])

# Tab 1: Single Prediction
with tab1:
    st.markdown('<p class="sub-header">Enter Patient Features</p>', unsafe_allow_html=True)
    
    # Feature names
    feature_names = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean',
        'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
        'fractal_dimension_se', 'radius_worst', 'texture_worst',
        'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave points_worst',
        'symmetry_worst', 'fractal_dimension_worst'
    ]
    
    # Default values (approximate means)
    default_values = [
        14.13, 19.29, 91.97, 654.89, 0.096, 0.104, 0.089, 0.048, 0.181, 0.063,
        0.405, 1.217, 2.866, 40.34, 0.007, 0.025, 0.032, 0.012, 0.021, 0.004,
        16.27, 25.68, 107.26, 880.58, 0.132, 0.254, 0.272, 0.114, 0.290, 0.084
    ]
    
    # Create input fields in columns
    col1, col2, col3 = st.columns(3)
    
    inputs = []
    for i, (feature, default) in enumerate(zip(feature_names, default_values)):
        if i % 3 == 0:
            with col1:
                value = st.number_input(f"{feature}", value=float(default), format="%.4f", key=f"input_{i}")
                inputs.append(value)
        elif i % 3 == 1:
            with col2:
                value = st.number_input(f"{feature}", value=float(default), format="%.4f", key=f"input_{i}")
                inputs.append(value)
        else:
            with col3:
                value = st.number_input(f"{feature}", value=float(default), format="%.4f", key=f"input_{i}")
                inputs.append(value)
    
    # Predict button
    if st.button("🔍 Predict", type="primary", use_container_width=True):
        # Prepare input
        input_array = np.array(inputs).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction_prob = model.predict(input_scaled, verbose=0)[0][0]
        prediction = int(prediction_prob > 0.5)
        
        # Display results
        st.markdown("---")
        st.markdown('<p class="sub-header">Prediction Results</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Prediction Probability", f"{prediction_prob * 100:.2f}%")
        
        with col2:
            if prediction == 0:
                st.markdown("""
                    <div class="result-box benign">
                        <h2>✅ BENIGN</h2>
                        <p>The tumor is likely non-cancerous</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="result-box malignant">
                        <h2>⚠️ MALIGNANT</h2>
                        <p>The tumor is likely cancerous</p>
                    </div>
                """, unsafe_allow_html=True)
        
        with col3:
            st.metric("Confidence", f"{max(prediction_prob, 1 - prediction_prob) * 100:.2f}%")
        
        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prediction_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Malignancy Probability"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if prediction == 1 else "darkgreen"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Batch Prediction
with tab2:
    st.markdown('<p class="sub-header">Upload CSV File for Batch Prediction</p>', unsafe_allow_html=True)
    
    st.info("📋 Upload a CSV file with the same 30 features as the training data (without the diagnosis column)")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("🚀 Run Batch Prediction", type="primary"):
                # Remove unnecessary columns if present
                if 'id' in df.columns:
                    df = df.drop('id', axis=1)
                if 'diagnosis' in df.columns:
                    true_labels = df['diagnosis'].map({'M': 1, 'B': 0})
                    df = df.drop('diagnosis', axis=1)
                    has_labels = True
                else:
                    has_labels = False
                
                # Scale and predict
                X_scaled = scaler.transform(df)
                predictions_prob = model.predict(X_scaled, verbose=0)
                predictions = (predictions_prob > 0.5).astype(int).flatten()
                
                # Create results dataframe
                results_df = df.copy()
                results_df['Prediction'] = ['Malignant' if p == 1 else 'Benign' for p in predictions]
                results_df['Probability'] = predictions_prob.flatten()
                
                if has_labels:
                    results_df['True_Label'] = ['Malignant' if p == 1 else 'Benign' for p in true_labels]
                    results_df['Correct'] = predictions == true_labels.values
                
                st.subheader("Prediction Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Samples", len(predictions))
                
                with col2:
                    benign_count = np.sum(predictions == 0)
                    st.metric("Benign Cases", benign_count)
                
                with col3:
                    malignant_count = np.sum(predictions == 1)
                    st.metric("Malignant Cases", malignant_count)
                
                if has_labels:
                    accuracy = np.mean(predictions == true_labels.values)
                    st.metric("Accuracy", f"{accuracy * 100:.2f}%")
                
                # Visualization
                fig = px.pie(
                    values=[benign_count, malignant_count],
                    names=['Benign', 'Malignant'],
                    title='Distribution of Predictions',
                    color_discrete_sequence=['#4CAF50', '#F44336']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="prediction_results.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Tab 3: About
with tab3:
    st.markdown('<p class="sub-header">About This Application</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Purpose
    This application uses an Artificial Neural Network (ANN) to predict whether a breast tumor is benign or malignant 
    based on various cellular features extracted from digitized images of fine needle aspirate (FNA) of breast mass.
    
    ### 🧠 Model Details
    - **Architecture**: 3-layer feedforward neural network
    - **Input Features**: 30 numerical features describing cell nuclei characteristics
    - **Output**: Binary classification (Benign/Malignant)
    - **Training Dataset**: Wisconsin Breast Cancer Dataset
    
    ### 📊 Features Used
    The model uses 30 features computed for each cell nucleus:
    
    **Mean Values (10 features):**
    - Radius, Texture, Perimeter, Area, Smoothness
    - Compactness, Concavity, Concave Points, Symmetry, Fractal Dimension
    
    **Standard Error (10 features):**
    - Same measurements as above
    
    **Worst/Largest Values (10 features):**
    - Same measurements as above
    
    ### ⚠️ Important Disclaimer
    This tool is for educational and research purposes only. It should NOT be used as a substitute for 
    professional medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.
    
    ### 👨‍💻 Developer Information
    - **Lab**: Lab 11 - Artificial Neural Networks
    - **Framework**: Keras with TensorFlow backend
    - **Frontend**: Streamlit
    - **Dataset**: Breast Cancer Wisconsin Dataset (Kaggle)
    
    ### 📚 References
    - Wolberg, W.H., Street, W.N., & Mangasarian, O.L. (1995). Breast Cancer Wisconsin (Diagnostic) Dataset
    - UCI Machine Learning Repository
    """)
    
    st.markdown("---")
    st.markdown("### 🔧 Hyperparameters Used")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Learning Rate**: 0.01
        - **Batch Size**: 16
        - **Epochs**: 50
        """)
    
    with col2:
        st.markdown("""
        - **Loss Function**: Mean Squared Error
        - **Optimizer**: Stochastic Gradient Descent
        - **Activation**: ReLU (hidden), Sigmoid (output)
        """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>© 2024 Breast Cancer Prediction System | Built with ❤️ using Streamlit & Keras</p>",
    unsafe_allow_html=True
)