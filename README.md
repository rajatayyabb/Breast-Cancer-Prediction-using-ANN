🏥 Breast Cancer Prediction using ANN

This project implements an Artificial Neural Network (ANN) for breast cancer classification using the Wisconsin Breast Cancer Dataset from Kaggle.

📋 Project Overview
The application predicts whether a breast tumor is Benign or Malignant based on 30 features extracted from digitized images of fine needle aspirate (FNA) of breast mass.

🎯 Model Specifications
Hyperparameters
Parameter	Value
Learning Rate	0.01
Batch Size	16
Epochs	50
Loss Function	Mean Squared Error
Optimizer	Stochastic Gradient Descent
Architecture
Input Layer: 30 features
Hidden Layer 1: 16 neurons (ReLU activation)
Hidden Layer 2: 8 neurons (ReLU activation)
Output Layer: 1 neuron (Sigmoid activation)
🚀 Getting Started
Prerequisites
Python 3.8 or higher
pip package manager
Installation
Clone the repository
bash
git clone https://github.com/yourusername/breast-cancer-ann.git
cd breast-cancer-ann
Install dependencies
bash
pip install -r requirements.txt
Download the dataset
Download the Breast Cancer Dataset from Kaggle
Place the breast-cancer.csv file in the project directory
Update the file path in train_model.py if needed
Training the Model
Run the training script:

bash
python train_model.py
This will:

Load and preprocess the dataset
Train the ANN model with specified hyperparameters
Save the trained model as breast_cancer_ann_model.keras
Save the scaler as scaler.pkl
Generate training history plots
Display evaluation metrics
Running the Streamlit App
After training, launch the web application:

bash
streamlit run app.py
The app will open in your default browser at http://localhost:8501

📊 Features
1. Single Prediction
Enter individual patient features manually
Get instant predictions with probability scores
Visual probability gauge
2. Batch Prediction
Upload CSV files for multiple predictions
Download results as CSV
View prediction distribution charts
3. About Section
Model information and architecture
Feature descriptions
Important disclaimers
📁 Project Structure
breast-cancer-ann/
│
├── train_model.py          # Model training script
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── breast_cancer_ann_model.h5   # Trained model (generated)
├── scaler.pkl                   # Feature scaler (generated)
├── training_history.png         # Training plots (generated)
└── confusion_matrix.png         # Confusion matrix (generated)
📈 Model Performance
After training, the model achieves:

High accuracy on test set
Good sensitivity and specificity
Reliable predictions for both classes
Exact metrics will be displayed after training

🎨 Screenshots
Single Prediction Interface
Show Image

Batch Prediction
Show Image

🌐 Deployment on GitHub
Step 1: Create Repository
bash
git init
git add .
git commit -m "Initial commit: Breast Cancer ANN with Streamlit"
git branch -M main
git remote add origin https://github.com/yourusername/breast-cancer-ann.git
git push -u origin main
Step 2: Deploy on Streamlit Cloud
Go to share.streamlit.io
Sign in with GitHub
Click "New app"
Select your repository
Set main file path to app.py
Click "Deploy"
🔬 Dataset Information
Source: Wisconsin Breast Cancer Dataset (Kaggle)

Features (30 total):

10 Mean values (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension)
10 Standard Error values
10 Worst/Largest values



