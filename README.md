# ❤️ Heart Disease Prediction System
A comprehensive machine learning pipeline for predicting heart disease risk using clinical parameters from the UCI Heart Disease dataset.


## 📋 Project Overview
This project implements a complete machine learning pipeline to analyze and predict heart disease risk. The workflow includes data preprocessing, feature selection, dimensionality reduction, supervised/unsupervised learning, hyperparameter tuning, and web deployment.
## 🎯 Objectives
- **Data Preprocessing & Cleaning**: Handle missing values, encoding, and feature scaling
- **Dimensionality Reduction**: Apply PCA to retain essential features
- **Feature Selection**: Use statistical methods and ML-based techniques
- **Supervised Learning**: Train classification models (Logistic Regression, Decision Trees, Random Forest, SVM)
- **Unsupervised Learning**: Apply K-Means and Hierarchical Clustering
- **Model Optimization**: Hyperparameter tuning with GridSearchCV and RandomizedSearchCV
- **Deployment**: Build and deploy a Streamlit UI with Ngrok for public access

## 📊 Dataset Information
The project uses the Heart Disease UCI Dataset containing 13 clinical features:

- `age`: age in years

- `sex`: sex (1 = male; 0 = female)

- `cp`: chest pain type
        -- Value 1: typical angina
        -- Value 2: atypical angina
        -- Value 3: non-anginal pain
        -- Value 4: asymptomatic

- `trestbps`: resting blood pressure (in mm Hg on admission to the hospital)

- `chol`: serum cholestoral in mg/dl

- `fbs`: (fasting blood sugar > 120 mg/dl)  (1 = true; 0 = false)

- `restecg`: resting electrocardiographic results
        -- Value 0: normal
        -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
        -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria

- `thalach`: maximum heart rate achieved

- `exang`: exercise induced angina (1 = yes; 0 = no)

- `oldpeak` = ST depression induced by exercise relative to rest

- `slope`: the slope of the peak exercise ST segment
        -- Value 1: upsloping
        -- Value 2: flat
        -- Value 3: downsloping

- `ca`: number of major vessels (0-3) colored by flourosopy

- `thal`: 3 = normal; 6 = fixed defect; 7 = reversable defect
- `target`: diagnosis of heart disease (angiographic disease status)
        -- Value 0: < 50% diameter narrowing
        -- Value 1: > 50% diameter narrowing

## 📊 Model Performance
- chose **Random Forest** as the best model since it has a good recall-precision balance

| Model               | CV Score | Test Accuracy | Test Precision | Test Recall | Test F1  | Test AUC |
|----------------------|----------|---------------|----------------|-------------|----------|----------|
| Logistic Regression  | 0.853067 | 0.672131      | 0.583333       | 1.000000    | 0.736842 | 0.951299 |
| Random Forest        | 0.793257 | 0.918033      | 0.896552       | 0.928571    | 0.912281 | 0.953463 |
| SVM                  | 0.802667 | 0.868852      | 0.812500       | 0.928571    | 0.866667 | 0.950216 |


## 🏗️ Project Structure
```
Heart_Disease_Project/
│── data/
│   ├── heart_disease.csv
│── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│── models/
│   ├── final_model.pkl
│── ui/
│   ├── app.py (Streamlit UI)
│── deployment/
│   ├── ngrok_setup.txt
│── results/
│   ├── evaluation_metrics.txt
│── README.md
│── requirements.txt
│── .gitignore
```
## 🛠️ Installation & Setup
1. Clone the Repository
```
git clone https://github.com/omniah7/Heart_Disease_Project.git
cd Heart_Disease_Project
```
2. Create Virtual Environment
```
python -m venv heart_env
source heart_env/bin/activate  # On Windows: heart_env\Scripts\activate
```
3. Install Dependencies
```
pip install -r requirements.txt
```
4. Run the Streamlit app locally
```
streamlit run ui/app.py
```
Access the application
Open your browser and go to: http://localhost:8501


## Author
- Omniah Arafah