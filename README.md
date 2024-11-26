Breast Cancer Prediction Application
This project is a Breast Cancer Prediction application developed using Python. It utilizes machine learning models to classify whether a breast tumor is malignant or benign based on input features derived from medical data.

Key Features:
Machine Learning Models: Implements multiple ML algorithms, including:
Logistic Regression
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
K-Means Clustering (unsupervised)


Data Preprocessing:
Cleans and prepares the dataset by handling missing values and mapping categorical labels.
Standardizes features using StandardScaler for improved model performance.

User-Friendly Interface:
A graphical user interface (GUI) built with Tkinter to simplify interactions.
Features buttons for step-by-step execution: Load Data, Clean Data, Train Models, and Make Predictions.
Radio buttons to select the desired prediction algorithm.
Prediction Functionality: Allows users to input 30 features and predict tumor classification using the chosen algorithm.

Tools and Libraries Used:
Data Analysis and Preprocessing:
pandas for data manipulation and cleaning.
numpy for numerical operations.

Machine Learning:
scikit-learn for implementing models and data preprocessing.

GUI Development:
Tkinter for creating an interactive interface.
Metrics:
accuracy_score for evaluating model performance.
