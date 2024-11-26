import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import tkinter as tk
from tkinter import messagebox

# Initialize global variables
data = None
x_train, x_test, y_train, y_test, scaler = None, None, None, None, None
accuracy_scores = {}

# Load Data Function
def load_data():
    global data
    try:
        data = pd.read_csv('data.csv')
        messagebox.showinfo("Success", "Data loaded successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load data: {str(e)}")

# Data Cleaning Function
def clean_data():
    global data, x_train, x_test, y_train, y_test, scaler
    try:
        if data is None:
            messagebox.showerror("Error", "No data loaded. Load the data first.")
            return
        # Data cleaning
        data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
        data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
        # Splitting dataset
        x = data.drop(columns='diagnosis', axis=1)
        y = data['diagnosis']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
        # Standardizing the features
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        messagebox.showinfo("Success", "Data cleaned and processed successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to clean data: {str(e)}")

# Train Models and Compute Accuracy
def train_models():
    global accuracy_scores
    try:
        if x_train is None or x_test is None:
            messagebox.showerror("Error", "Data not prepared. Clean the data first.")
            return
        
        # Logistic Regression
        logistic_model = LogisticRegression(max_iter=10000)
        logistic_model.fit(x_train, y_train)
        logistic_pred = logistic_model.predict(x_test)
        accuracy_scores['Logistic Regression'] = accuracy_score(y_test, logistic_pred)

        # SVM
        svm_model = SVC()
        svm_model.fit(x_train, y_train)
        svm_pred = svm_model.predict(x_test)
        accuracy_scores['SVM'] = accuracy_score(y_test, svm_pred)

        # KNN
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(x_train, y_train)
        knn_pred = knn_model.predict(x_test)
        accuracy_scores['KNN'] = accuracy_score(y_test, knn_pred)

        # KMeans
        kmean_model = KMeans(n_clusters=2, random_state=42)
        kmean_model.fit(x_train)
        kmean_pred = [1 if cluster == 0 else 0 for cluster in kmean_model.predict(x_test)]
        accuracy_scores['KMeans'] = accuracy_score(y_test, kmean_pred)

        # Display accuracy scores
        scores_msg = "\n".join([f"{algo}: {score * 100:.2f}%" for algo, score in accuracy_scores.items()])
        messagebox.showinfo("Accuracy Scores", f"Model Accuracy:\n\n{scores_msg}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to train models: {str(e)}")

# Prediction Functions
def predict_cancer(input_features):
    logistic_model = LogisticRegression(max_iter=10000)
    logistic_model.fit(x_train, y_train)
    input_features = scaler.transform([input_features])
    prediction = logistic_model.predict(input_features)
    return "Malignant (Cancer detected)" if prediction == 1 else "Benign (No cancer)"

def predict_cancer_svm(input_features):
    svm_model = SVC()
    svm_model.fit(x_train, y_train)
    input_features = scaler.transform([input_features])
    prediction = svm_model.predict(input_features)
    return "Malignant (Cancer detected)" if prediction == 1 else "Benign (No cancer)"

def predict_cancer_knn(input_features):
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(x_train, y_train)
    input_features = scaler.transform([input_features])
    prediction = knn_model.predict(input_features)
    return "Malignant (Cancer detected)" if prediction == 1 else "Benign (No cancer)"

def predict_cancer_kmeans(input_features):
    kmean_model = KMeans(n_clusters=2, random_state=42)
    kmean_model.fit(x_train)
    input_features = scaler.transform([input_features])
    cluster = kmean_model.predict(input_features)[0]
    return "Malignant (Cancer detected)" if cluster == 0 else "Benign (No cancer)"

# Prediction Function for GUI
def make_prediction():
    try:
        input_data = entry.get()
        input_features = list(map(float, input_data.split(',')))
        if len(input_features) != 30:
            messagebox.showerror("Error", "Please enter exactly 30 features.")
            return
        selected_algo = prediction_method.get()
        if selected_algo == "logistic_regression":
            result = predict_cancer(input_features)
        elif selected_algo == "svm":
            result = predict_cancer_svm(input_features)
        elif selected_algo == "KNN":
            result = predict_cancer_knn(input_features)
        elif selected_algo == "KMeans":
            result = predict_cancer_kmeans(input_features)
        else:
            messagebox.showerror("Error", "No algorithm selected.")
            return
        messagebox.showinfo("Prediction Result", f"Algorithm: {selected_algo}\nPrediction: {result}")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values.")
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed: {str(e)}")

# GUI Code
root = tk.Tk()
root.title("Breast Cancer Prediction")

# Buttons for each step
tk.Button(root, text="1. Load Data", command=load_data, font=("Arial", 12), bg="lightblue").pack(pady=10)
tk.Button(root, text="2. Clean Data", command=clean_data, font=("Arial", 12), bg="lightblue").pack(pady=10)
tk.Button(root, text="3. Train Models", command=train_models, font=("Arial", 12), bg="lightblue").pack(pady=10)

# Input field and algorithm selection
tk.Label(root, text="Enter the 30 features as a comma-separated list:", font=("Arial", 14)).pack(pady=10)
entry = tk.Entry(root, width=100)
entry.pack(pady=10)

prediction_method = tk.StringVar(value="logistic_regression")
tk.Radiobutton(root, text="Logistic Regression", variable=prediction_method, value="logistic_regression", font=("Arial", 12)).pack(anchor="w")
tk.Radiobutton(root, text="SVM", variable=prediction_method, value="svm", font=("Arial", 12)).pack(anchor="w")
tk.Radiobutton(root, text="KNN", variable=prediction_method, value="KNN", font=("Arial", 12)).pack(anchor="w")
tk.Radiobutton(root, text="KMeans", variable=prediction_method, value="KMeans", font=("Arial", 12)).pack(anchor="w")

tk.Button(root, text="4. Predict", command=make_prediction, font=("Arial", 12), bg="lightblue").pack(pady=10)

# Run the GUI
root.mainloop()
