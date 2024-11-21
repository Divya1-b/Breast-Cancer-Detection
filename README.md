Breast Cancer Prediction Using Machine Learning
This project demonstrates a Breast Cancer Prediction System using machine learning algorithms to predict whether a tumor is malignant or benign. It utilizes the Logistic Regression model, trained on a dataset that includes various features of cell nuclei from breast cancer biopsies.

Objective:
The primary objective of this project is to predict the nature of breast cancer (either malignant or benign) based on various features of cell nuclei. The prediction model is based on a machine learning classifier (Logistic Regression), and the project also includes a Graphical User Interface (GUI) built using Tkinter to interact with the model.

Dataset:
The dataset used for this project is the Breast Cancer Wisconsin (Diagnostic) dataset, which can be loaded from the sklearn.datasets module. The dataset includes 569 samples with 30 features for each sample. These features describe various characteristics of the cell nuclei, such as:

Radius of the tumor
Texture of the tumor
Perimeter of the tumor
Area of the tumor
Smoothness, compactness, concavity, and other metrics
The target variable is diagnosis, where:

1 represents Malignant (cancerous) tumors.
0 represents Benign (non-cancerous) tumors.
The dataset is preprocessed by:

Removing unnecessary columns (like id and Unnamed: 32).
Encoding the diagnosis column (M as 1, B as 0).
Splitting the data into training and testing sets.
