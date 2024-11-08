# Loan Prediction System using Support Vector Machines (SVM)

## Project Overview
This project implements a **Loan Prediction System** using a **Support Vector Machine (SVM)** classifier. The model predicts whether a loan application will be approved or rejected based on features like income, loan amount, credit history, and education. The project demonstrates data preprocessing, feature selection, model training, and evaluation using Python.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Model Selection](#model-selection)
- [Evaluation](#evaluation)
- [Confusion Matrix Heatmap](#confusion-matrix-heatmap)
- [Future Scope](#future-scope)

## Dataset
The dataset contains **500 records** with features relevant to loan applications, such as:
- **ApplicantIncome**
- **CoapplicantIncome**
- **LoanAmount**
- **Loan_Amount_Term**
- **Credit_History**
- **Education**
- **Loan_Status** (Target variable: 1 for Approved, 0 for Rejected)

**Note**: The dataset used here is synthetic and may need real-world data for accurate predictions.

## Installation
To run this project, you will need:
1. **Python 3.7 or later**
2. **PyCharm Community Edition** (or any Python IDE)

### Dependencies
Install the required libraries by running:
```bash
pip install -r requirements.txt
```

### Clone the Repository
```bash
git clone https://github.com/username/Loan_Prediction_SVM.git
cd Loan_Prediction_SVM
```

## Project Structure
The project is organized as follows:
```
Loan_Prediction_SVM/
│
├── data/
│   ├── dataset.csv               # Sample dataset (synthetic data)
│
├── main.py                        # Main script for training and evaluating the model
├── requirements.txt               # Required Python libraries
├── README.md                      # Project README file
└── images/
    ├── heatmap.png                # Heatmap image of the confusion matrix
```

## Data Preprocessing
The preprocessing steps include:
- **Handling Missing Data**: Ensured no missing values.
- **Encoding Categorical Variables**: Education feature encoded (Graduate = 1, Not Graduate = 0).
- **Feature Scaling**: Scaled numerical features using StandardScaler.

## Model Selection
We used a **Support Vector Machine (SVM)** with a linear kernel due to its effectiveness in binary classification tasks. The model was trained on **80% of the data** and tested on **20%**.

### Hyperparameters
Default hyperparameters were used. Further tuning can be applied using `GridSearchCV`.

## Evaluation
The model's performance was evaluated using:
- **Accuracy**: 50%
- **Confusion Matrix**:
  - True Positives (TP): Correctly predicted approved loans
  - True Negatives (TN): Correctly predicted rejected loans
  - False Positives (FP): Incorrectly predicted approved loans
  - False Negatives (FN): Incorrectly predicted rejected loans

### Classification Report
- Precision, Recall, F1-score metrics for both classes.

## Confusion Matrix Heatmap
The heatmap below represents the confusion matrix for the model, showing correct and incorrect predictions:

![Confusion Matrix Heatmap](images/heatmap.png)

- The model performs well on approved loans but struggles with predicting rejected loans, indicating a need for class balancing.

## Future Scope
1. **Handle Class Imbalance**: Apply techniques like **SMOTE** or class weighting.
2. **Hyperparameter Tuning**: Use `GridSearchCV` to optimize SVM parameters.
3. **Feature Engineering**: Add new features to improve model performance.
4. **Comparison with Other Models**: Test models like **Random Forest** or **XGBoost**.
5. **Deployment**: Deploy the model using a web or mobile application for practical use.

## License
This project is licensed under the MIT License.

---

This **README.md** provides a structured overview of the project, including setup instructions, technical details, and future improvements.
