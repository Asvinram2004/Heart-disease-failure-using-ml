# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns

st.set_page_config(
        page_title="Heart Failure Prediction Dashboard",
)

# Function to load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
    return data

# Function to preprocess new patient data
@st.cache_data
def preprocess_new_data(_scaler, data, _training_columns):
    # Ensure the new patient data has the same columns as the training data
    data = data.reindex(columns=_training_columns, fill_value=0)
    return _scaler.transform(data)

# Function to train the stacked model
def train_model(X_train, y_train):
    base_models = [
        ('svm', SVC(probability=True, random_state=42)),
        ('log_reg', LogisticRegression(random_state=42))
    ]
    meta_model = LogisticRegression(random_state=42)
    stacked_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
    stacked_model.fit(X_train, y_train)
    return stacked_model

# Function to display accuracy comparison
def plot_accuracy(past_accuracy, current_accuracy):
    plt.figure(figsize=(8, 4))
    plt.figure(figsize=(8, 4))
    
    # Plot lines
    plt.plot(['Past Accuracy', 'Current Accuracy'], [past_accuracy, current_accuracy], marker='o', linestyle='-', color='green', linewidth=2, markersize=8)
    plt.plot(['Past Accuracy'], [past_accuracy], marker='o', linestyle='-', color='red', linewidth=2, markersize=8)
    
    # Annotate the points with their values
    plt.text(0, past_accuracy, f'{past_accuracy:.2f}', ha='right', va='bottom', color='red')
    plt.text(1, current_accuracy, f'{current_accuracy:.2f}', ha='left', va='top', color='green')
    
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.title('Accuracy Comparison')
    plt.legend(['Current Accuracy', 'Past Accuracy'], loc='lower right')  # Add legend
    st.pyplot(plt)

# Load and preprocess the data
data = load_data()
X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']
training_columns = X.columns  # Get columns used during training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the stacked model
stacked_model = train_model(X_train, y_train)
y_pred = stacked_model.predict(X_test)
current_accuracy = accuracy_score(y_test, y_pred)

# Streamlit app
st.title('Heart Failure Prediction Dashboard')

st.header('Accuracy Comparison')
plot_accuracy(past_accuracy=0.75, current_accuracy=current_accuracy)

st.header('Classification Report')
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

st.header('Confusion Matrix')
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot(fig)

st.header('ROC Curve')
y_pred_proba = stacked_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
ax.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
st.pyplot(fig)

st.header('Base Logistic Regression Coefficients')
log_reg_model = stacked_model.named_estimators_['log_reg']
coefficients = log_reg_model.coef_[0]
features = training_columns  # Use columns used during training
coef_df = pd.DataFrame({'Feature': features, 'Coefficient': coefficients})
st.dataframe(coef_df)

st.header('Predict Heart Failure for a New Patient')
uploaded_file = st.file_uploader("Choose a file with new patient data (in CSV format)", type="csv")
if uploaded_file is not None:
    new_patient_data = pd.read_csv(uploaded_file)
    
    # Preprocess the new patient data
    try:
        new_patient_data = preprocess_new_data(scaler, new_patient_data, training_columns)
        prediction = stacked_model.predict(new_patient_data)
        probability = stacked_model.predict_proba(new_patient_data)[:, 1]  # Probability of class 1 (heart failure)
        
        result = 'Heart failure predicted' if prediction[0] == 1 else 'No heart failure predicted'
        st.write(result)
        st.write(f'Probability of heart failure: {probability[0]:.2f}')
    except ValueError as e:
        st.error(f"Error: {e}. Ensure the new patient data matches the format and columns used during training.")
