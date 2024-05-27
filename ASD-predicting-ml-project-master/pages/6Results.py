import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the dataset
autism_dataset = pd.read_csv('asd_data_csv.csv') 

# Separate features and labels
X = autism_dataset.drop(columns='Outcome', axis=1)
Y = autism_dataset['Outcome']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train SVM model
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_scaled, Y)

# Train logistic regression model
log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X_scaled, Y)

# Define function to calculate performance metrics
def calculate_metrics(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred)
    return accuracy, precision, recall, f1, roc_auc

# Calculate performance metrics for SVM
svm_metrics = calculate_metrics(svm_classifier, X_scaled, Y)

# Calculate performance metrics for logistic regression
log_reg_metrics = calculate_metrics(log_reg_classifier, X_scaled, Y)

# Create a DataFrame to display the metrics
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
    'SVM': svm_metrics,
    'Logistic Regression': log_reg_metrics
})

# Display the performance metrics in a table
st.title("Performance Metrics")
st.write(metrics_df)

# Load the dataset
autism_dataset = pd.read_csv('asd_data_csv.csv') 

# Separate features and labels
X = autism_dataset.drop(columns='Outcome', axis=1)
Y = autism_dataset['Outcome']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Train the SVM model
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, Y_train)

# Make predictions on training and testing data
Y_train_pred = svm_classifier.predict(X_train)
Y_test_pred = svm_classifier.predict(X_test)

# Calculate accuracy scores
train_accuracy = accuracy_score(Y_train, Y_train_pred)
test_accuracy = accuracy_score(Y_test, Y_test_pred)

# Plot the accuracy scores
fig, ax = plt.subplots(figsize=(8, 6))
labels = ['Training Accuracy', 'Testing Accuracy']
values = [train_accuracy, test_accuracy]
ax.bar(labels, values, color=['blue', 'green'])
ax.set_ylim(0, 1)
ax.set_title('Training and Testing Accuracy')
ax.set_ylabel('Accuracy')

# Display the accuracy scores
st.title('Training and Testing Accuracy')
st.write(f'Training Accuracy: {train_accuracy}')
st.write(f'Testing Accuracy: {test_accuracy}')

# Show plot
st.pyplot(fig)



# Load the dataset
autism_dataset = pd.read_csv('asd_data_csv.csv') 

# Separate features and labels
X = autism_dataset.drop(columns='Outcome', axis=1)
Y = autism_dataset['Outcome']

# Split the data into training, testing, and validation sets
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=42)
X_test, X_val, Y_test, Y_val = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Count the number of samples in each set
train_count = len(Y_train)
test_count = len(Y_test)
val_count = len(Y_val)

# Create labels and values for the pie chart
labels = ['Training', 'Testing', 'Validation']
sizes = [train_count, test_count, val_count]

# Plot the pie chart
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Display the pie chart
st.pyplot(fig)