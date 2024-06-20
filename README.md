## Detection and Recommendation of Autism Spectrum Disorder using Python Machine Learning
## Description
This project aims to detect Autism Spectrum Disorder (ASD) and provide recommendations based on the predicted results. It is developed using Jupyter Notebook and Streamlit.

## The application includes the following features:

1. Register or Sign-in Page: User authentication.
2. Homepage: Information about ASD.
3. Dashboard Page: Displays statistics over 5 years, including gender-based statistics, jaundice statistics, Childhood Autism Rating Scale statistics, Family Member with ASD statistics, and Social Responsiveness Scale statistics.
4. Form Page: Includes AQ-10 questionnaires for users to fill out. Based on their responses, the model predicts whether the user has ASD and provides recommendations accordingly.
5. Contact Page: Contact information for further assistance.
6. Results Page: Displays a pie chart of accuracy performance metrics as a result of training and testing data.

## Installation Instructions
1. Clone the repository
2. Install the required packages:
  pip install -r requirements.txt

## Usage Instructions
To run the project, use the following command:

streamlit run register.py

## Dataset
The dataset used for this project is taken from Kaggle. It includes various features relevant to ASD detection.

## Model Details
Algorithms Used: Support Vector Machine (SVM) and Logistic Regression.
Feature Scaling Techniques: Standard Scaler and Normalization.

## Results
Accuracy:
1. SVM: 69%
2. Logistic Regression: 70%

## Publication
This project has been published in the Vietnam Journal of Computer Science. 
## Contributors
Srushti Talwekar
Rutuja Anemwad
