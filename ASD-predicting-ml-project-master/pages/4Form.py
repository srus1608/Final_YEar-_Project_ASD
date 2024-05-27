import streamlit as st
import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load dataset
autism_dataset = pd.read_csv('asd_data_csv.csv') 

# Assign column names
autism_dataset.columns = ["Social_Responsiveness", "Age", "Speech_Delay", "Learning_Disorder", 
                          "Genetic_Disorders", "Depression", "Intellectual_Disability", 
                          "Social_Behavioural_Issues", "Anxiety_Disorder", "Gender", 
                          "Suffers_from_Jaundice", "Family_member_history_with_ASD", "Outcome"]

# Separate features and target variable
X = autism_dataset.drop(columns='Outcome', axis=1)
Y = autism_dataset['Outcome']

# Standardize features
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Split dataset into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Define functions to preprocess input data
def value_count(str):
    if str == "Yes":
        return 1
    else:
        return 0

def sex(str):
    if str == "Female":
        return 1
    else:
        return 0

# Streamlit UI
st.title(":bookmark_tabs: :blue[Autism data assessment]")
st.write("---")
st.write("Fill the form below to check if your child is suffering from ASD ")

d1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
val1 = st.selectbox("Social Responsiveness ", d1)

d2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
val2 = st.selectbox("Age  ", d2)

d3 = ["No", "Yes"]
val3 = st.selectbox("Speech Delay  ", d3)
val3 = value_count(val3)

val4 = st.selectbox("Learning disorder  ", d3)
val4 = value_count(val4)

val5 = st.selectbox("Genetic disorders  ", d3)
val5 = value_count(val5)

val6 = st.selectbox("Depression  ", d3)
val6 = value_count(val6)

val7 = st.selectbox("Intellectual disability  ", d3)
val7 = value_count(val7)

val8 = st.selectbox("Social/Behavioural issues  ", d3)
val8 = value_count(val8)

val9 = st.selectbox("Anxiety disorder  ", d3)
val9 = value_count(val9)

d4 = ["Female", "Male"]
val10 = st.selectbox("Gender  ", d4)
val10 = sex(val10)

val11 = st.selectbox("Suffers from Jaundice ", d3)
val11 = value_count(val11)

val12 = st.selectbox("Family member history with ASD  ", d3)
val12 = value_count(val12)

input_data = [val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12]

# Convert input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape input data
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize input data
std_data = scaler.transform(input_data_reshaped)

# Make prediction
prediction = classifier.predict(std_data)

# Display prediction result
with st.expander("Analyze provided data"):
    st.subheader("Results:")
    if prediction[0] == 0:
        st.info('The person is not with Autism spectrum disorder')
        diagnosis = 'Not ASD'
    else:
        st.warning('The person is with Autism spectrum disorder')

# Define function to get recommendations based on age and ASD status
def get_age_recommendations(age, has_asd):
    recommendations = []
    if has_asd:
        if age < 5:
            recommendations.append("Consider enrolling the individual in early intervention programs specific to ASD.")
            recommendations.append("Encourage activities that promote social interaction and communication skills.")
            recommendations.append("https://www.cdc.gov/ncbddd/autism/links.html")
            recommendations.append("https://www.nimh.nih.gov/health/topics/autism-spectrum-disorders-asd")
        elif age >= 5 and age < 13:
            recommendations.append("Explore school-based support services and therapies tailored for ASD.")
            recommendations.append("Encourage participation in structured social activities.")
            recommendations.append("https://www.cdc.gov/ncbddd/autism/links.html")
            recommendations.append("https://www.nimh.nih.gov/health/topics/autism-spectrum-disorders-asd")
        elif age >= 13 and age < 18:
            recommendations.append("Support the development of self-advocacy skills for individuals with ASD.")
            recommendations.append("Explore transition planning for post-secondary education and employment with ASD in mind.")
            recommendations.append("https://www.cdc.gov/ncbddd/autism/links.html")
            recommendations.append("https://www.nimh.nih.gov/health/topics/autism-spectrum-disorders-asd")
        else:
            recommendations.append("Encourage independence in daily living skills for individuals with ASD.")
            recommendations.append("Explore adult services and support networks for individuals with ASD.")
            recommendations.append("https://www.cdc.gov/ncbddd/autism/links.html")
            recommendations.append("https://www.nimh.nih.gov/health/topics/autism-spectrum-disorders-asd")
    else:
        recommendations.append("No specific recommendations available for individuals without ASD.")

    return recommendations

# Main Streamlit code
def main(val2, prediction):
    st.title("Recommendation System")

    # Get age input from user
    age = val2 

    # Get ASD status input from prediction
    asd_status = prediction[0]

    # Create a button to generate recommendations
    button_clicked = st.button("Get Recommendations")

    # If the button is clicked
    if button_clicked:
        # Get recommendations based on age and ASD status
        age_recommendations = get_age_recommendations(age, asd_status)

        # Display recommendations
        st.subheader("Recommendations:")
        if age_recommendations:
            for recommendation in age_recommendations:
                st.write("- " + recommendation)
        else:
            st.write("No recommendations available.")

if __name__ == "__main__":
    main(val2, prediction)
