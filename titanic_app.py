import numpy as np
import pickle
import streamlit as st

# Load model
try:
    model_file = pickle.load(open('titanic.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model: {e}")

# Prediction function
def pred_output(user_input):
    model_input = np.array(user_input).astype(float)  # Ensure all inputs are numeric
    ypred = model_file.predict(model_input.reshape(1, -1))  # Reshape correctly
    return ypred[0]

# Main function
def main():
    st.title("Titanic Classification Report")

    # Input variables
    passenger_class = st.text_input("Enter the passenger class: (1/2/3)")
    sex = st.radio("Select your sex:", options=['Male', 'Female'])

    # Convert sex to numeric
    sex = 0 if sex == 'Male' else 1

    age = st.text_input("Enter their age:")
    sibsp = st.text_input("Enter their siblings:")
    parch = st.text_input("Enter their parch:")
    fare = st.text_input("Enter their fare:")
    
    embarked = st.radio("Select your Embarked:", options=['C(Cherbourg)', 'S(Southampton)', 'Q(Queentown)'])

    # Convert embarked input to numeric
    if embarked == 'C(Cherbourg)':
        embarked = 1
    elif embarked == 'S(Southampton)':
        embarked = 0
    elif embarked == 'Q(Queentown)':
        embarked = 2
    else:
        st.error("Invalid Input!", icon='⚠️')

    # Predict button
    if st.button('Predict'):
        try:
            user_input = [int(passenger_class), int(sex), float(age), int(sibsp), int(parch), float(fare), int(embarked)]
            make_prediction = pred_output(user_input)

            result = "Survived" if make_prediction == 1 else "Not Survived"
            st.success(result)
        except ValueError:
            st.error("Please enter valid numeric inputs!")

# Run application
if __name__ == "__main__":
    main()