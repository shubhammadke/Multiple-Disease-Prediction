import streamlit as st
import pickle
import numpy as np
import pandas as pd
#from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

model=pickle.load(open('C:/Users/shubh/OneDrive/Desktop/DISEASE/model.pkl','rb'))

def predict_forest(nop,g,bp,skin,insulin,bmi,dpf,age):
    input=np.array([[nop,g,bp,skin,insulin,bmi,dpf,age]]).astype(np.float64)
    prediction=model.predict_proba(input)
    pred='{0:.{1}f}'.format(prediction[0][0], 9)
    return float(pred)

from PIL import Image
st.header("PREDICTIVE MODEL FOR DIABETES")
image = Image.open('1.jpg')
st.image(image, width=400)
left_column, right_column = st.beta_columns(2)
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

if st.button("View Table"):
    st.table(df.head(5))

st.text('Enter values to Predict Diabetes')
nop = st.text_input("Number of Pregnancies")
g = st.text_input("Glucose")
bp = st.text_input("Blood Pressure")
skin = st.text_input("Skin Thickness")
insulin = st.text_input("Insulin Level")
bmi = st.text_input("Body Mass Index")
dpf = st.text_input("Diabetes Pedigree Function")
age= st.text_input("Age")

def main():

    if st.button("Predict"):
        output=predict_forest(nop,g,bp,skin,insulin,bmi,dpf,age)

        if output < 0.85:
            st.success('The probability of patient having Diabetes'.format(output))
        else:
            st.success('The probability of patient not having Diabetes'.format(output))


if __name__=='__main__':
    main()
