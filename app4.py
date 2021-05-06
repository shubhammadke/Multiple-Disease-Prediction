import streamlit as st
import pickle
import numpy as np
import pandas as pd
#from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

model=pickle.load(open('C:/Users/shubh/OneDrive/Desktop/DISEASE/model4.pkl','rb'))

def predict_forest(rm,tm,pm,am,sm):
    input=np.array([[rm,tm,pm,am,sm]]).astype(np.float64)
    prediction=model.predict_proba(input)
    pred='{0:.{1}f}'.format(prediction[0][0], 9)
    return float(pred)

from PIL import Image
st.header("PREDICTIVE MODEL FOR MALARIA TYPE DISEASE")
image = Image.open('5.png')
st.image(image, width=400)
left_column, right_column = st.beta_columns(2)

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

if st.button("View Table"):
    st.table(df.head(5))

st.text('Enter values to Predict Malaria')
rm = st.text_input("Chills per week")
tm = st.text_input("Fatigue level per week")
pm = st.text_input("Headache level")
am = st.text_input("Body Aches Level")
sm = st.text_input("Sweat level per week")

def main():

    if st.button("Predict"):
        output=predict_forest(rm,tm,pm,am,sm)

        if output < 0.85:
            st.success('The patient having Malaria Plasmodium falciparum '.format(output))
        else:
            st.success('The patient having Malaria Plasmodium vivax /ovale'.format(output))


if __name__=='__main__':
    main()
