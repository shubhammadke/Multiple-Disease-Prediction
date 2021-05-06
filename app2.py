import streamlit as st
import pickle
import numpy as np
import pandas as pd
#from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

model=pickle.load(open('C:/Users/shubh/OneDrive/Desktop/DISEASE/model2.pkl','rb'))

def predict_forest(rm,tm,pm,am,sm,cpm,sym,rse,pse,ase,cse,cvse,zz):
    input=np.array([[rm,tm,pm,am,sm,cpm,sym,rse,pse,ase,cse,cvse,zz]]).astype(np.float64)
    prediction=model.predict_proba(input)
    pred='{0:.{1}f}'.format(prediction[0][0], 9)
    return float(pred)

from PIL import Image
st.header("PREDICTIVE MODEL FOR HEART DISEASE")
image = Image.open('3.jfif')
st.image(image, width=400)
left_column, right_column = st.beta_columns(2)
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

if st.button("View Table"):
    st.table(df.head(5))

st.text('Enter values to Predict Heart Disease')
rm = st.text_input("Age")
tm = st.text_input("Sex 0 and 1")
pm = st.text_input("Chest Pain Type")
am = st.text_input("Resting Blood Pressure")
sm = st.text_input("Serum Cholestoral in mg/dl")
cpm = st.text_input("Fasting Blood Sugar")
sym= st.text_input("Resting EletroCardio Graph")
rse= st.text_input("Maximum Heat rate Achieved")
pse= st.text_input("Excercise Induced Angina")
ase= st.text_input("ST Depression")
cse= st.text_input("The Slope of Peak Exercise")
cvse= st.text_input("Number of Major Vessels")
zz= st.text_input("3=Normal,6=Fixed Defect,7=Never")





def main():

    if st.button("Predict"):
        output=predict_forest(rm,tm,pm,am,sm,cpm,sym,rse,pse,ase,cse,cvse,zz)

        if output < 0.85:
            st.success('The patient having Heart Disease'.format(output))
        else:
            st.success('The patient not having Heart Disease'.format(output))


if __name__=='__main__':
    main()
