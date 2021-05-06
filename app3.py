import streamlit as st
import pickle
import numpy as np
import pandas as pd
#from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

model=pickle.load(open('C:/Users/shubh/OneDrive/Desktop/DISEASE/model3.pkl','rb'))

def predict_forest(rm,tm,pm,am,sm,cm,cpm,sym,rse,pse,ase,cse,cvse,fd,rw,tw,pw,aw):
    input=np.array([[rm,tm,pm,am,sm,cm,cpm,sym,rse,pse,ase,cse,cvse,fd,rw,tw,pw,aw]]).astype(np.float64)
    prediction=model.predict_proba(input)
    pred='{0:.{1}f}'.format(prediction[0][0], 9)
    return float(pred)

from PIL import Image
st.header("PREDICTIVE MODEL FOR KIDNEY DISEASE")
image = Image.open('4.jpg')
st.image(image, width=400)
left_column, right_column = st.beta_columns(2)
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

if st.button("View Table"):
    st.table(df.head(5))

st.text('Enter values to Predict Kidney Disease')
rm = st.text_input("Age")
tm = st.text_input("BP")
pm = st.text_input("AL")
am = st.text_input("SU")
sm = st.text_input("RBC")
cpm = st.text_input("PC")
sym= st.text_input("PCC")
rse= st.text_input("BA")
pse= st.text_input("BGR")
ase= st.text_input("BU")
cse= st.text_input("SC")
cvse= st.text_input("POT")
cvpse= st.text_input("WC")
fd= st.text_input("HTN")
rw= st.text_input("DM")
tw= st.text_input("CAD")
pw= st.text_input("PE")
aw= st.text_input("ANE")

def main():

    if st.button("Predict"):
        output=predict_forest(rm,tm,pm,am,sm,cpm,sym,sym,rse,pse,ase,cse,cvse,fd,rw,tw,pw,aw)

        if output < 0.85:
            st.success('The patient having Kidney Disease/Failure'.format(output))
        else:
            st.success('The patient not having Kidney Disease/Failure'.format(output))


if __name__=='__main__':
    main()
