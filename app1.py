import streamlit as st
import pickle
import numpy as np
import pandas as pd
#from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

model=pickle.load(open('C:/Users/shubh/OneDrive/Desktop/DISEASE/model1.pkl','rb'))

def predict_forest(rm,tm,pm,am,sm,cm,cv,cpm,sym,rse,pse,ase,cse,cvse,cvpse,fd,rw,tw,pw,aw,sw,cw,cvw,cpw,syw,fdw):
    input=np.array([[rm,tm,pm,am,sm,cm,cv,cpm,sym,rse,pse,ase,cse,cvse,cvpse,fd,rw,tw,pw,aw,sw,cw,cvw,cpw,syw,fdw]]).astype(np.float64)
    prediction=model.predict_proba(input)
    pred='{0:.{1}f}'.format(prediction[0][0], 9)
    return float(pred)

from PIL import Image
st.header("PREDICTIVE MODEL FOR BREAST CANCER")
image = Image.open('2.png')
st.image(image, width=400)
left_column, right_column = st.beta_columns(2)
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

if st.button("View Table"):
    st.table(df.head(5))

st.text('Enter values to Predict Breast Cancer')
rm = st.text_input("Radius Mean")
tm = st.text_input("Texture Mean")
pm = st.text_input("Perimeter Mean")
am = st.text_input("Area Mean")
sm = st.text_input("Smoothness Mean")
cm = st.text_input("Compactness Mean")
cv = st.text_input("Concavity Mean")
cpm = st.text_input("Concave points Mean")
sym= st.text_input("Symmetry Mean")
rse= st.text_input("Radius SE")
pse= st.text_input("Perimeter SE")
ase= st.text_input("Area SE")
cse= st.text_input("Compactness SE")
cvse= st.text_input("Concavity SE")
cvpse= st.text_input("Concave Points SE")
fd= st.text_input("Fractal Dimensions SE")
rw= st.text_input("Radius Worst")
tw= st.text_input("Texture Worst")
pw= st.text_input("Perimeter Worst")
aw= st.text_input("Area Worst")
sw= st.text_input("Smoothness Worst")
cw= st.text_input("Compactness Worst")
cvw= st.text_input("Concavity Worst")
cpw= st.text_input("Concave points Worst")
syw= st.text_input("Symmetry Worst")
fdw= st.text_input("Fractal Dimensions Worst")





def main():

    if st.button("Predict"):
        output=predict_forest(rm,tm,pm,am,sm,cm,cv,cpm,sym,rse,pse,ase,cse,cvse,cvpse,fd,rw,tw,pw,aw,sw,cw,cvw,cpw,syw,fdw)

        if output < 0.85:
            st.success('The probability is that patient might have Breast Cancer'.format(output))
        else:
            st.success('The probability is that patient might not have Breast Cancer'.format(output))


if __name__=='__main__':
    main()
