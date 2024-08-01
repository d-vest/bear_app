import streamlit as st
from fastai.vision.all import *
from fastai.vision.widgets import *
import pandas as pd
from io import StringIO

learn_inf = load_learner('model.pkl', cpu=True)

st.title("Bear Classifier App")
multi = '''Choose a picture of a **bear** to upload and the app can tell you if it's 
:red[black], :blue[grizzly], or :orange[teddy]. Exclusively these categories. 

Happy classification! :bear:
'''
st.markdown(multi)
uploaded_file = st.file_uploader("Upload a photo")

if uploaded_file is not None:
        img = PILImage.create(uploaded_file)
        st.image(img)
        pred,pred_idx,probs = learn_inf.predict(img)
        st.write(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04}')


end = '''Inspired by fast.ai's course **Deep Learning for Coders with Fastai and PyTorch**. 
Built on `Fastai`, `PyTorch`, and `Streamlit`.'''
st.markdown(end)