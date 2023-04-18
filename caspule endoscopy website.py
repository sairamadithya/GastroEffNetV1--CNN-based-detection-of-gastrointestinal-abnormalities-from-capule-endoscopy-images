#!/usr/bin/env python
# coding: utf-8

# In[5]:


#%%writefile capsule.py
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
import pandas as pd
st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.pexels.com/photos/956999/milky-way-starry-sky-night-sky-star-956999.jpeg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
html_temp = """ 
  <div style="background-color:pink ;padding:6px;">
  <h2 style="color:white;text-align:center;">GastroEffNetV1- CNN based Automated detection of Gastrointestinal abnormalities from capsule endoscopy images</h2>
  </div>
  """ 
st.markdown(html_temp,unsafe_allow_html=True)
st.write(' ')
st.write(' ')
html_temp1 = """ 
  <div style="background-color:#ABBAEA ;padding:6px;">
  <h2 style="color:white;text-align:center;">CENTER OF EXCELLENCE IN MEDICAL IMAGING</h2>
  </div>
  """ 
st.markdown(html_temp1,unsafe_allow_html=True)
activities=['SECTION 1-Introduction','SECTION 2-Deep learning tool for detection of gastrointestinal abnormalities using capsule endoscopy','SECTION 3- About the team']
option=st.sidebar.selectbox('choose the options displayed below',activities) 
st.subheader(option) 
if option=='SECTION 1-Introduction':
    st.subheader('This website can predict the following gastrointestinal abnormalities from capsule endoscopy images using deep learning algorithms.')
    st.subheader('1. Polyp')
    st.write("A colon polyp is a small clump of cells that forms on the lining of the colon. Most colon polyps are harmless. But over time, some colon polyps can develop into colon cancer, which may be fatal when found in its later stages.")
    st.subheader('2. Ulcer')
    st.write("Ulcer is an inflammatory bowel disease (IBD) that causes inflammation and ulcers (sores) in your digestive tract. Ulcerative colitis affects the innermost lining of your colon, and rectum.")
    st.subheader('3. Esophagitis')
    st.write("Esophagitis is inflammation of the esophagus. Esophagitis can cause painful, difficult swallowing and chest pain.")
elif option=='SECTION 2-Deep learning tool for detection of gastrointestinal abnormalities using capsule endoscopy':
    @st.cache(allow_output_mutation=True)
    def load_model():
        model=tf.keras.models.load_model(r"D:\medical datasets\capsule endoscopy.h5")
        return model
    with st.spinner('Model is being loaded..'):
        model=load_model()
    file = st.file_uploader("Please upload any image from the local machine in case of computer or upload camera image in case of mobile.", type=["jpg", "png","jpeg"])
    st.set_option('deprecation.showfileUploaderEncoding', False)
    if file is None:
         st.text("Please upload an image file within the allotted file size")
    else:
        img = Image.open(file)
        st.image(img, use_column_width=True)
        size = (224,224)    
        image = ImageOps.fit(img, size, Image.ANTIALIAS)
        imag = np.asarray(image)
        imaga = np.expand_dims(imag,axis=0) 
        predictions = model.predict(imaga)
     #predictions = import_and_predict(image, model)
        a=np.argmax(predictions,axis=1)
    if st.button('predict pathology'):
        if a==0:
            st.markdown(""" 
  <div style="background-color: green;padding:3px;border: 3px solid;">
  <h2 style="color:white;text-align:center;">No pathology detected!!</h2>
  </div>
  """ ,unsafe_allow_html=True)
        elif a==1:
            st.markdown(""" 
  <div style="background-color: rgb(200,0,0);padding:3px;border: 3px solid;">
  <h2 style="color:white;text-align:center;">The predicted pathology is ulcerative colitis</h2>
  </div>
  """ ,unsafe_allow_html=True)
        elif a==2:
            st.markdown(""" 
  <div style="background-color: rgb(200,0,0);padding:3px;border: 3px solid;">
  <h2 style="color:white;text-align:center;">The predicted pathology is polyp</h2>
  </div>
  """ ,unsafe_allow_html=True)
        else:
            st.markdown(""" 
  <div style="background-color: rgb(170,0,0);padding:3px;border: 3px solid;">
  <h2 style="color:white;text-align:center;">The predicted pathology is esophagitis</h2>
  </div>
  """ ,unsafe_allow_html=True)
elif option=='SECTION 3- About the team':
    st.markdown(""" 
  <div style="background-color: rgb(43, 204, 166);padding:3px;border: 3px solid;">
  <h2 style="color:white;text-align:center;">V.A.Sairam, Department of Biomedical Engineering, Rajalakshmi Engineering College, Chennai, India</h2>
  </div>
  """ ,unsafe_allow_html=True)
    st.write('   ')
    st.write('   ')
    st.markdown(""" 
  <div style="background-color: rgb(230, 149, 213);padding:3px;border: 3px solid;">
  <h2 style="color:white;text-align:center;">G.K.Krithika, Department of Biomedical Engineering, Rajalakshmi Engineering College, Chennai, India</h2>
  </div>
  """ ,unsafe_allow_html=True)
    st.write('   ')
    st.write('   ')
    st.markdown(""" 
  <div style="background-color: rgb(192, 219, 103);padding:3px;border: 3px solid;">
  <h2 style="color:white;text-align:center;">P.Dhanusha, Department of Biomedical Engineering, Rajalakshmi Engineering College, Chennai, India</h2>
  </div>
  """ ,unsafe_allow_html=True)
    st.write('   ')
    st.write('   ')
    st.markdown(""" 
  <div style="background-color: rgb(219, 103, 165);padding:3px;border: 3px solid;">
  <h2 style="color:white;text-align:center;">C.S.Harini, Department of Biomedical Engineering, Rajalakshmi Engineering College, Chennai, India</h2>
  </div>
  """ ,unsafe_allow_html=True)
    st.write('   ')
    st.write('   ')
    st.markdown(""" 
  <div style="background-color: rgb(252, 172, 73);padding:3px;border: 3px solid;">
  <h2 style="color:white;text-align:center;">G.E.Chandrashekhar, Department of Biomedical Engineering, Rajalakshmi Engineering College, Chennai, India</h2>
  </div>
  """ ,unsafe_allow_html=True)
    st.write('   ')
    st.write('   ')
    st.subheader('Mentors')
    st.markdown(""" 
  <div style="background-color: green;padding:3px;border: 3px solid;">
  <h2 style="color:white;text-align:center;">Dr.S.Rajkumar, Head of the Department, Department of Biomedical Engineering, Rajalakshmi Engineering College, Chennai, India</h2>
  </div>
  """ ,unsafe_allow_html=True)
    st.write('   ')
    st.write('   ')
    st.markdown(""" 
  <div style="background-color: green;padding:3px;border: 3px solid;">
  <h2 style="color:white;text-align:center;">Dr.V.Sapthagirivasan, Capegemini Technology Sevices Limited India</h2>
  </div>
  """ ,unsafe_allow_html=True)
    st.subheader('Research publication')
    st.write('Check onto this link to read the research pre-print')
    st.write('https://www.researchsquare.com/article/rs-2588671/v1')
st.write(' ')
st.write(' ')
st.write('https://www.linkedin.com/in/sairamadithya')
st.write('https://github.com/sairamadithya')
st.write('https://medium.com/@sairamadithya2002')
st.write('https://www.quora.com/profile/Sairam-Adithya')

