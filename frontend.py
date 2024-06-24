import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import requests
from PIL import Image
import io

st.title('Image Captioning System')

uploaded_file = st.file_uploader('choose an image file.....', type = "jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption = 'uploaded Image', use_column_width = True)

    if st.button('Generate Caption'):
        files = {'image': uploaded_file.getvalue()}
        response = requests.post('http://localhost:5000/predict', files=files)

        if response.status_code == 200:
            caption = requests.json()['caption']
            st.write('Generated Caption: ', caption)

        else:
            st.write("Error generating caption. Please try again")

            