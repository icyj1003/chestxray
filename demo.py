import streamlit as st
import pandas as pd
import os
from PIL import Image
import numpy as np
import tensorflow as tf

image = None
file = None
label_list = ['covid', 'normal', 'pneumonia']
show_text = 'Vui lòng chọn ảnh'

with st.sidebar:
    model_name = st.selectbox('Chọn mô hình', os.listdir('./models/'))
    file = st.file_uploader('Chọn ảnh')
    if file is not None:
        image = np.array(Image.open(file).convert('RGB').resize((224, 224), Image.ANTIALIAS))/255
        model = tf.keras.models.load_model(f'./models/{model_name}')
        dataset = tf.data.Dataset.from_generator(lambda: [image], tf.float32)
        dataset = dataset.batch(32)
        ans = model.predict(dataset)[0]
        label_index = np.argmax(ans)
if file:
    col1, col2 = st.columns(2)
    with col1:
        st.image(image)
        for label, proba in zip(['covid', 'normal', 'pneumonia'], list(ans)):
            st.metric(label, proba)
    with col2:
        if label_index == 0:
            st.markdown("""
<style>
.big-font {
    font-size:100px !important;
    color: red;
}
</style>
""", unsafe_allow_html=True)
        elif label_index ==1:
            st.markdown("""
<style>
.big-font {
    font-size:100px !important;
    color: green;
}
</style>
""", unsafe_allow_html=True)
        elif label_index ==2:
            st.markdown("""
<style>
.big-font {
    font-size:100px !important;
    color: orange;
}
</style>
""", unsafe_allow_html=True)
        
        st.markdown(f'<p class="big-font">{label_list[label_index]}</p>', unsafe_allow_html=True)

else:
    st.text(show_text)

# streamlit run ./demo.py