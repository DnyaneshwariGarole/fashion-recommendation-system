import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

#SAFETY 
os.makedirs("uploads", exist_ok=True)

# LOAD DATA
feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

# MODEL
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

# FUNCTIONS 
def save_uploaded_file(uploaded_file):
    try:
        path = os.path.join("uploads", uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return path
    except:
        return None

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded = np.expand_dims(img_array, axis=0)
    processed = preprocess_input(expanded)
    result = model.predict(processed).flatten()
    return result / norm(result)

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    _, indices = neighbors.kneighbors([features])
    return indices

uploaded_file = st.file_uploader(
    "Choose a fashion image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    saved_path = save_uploaded_file(uploaded_file)

    if saved_path:
        
        display_image = Image.open(uploaded_file).convert("RGB")
        st.image(display_image, caption="Uploaded Image", use_column_width=True)

        features = feature_extraction(saved_path, model)
        indices = recommend(features, feature_list)

        cols = st.columns(5)
        for i, col in enumerate(cols):
            with col:
                st.image(filenames[indices[0][i]])
    else:
        st.error("Some error occurred while uploading the file")
