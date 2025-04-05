# Step 1: Import Libraries and Load the Model
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
# Get the absolute path to the model file
root_folder = os.path.dirname(__file__)
model_path = os.path.join(root_folder, 'simple_rnn_model.keras')
try:
    model = load_model(model_path, compile=False)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    """
    Decodes an integer-encoded review back into words.
    """
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    """
    Preprocesses the user input by tokenizing, encoding, and padding.
    """
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # Use 2 for unknown words
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    if not user_input.strip():
        st.warning("Please enter a valid movie review.")
    else:
        # Preprocess the input
        preprocessed_input = preprocess_text(user_input)

        # Make prediction
        try:
            prediction = model.predict(preprocessed_input)
            sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

            # Display the result
            st.write(f'Sentiment: {sentiment}')
            st.write(f'Prediction Score: {prediction[0][0]:.4f}')
        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.write('Please enter a movie review.')