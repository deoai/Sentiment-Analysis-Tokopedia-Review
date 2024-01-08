 %%writefile app.py
pip install tensorflow
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import pickle
with open('tokenizer.picklee', 'rb') as handle:
    tokenizer = pickle.load(handle)
nltk.download('punkt')
nltk.download('stopwords')

# Load your model
model = load_model('/content/model1 lastt.h5')  # Replace 'your_model.h5' with the actual path to your saved model

# Streamlit app
st.title("Text Sentiment Prediction")

# Input text box for the user to enter a sentence
sentence = st.text_area("Enter a sentence:")

#preprocessing
# Lowercase the sentence
sentence = sentence.lower()
# Remove punctuation marks
sentence = re.sub("[^A-Za-z]", " ", sentence)
# Remove additional blank spaces
sentence = re.sub(r"\s+", " ", sentence)
sentence = sentence.strip()
# Tokenize the words
sentence = word_tokenize(sentence)
# Remove stopwords
nltk.download('stopwords')
stop_words = stopwords.words("indonesian")
sentence = [word for word in sentence if word not in stop_words]
# Stemming using Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()
sentence = [stemmer.stem(word) for word in sentence]
# Joining it back
sentence = " ".join(map(str, sentence))

pad_len = 299
tokenized_sequence = tokenizer.texts_to_sequences([sentence])
padded_sequence = pad_sequences(tokenized_sequence, maxlen=pad_len)


# Tokenize and pad the sequence
# tokenizer = Tokenizer(6938)
# tokenizer.fit_on_texts([sentence])
# pad_len = 299
# tokenized_sequence = tokenizer.texts_to_sequences([sentence])
# padded_sequence = pad_sequences(tokenized_sequence, maxlen=pad_len)

# Make predictions when the user clicks the button
if st.button("Predict Sentiment"):
    if sentence:
        prediction = model.predict(padded_sequence)
        if prediction[0][0] > 0.5:
            st.success("Positive sentiment")
        else:
            st.error("Negative sentiment")
        st.write("Predicted sentiment score:", prediction[0][0])
        st.write("True sentiment:", "Positive" if prediction[0][0] > 0.5 else "Negative")
    else:
        st.warning("Please enter a sentence.")