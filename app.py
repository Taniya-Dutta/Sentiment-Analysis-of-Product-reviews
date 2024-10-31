import os
from flask import Flask, render_template, request
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Updated import
app = Flask(__name__)

# Load the trained model and tokenizer
try:
    with open('bidirectional_LSTM.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except Exception as e:
    print(f"Error loading model: {e}")

with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Define route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        # Preprocess the review text
        seq = tokenizer.texts_to_sequences([review])
        padded_sequence = pad_sequences(seq, maxlen=200)
        # Predict sentiment (0 or 1)
        prediction = model.predict(padded_sequence)
        sentiment = 'Positive' if prediction[0][0] >= 0.3 else 'Negative'
        sentiment_score = prediction[0][0]  # Display the raw score
        return render_template('result.html', review=review, sentiment=sentiment, sentiment_score=sentiment_score)

if __name__ == '__main__':
    app.run(debug=True)


