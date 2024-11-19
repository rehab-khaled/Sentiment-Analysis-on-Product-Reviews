import numpy as np
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__, template_folder='templates')

model = load_model('SaveModels/lstm_model.keras')

maxlen = 30
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

with open('SaveModels/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html', result='')

@app.route('/analyze', methods=['POST'])
def analyze():
    value = request.form.get("user")
    text = re.sub(r'(#|@)\w*', '', value)
    text = re.sub(r'https?://[^\s]+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = word_tokenize(text.lower())
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    
    sequence = tokenizer.texts_to_sequences([' '.join(text)])
    padded_sequence = pad_sequences(sequence, maxlen=maxlen)
    prediction = model.predict(padded_sequence)
    
    predicted_class = np.argmax(prediction, axis=1)[0]
    result = "Positive" if predicted_class == 1 else "Negative"
    return render_template('index.html', result=result, user_input=value)

if __name__ == "__main__":
    app.run(debug=True)