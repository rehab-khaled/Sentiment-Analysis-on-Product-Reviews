import numpy as np
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__, template_folder='templates')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

with open('SaveModels/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
with open('SaveModels/vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

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
    text = [" ".join(text)]
    input_data = tfidf.transform(text).toarray()
    output = model.predict(input_data)
    result = "Positive" if output[0] == 1 else "Negative"
    return render_template('index.html', result=result)


if __name__ == "__main__":
    app.run(debug=True)
