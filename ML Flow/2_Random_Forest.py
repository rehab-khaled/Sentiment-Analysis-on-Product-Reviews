# Import necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import mlflow
import mlflow.sklearn
# Download necessary NLTK datasets
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

# Set the experiment to your existing one
mlflow.set_experiment('Sentiment_Analysis_Experiment')

# Load the dataset
df = pd.read_csv(r"C:\Users\BESTWAY\Downloads\Datasets\Dataset-SA.csv\Dataset-SA.csv")
df = df[:35000]
text = df['Summary']
sentiment = df['Sentiment']
data = pd.DataFrame({'text': text, 'sentiment': sentiment})

# Drop neutral data and missing values
data = data.dropna()
data = data[data['sentiment'] != 'neutral']
print(data['sentiment'].value_counts())

# Create a lemmatizer and stopwords list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to clean and lemmatize the text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply text preprocessing
data['text'] = data['text'].apply(lambda x: re.sub(r'(#|@)\w*', '', x))  # Remove hashtags
data['text'] = data['text'].apply(lambda x: re.sub("https?://S+", '', x))  # Remove links
data['text'] = data['text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))  # Remove special characters
data['text'] = data['text'].apply(preprocess_text)

# Convert text data into TF-IDF features
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(data['text']).toarray()

# Convert labels to binary (0 for negative, 1 for positive)
y = [0 if cls == 'negative' else 1 for cls in data['sentiment']]

# Split the data into training and test sets
# dataset = list(zip(X, y))
# random.shuffle(dataset)
# train_size = int(0.8 * len(X))
# X_train = X[:train_size]
# X_test = X[train_size:]
# y_train = y[:train_size]
# y_test = y[train_size:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

# Calculate class weights for handling imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# Function to train and log models in MLflow
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def train_and_log_model(model, model_name):
    with mlflow.start_run(run_name=model_name):
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        mlflow.sklearn.log_model(model, model_name)
        
        # Visualize Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title(f'Confusion Matrix - {model_name}')
        
        # Log the figure directly to MLflow
        mlflow.log_figure(plt.gcf(), f'confusion_matrix_{model_name}.png')
        plt.close()  # Close the plot to avoid displaying it in the notebook

# Instantiate model
rf_model = RandomForestClassifier(class_weight=class_weights)

# Train and log the model
train_and_log_model(rf_model, "Random Forest")