import os
import joblib
import logging
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Security: API Key
API_KEY = os.getenv("API_KEY", "your-secure-api-key")  # Load API key from .env
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate Limiting to prevent abuse (10 requests per minute per user)
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI
app = FastAPI()

# Register rate limiter exception handler
app.state.limiter = limiter
app.add_exception_handler(_rate_limit_exceeded_handler)

# Enable CORS (Allow frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained models and vectorizers
try:
    vectorizer = joblib.load("vectorizer.pkl")
    model = joblib.load("model.pkl")
    email_vectorizer = joblib.load("email_vectorizer.pkl")
    email_model = joblib.load("email_model.pkl")
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise RuntimeError("Failed to load machine learning models.")

# Input models
class CommentRequest(BaseModel):
    comment: str = Field(..., min_length=3, max_length=500, description="Text of the comment")

class EmailRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=1000, description="Email content")

# API Key Validation
def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

@app.get("/", tags=["Root"])
def read_root():
    """Root endpoint to check API status."""
    return {"message": "Welcome to the Fake Comment & Spam Email Detector API"}

@app.post("/detect-fake-comment", tags=["Fake Comment Detection"])
@limiter.limit("10/minute")
async def detect_fake_comment(comment: CommentRequest, api_key: str = Depends(verify_api_key)):
    """
    Detects if a comment is fake or real.
    """
    try:
        comment_vector = vectorizer.transform([comment.comment])
        prediction = model.predict(comment_vector)[0]
        result = "Fake Comment" if prediction == 1 else "Real Comment"
        return {"result": result}
    except Exception as e:
        logger.error(f"Error detecting fake comment: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/detect-spam-email", tags=["Spam Email Detection"])
@limiter.limit("10/minute")
async def detect_spam_email(email: EmailRequest, api_key: str = Depends(verify_api_key)):
    """
    Detects if an email is spam or not.
    """
    try:
        email_vector = email_vectorizer.transform([email.email])
        prediction = email_model.predict(email_vector)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        return {"result": result}
    except Exception as e:
        logger.error(f"Error detecting spam email: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

//


import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

# Libraries for NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.sparse import hstack, csr_matrix

# Ensure all required NLTK resources are downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
import nltk
nltk.download('punkt_tab')

# Load multiple datasets
file_paths = ['fakecomments1.csv', 'fakecomments2.csv', 'fakecomments3.csv', 'fakecomments4.csv', 'fakecomments5.csv']
dfs = []

for file_path in file_paths:
    temp_df = pd.read_csv(file_path)
    temp_df.drop(columns=['DATE', 'COMMENT_ID'], inplace=True)  # Drop unwanted columns
    dfs.append(temp_df)

# Combine datasets
df = pd.concat(dfs, ignore_index=True)

# Check target column
df['CLASS'] = df['CLASS'].fillna(0).astype(int)  # Ensure CLASS is integer and no NaN values

# Define preprocessing functions
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def advanced_text_preprocessing(text):
    if isinstance(text, float):
        text = str(text)
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

df['transformed_text'] = df['CONTENT'].apply(advanced_text_preprocessing)

# Split data into train and test sets
X = df['transformed_text']
y = df['CLASS']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Extraction Methods
# 1. TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
x_test_tfidf = tfidf_vectorizer.transform(x_test)

# 2. Transformer-based Embeddings (e.g., BERT)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

def get_bert_embeddings_in_batches(texts, tokenizer, model, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)  # Move the model to the specified device
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        tokens = tokenizer(list(batch_texts), padding=True, truncation=True, return_tensors="pt", max_length=512)
        tokens = {key: value.to(device) for key, value in tokens.items()}  # Move tokens to the device
        with torch.no_grad():
            outputs = model(**tokens)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Move back to CPU for numpy
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

x_train_bert = get_bert_embeddings_in_batches(x_train, tokenizer, model)
x_test_bert = get_bert_embeddings_in_batches(x_test, tokenizer, model)

# Combine Features (TF-IDF + BERT)
x_train_combined = hstack([csr_matrix(x_train_tfidf), csr_matrix(x_train_bert)])
x_test_combined = hstack([csr_matrix(x_test_tfidf), csr_matrix(x_test_bert)])

# Calculate class weights
class_weights = {
    0: 1,
    1: len(y_train[y_train == 0]) / len(y_train[y_train == 1])
}

# Model Training and Evaluation
models = {
    'Random Forest': RandomForestClassifier(class_weight=class_weights),
    'Logistic Regression': LogisticRegression(max_iter=200, class_weight='balanced'),
    'Support Vector Machine': SVC(class_weight='balanced'),
    'XGBoost': XGBClassifier(scale_pos_weight=class_weights[0] / class_weights[1])
}

results = []

for model_name, model in models.items():
    # Train model
    model.fit(x_train_combined, y_train)
    # Predictions
    y_pred = model.predict(x_test_combined)
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append([model_name, accuracy, precision, recall, f1])

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Spam", "Spam"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.show()

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

# Visualize Metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
for i, metric in enumerate(metrics):
    sns.barplot(x='Model', y=metric, data=results_df, ax=axes[i], palette="viridis")
    axes[i].set_title(metric)

plt.tight_layout()
plt.show()

# Print model accuracies
print("Model Accuracies:")
for index, row in results_df.iterrows():
    print(f"{row['Model']}: {row['Accuracy']:.4f}")
