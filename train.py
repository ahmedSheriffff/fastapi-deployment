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
