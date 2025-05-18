import joblib
import logging
import os
from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Security: API Key
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY must be set in environment variables.")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)








import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Paths to the pre-trained model and vectorizer
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
DATASET_PATH = "fake_comments_dataset.csv"

class CommentRequest(BaseModel):
    comment: str

class CommentResponse(BaseModel):
    comment: str
    is_fake: bool

class CommentLabelRequest(BaseModel):
    comment: str
    label: int

# Load the pre-trained model and vectorizer
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
else:
    raise FileNotFoundError("Model and vectorizer files not found. Please ensure 'model.pkl' and 'vectorizer.pkl' are in the directory.")

@app.post("/detect_fake_comment", response_model=CommentResponse)
async def detect_fake_comment(request: CommentRequest):
    try:
        if not request.comment.strip():
            logger.warning("Empty comment received")
            raise HTTPException(status_code=400, detail="Comment cannot be empty.")
        transformed_comment = vectorizer.transform([request.comment])
        prediction = model.predict(transformed_comment)
        result = bool(prediction[0])
        logger.info(f"Comment: {request.comment} -> Predicted: {result}")
        return CommentResponse(comment=request.comment, is_fake=result)
    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the comment.")
    except Exception as e:
        logger.error(f"Unhandled Exception: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@app.post("/add_comment")
async def add_comment(request: CommentLabelRequest):
    try:
        if not request.comment.strip():
            logger.warning("Empty comment received for addition")
            raise HTTPException(status_code=400, detail="Comment cannot be empty.")
        
        new_data = pd.DataFrame({"comment": [request.comment], "label": [request.label]})
        if os.path.exists(DATASET_PATH):
            df = pd.read_csv(DATASET_PATH)
            df = pd.concat([df, new_data], ignore_index=True)
        else:
            df = new_data
        df.to_csv(DATASET_PATH, index=False)
        
        logger.info(f"Added new comment and retraining model: {request.comment}")
        train_model()
        
        return {"message": "Comment added and model retrained successfully."}
    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while adding the comment.")
    except Exception as e:
        logger.error(f"Unhandled Exception: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

# Training function
def train_model():
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)

        if "comment" not in df.columns or "label" not in df.columns:
            raise ValueError("Dataset must contain 'comment' and 'label' columns.")

        # Balance the dataset
        fake_comments = df[df["label"] == 1]
        real_comments = df[df["label"] == 0]

        min_samples = min(len(fake_comments), len(real_comments))

        df_balanced = pd.concat([
            fake_comments.sample(min_samples, random_state=42),
            real_comments.sample(min_samples, random_state=42)
        ])

        # Train the model with balanced data
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))  # Use n-grams
        X = vectorizer.fit_transform(df_balanced["comment"])
        y = df_balanced["label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Hyperparameter tuning using Grid Search
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_features': ['sqrt', 'log2'],  # Corrected values
            'max_depth': [4, 5, 6, 7, 8],
            'criterion': ['gini', 'entropy']
        }

        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, error_score='raise')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {accuracy:.2f}")
        print(classification_report(y_test, y_pred, zero_division=0))

        joblib.dump(best_model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)
    else:
        raise FileNotFoundError("Dataset file not found. Please add 'fake_comments_dataset.csv'.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




new 


import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of valid API keys
VALID_API_KEYS = ["actual_api_key_1", "actual_api_key_2"]

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        api_key = request.headers.get("X-API-Key")
        if api_key not in VALID_API_KEYS:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return await call_next(request)

# Initialize FastAPI application with middleware
app = FastAPI()
app.add_middleware(APIKeyMiddleware)

# Paths to the pre-trained model and vectorizer
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

class CommentRequest(BaseModel):
    comment: str

class CommentResponse(BaseModel):
    comment: str
    is_fake: bool

# Load the pre-trained model and vectorizer
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        logger.info("Model and vectorizer loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model or vectorizer: {str(e)}")
        raise
else:
    raise FileNotFoundError("Model and vectorizer files not found. Please ensure 'model.pkl' and 'vectorizer.pkl' are in the directory.")

@app.post("/detect_fake_comment", response_model=CommentResponse)
async def detect_fake_comment(request: CommentRequest):
    try:
        if not request.comment.strip():
            logger.warning("Empty comment received")
            raise HTTPException(status_code=400, detail="Comment cannot be empty.")
        transformed_comment = vectorizer.transform([request.comment])
        prediction = model.predict(transformed_comment)
        result = bool(prediction[0])
        logger.info(f"Comment: {request.comment} -> Predicted: {result}")
        return CommentResponse(comment=request.comment, is_fake=result)
    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the comment.")
    except Exception as e:
        logger.error(f"Unhandled Exception: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



email 

import joblib
from fastapi import FastAPI, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of valid API keys
VALID_API_KEYS = ["actual_api_key_1", "actual_api_key_2"]

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        api_key = request.headers.get("X-API-Key")
        if api_key not in VALID_API_KEYS:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return await call_next(request)

# Initialize FastAPI application with middleware
app = FastAPI()
app.add_middleware(APIKeyMiddleware)

# Paths to the pre-trained model and vectorizer
MODEL_PATH = "email_model.pkl"
VECTORIZER_PATH = "email_vectorizer.pkl"

class EmailRequest(BaseModel):
    email: str

class EmailResponse(BaseModel):
    email: str
    is_spam: bool

# Load the pre-trained model and vectorizer
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    try:
        spam_model = joblib.load(MODEL_PATH)
        spam_vectorizer = joblib.load(VECTORIZER_PATH)
        logger.info("Model and vectorizer loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model or vectorizer: {str(e)}")
        raise
else:
    raise FileNotFoundError("Spam model and vectorizer files not found. Please ensure 'email_model.pkl' and 'email_vectorizer.pkl' are in the directory.")

@app.post("/detect_spam_email", response_model=EmailResponse)
async def detect_spam_email(request: EmailRequest):
    try:
        if not request.email.strip():
            logger.warning("Empty email received")
            raise HTTPException(status_code=400, detail="Email cannot be empty.")
        logger.info(f"Received email: {request.email}")
        
        # Transform the email
        try:
            transformed_email = spam_vectorizer.transform([request.email])
            logger.info(f"Transformed email shape: {transformed_email.shape}")
        except Exception as e:
            logger.error(f"Error transforming email: {str(e)}")
            raise HTTPException(status_code=500, detail="Error transforming email.")
        
        # Predict using the model
        try:
            prediction = spam_model.predict(transformed_email)
        except Exception as e:
            logger.error(f"Error predicting spam: {str(e)}")
            raise HTTPException(status_code=500, detail="Error predicting spam.")
        
        result = bool(prediction[0])
        logger.info(f"Email: {request.email} -> Predicted: {result}")
        return EmailResponse(email=request.email, is_spam=result)
    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the email.")
    except Exception as e:
        logger.error(f"Unhandled Exception: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import os
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of valid API keys
VALID_API_KEYS = ["actual_api_key_1", "actual_api_key_2"]

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        api_key = request.headers.get("X-API-Key")
        if api_key not in VALID_API_KEYS:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return await call_next(request)

# Initialize FastAPI application with middleware
app = FastAPI()
app.add_middleware(APIKeyMiddleware)

# Paths to the pre-trained model and vectorizer
COMMENT_MODEL_PATH = "model.pkl"
COMMENT_VECTORIZER_PATH = "vectorizer.pkl"
SPAM_MODEL_PATH = "spam_model.pkl"
SPAM_VECTORIZER_PATH = "spam_vectorizer.pkl"
DATASET_PATH = "fake_comments_dataset.csv"

class CommentRequest(BaseModel):
    comment: str

class CommentResponse(BaseModel):
    comment: str
    is_fake: bool

class EmailRequest(BaseModel):
    content: str

class EmailResponse(BaseModel):
    content: str
    is_spam: bool

class CommentLabelRequest(BaseModel):
    comment: str
    label: int

# Load the models and vectorizers
if os.path.exists(COMMENT_MODEL_PATH) and os.path.exists(COMMENT_VECTORIZER_PATH):
    try:
        comment_model = joblib.load(COMMENT_MODEL_PATH)
        comment_vectorizer = joblib.load(COMMENT_VECTORIZER_PATH)
        logger.info("Comment model and vectorizer loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading comment model or vectorizer: {str(e)}")
        raise
else:
    raise FileNotFoundError("Comment model and vectorizer files not found.")

if os.path.exists(SPAM_MODEL_PATH) and os.path.exists(SPAM_VECTORIZER_PATH):
    try:
        spam_model = joblib.load(SPAM_MODEL_PATH)
        spam_vectorizer = joblib.load(SPAM_VECTORIZER_PATH)
        logger.info("Spam model and vectorizer loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading spam model or vectorizer: {str(e)}")
        raise
else:
    raise FileNotFoundError("Spam model and vectorizer files not found.")

@app.post("/detect_fake_comment", response_model=CommentResponse)
async def detect_fake_comment(request: CommentRequest):
    try:
        if not request.comment.strip():
            logger.warning("Empty comment received")
            raise HTTPException(status_code=400, detail="Comment cannot be empty.")
        transformed_comment = comment_vectorizer.transform([request.comment])
        prediction = comment_model.predict(transformed_comment)
        result = bool(prediction[0])
        logger.info(f"Comment: {request.comment} -> Predicted: {result}")
        return CommentResponse(comment=request.comment, is_fake=result)
    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the comment.")
    except Exception as e:
        logger.error(f"Unhandled Exception: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@app.post("/detect_spam_email", response_model=EmailResponse)
async def detect_spam_email(request: EmailRequest):
    """
    Detects if the email content is spam.
    """
    try:
        if not request.content.strip():
            logger.warning("Empty email content received.")
            raise HTTPException(status_code=400, detail="Email content cannot be empty.")

        # Preprocess the email content
        processed_content = re.sub(r"[^\w\s]", "", request.content.lower())
        transformed_content = spam_vectorizer.transform([processed_content])
        
        # Predict spam status
        prediction = spam_model.predict(transformed_content)
        is_spam = bool(prediction[0])
        
        logger.info(f"Email: {request.content} -> Predicted: {'spam' if is_spam else 'not spam'}")
        return EmailResponse(content=request.content, is_spam=is_spam)
    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the email.")
    except Exception as e:
        logger.error(f"Unhandled Exception: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@app.post("/add_comment")
async def add_comment(request: CommentLabelRequest):
    try:
        if not request.comment.strip():
            logger.warning("Empty comment received for addition")
            raise HTTPException(status_code=400, detail="Comment cannot be empty.")
        
        new_data = pd.DataFrame({"comment": [request.comment], "label": [request.label]})
        if os.path.exists(DATASET_PATH):
            df = pd.read_csv(DATASET_PATH)
            df = pd.concat([df, new_data], ignore_index=True)
        else:
            df = new_data
        df.to_csv(DATASET_PATH, index=False)
        
        logger.info(f"Added new comment and retraining model: {request.comment}")
        
        return {"message": "Comment added and model retrained successfully."}
    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while adding the comment.")
    except Exception as e:
        logger.error(f"Unhandled Exception: {str(e)}")
        raise
    
    
    
    
    
    #main 
    
    
    
    import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of valid API keys
VALID_API_KEYS = ["actual_api_key_1", "actual_api_key_2"]

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        api_key = request.headers.get("X-API-Key")
        if api_key not in VALID_API_KEYS:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return await call_next(request)

# Initialize FastAPI application with middleware
app = FastAPI()
app.add_middleware(APIKeyMiddleware)

# Paths to the pre-trained model and vectorizer
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
DATASET_PATH = "fake_comments_dataset.csv"

class CommentRequest(BaseModel):
    comment: str

class CommentResponse(BaseModel):
    comment: str
    is_fake: bool

class CommentLabelRequest(BaseModel):
    comment: str
    label: int

# Load the pre-trained model and vectorizer
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        logger.info("Model and vectorizer loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model or vectorizer: {str(e)}")
        raise
else:
    raise FileNotFoundError("Model and vectorizer files not found. Please ensure 'model.pkl' and 'vectorizer.pkl' are in the directory.")

@app.post("/detect_fake_comment", response_model=CommentResponse)
async def detect_fake_comment(request: CommentRequest):
    try:
        if not request.comment.strip():
            logger.warning("Empty comment received")
            raise HTTPException(status_code=400, detail="Comment cannot be empty.")
        transformed_comment = vectorizer.transform([request.comment])
        prediction = model.predict(transformed_comment)
        result = bool(prediction[0])
        logger.info(f"Comment: {request.comment} -> Predicted: {result}")
        return CommentResponse(comment=request.comment, is_fake=result)
    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the comment.")
    except Exception as e:
        logger.error(f"Unhandled Exception: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@app.post("/add_comment")
async def add_comment(request: CommentLabelRequest):
    try:
        if not request.comment.strip():
            logger.warning("Empty comment received for addition")
            raise HTTPException(status_code=400, detail="Comment cannot be empty.")
        
        new_data = pd.DataFrame({"comment": [request.comment], "label": [request.label]})
        if os.path.exists(DATASET_PATH):
            df = pd.read_csv(DATASET_PATH)
            df = pd.concat([df, new_data], ignore_index=True)
        else:
            df = new_data
        df.to_csv(DATASET_PATH, index=False)
        
        logger.info(f"Added new comment and retraining model: {request.comment}")
        
        return {"message": "Comment added and model retrained successfully."}
    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while adding the comment.")
    except Exception as e:
        logger.error(f"Unhandled Exception: {str(e)}")
        raise 
    
    
    
    
    
    
    
    import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import os
import logging
import re


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of valid API keys
VALID_API_KEYS = ["actual_api_key_1", "actual_api_key_2"]

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        api_key = request.headers.get("X-API-Key")
        if api_key not in VALID_API_KEYS:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return await call_next(request)

# Initialize FastAPI application with middleware
app = FastAPI()
app.add_middleware(APIKeyMiddleware)

# Paths to the pre-trained model and vectorizer
COMMENT_MODEL_PATH = "fake_comment_model.pkl"
COMMENT_VECTORIZER_PATH = "vectorizer.pkl"
SPAM_MODEL_PATH = "emailmodel.pkl"
SPAM_VECTORIZER_PATH = "spam_vectorizer.pkl"
DATASET_PATH = "fake_comments_dataset.csv"

class CommentRequest(BaseModel):
    comment: str

class CommentResponse(BaseModel):
    comment: str
    is_fake: bool

class EmailRequest(BaseModel):
    content: str

class EmailResponse(BaseModel):
    content: str
    is_spam: bool

class CommentLabelRequest(BaseModel):
    comment: str
    label: int

# Load the models and vectorizers
if os.path.exists(COMMENT_MODEL_PATH) and os.path.exists(COMMENT_VECTORIZER_PATH):
    try:
        comment_model = joblib.load(COMMENT_MODEL_PATH)
        comment_vectorizer = joblib.load(COMMENT_VECTORIZER_PATH)
        logger.info("Comment model and vectorizer loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading comment model or vectorizer: {str(e)}")
        raise
else:
    raise FileNotFoundError("Comment model and vectorizer files not found.")

if os.path.exists(SPAM_MODEL_PATH) and os.path.exists(SPAM_VECTORIZER_PATH):
    try:
        spam_model = joblib.load(SPAM_MODEL_PATH)
        spam_vectorizer = joblib.load(SPAM_VECTORIZER_PATH)
        logger.info("Spam model and vectorizer loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading spam model or vectorizer: {str(e)}")
        raise
else:
    raise FileNotFoundError("Spam model and vectorizer files not found.")

@app.post("/detect_fake_comment", response_model=CommentResponse)
async def detect_fake_comment(request: CommentRequest):
    try:
        if not request.comment.strip():
            logger.warning("Empty comment received")
            raise HTTPException(status_code=400, detail="Comment cannot be empty.")
        transformed_comment = comment_vectorizer.transform([request.comment])
        prediction = comment_model.predict(transformed_comment)
        result = bool(prediction[0])
        logger.info(f"Comment: {request.comment} -> Predicted: {result}")
        return CommentResponse(comment=request.comment, is_fake=result)
    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the comment.")
    except Exception as e:
        logger.error(f"Unhandled Exception: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@app.post("/detect_spam_email", response_model=EmailResponse)
async def detect_spam_email(request: EmailRequest):
    """
    Detects if the email content is spam.
    """
    try:
        if not request.content.strip():
            logger.warning("Empty email content received.")
            raise HTTPException(status_code=400, detail="Email content cannot be empty.")

        # Preprocess the email content
        processed_content = re.sub(r"[^\w\s]", "", request.content.lower())
        transformed_content = spam_vectorizer.transform([processed_content])
        
        # Predict spam status
        prediction = spam_model.predict(transformed_content)
        is_spam = bool(prediction[0])
        
        logger.info(f"Email: {request.content} -> Predicted: {'spam' if is_spam else 'not spam'}")
        return EmailResponse(content=request.content, is_spam=is_spam)
    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the email.")
    except Exception as e:
        logger.error(f"Unhandled Exception: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@app.post("/add_comment")
async def add_comment(request: CommentLabelRequest):
    try:
        if not request.comment.strip():
            logger.warning("Empty comment received for addition")
            raise HTTPException(status_code=400, detail="Comment cannot be empty.")
        
        new_data = pd.DataFrame({"comment": [request.comment], "label": [request.label]})
        if os.path.exists(DATASET_PATH):
            df = pd.read_csv(DATASET_PATH)
            df = pd.concat([df, new_data], ignore_index=True)
        else:
            df = new_data
        df.to_csv(DATASET_PATH, index=False)
        
        logger.info(f"Added new comment and retraining model: {request.comment}")
        
        return {"message": "Comment added and model retrained successfully."}
    except ValueError as e:
        logger.error(f"ValueError: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while adding the comment.")
    except Exception as e:
        logger.error(f"Unhandled Exception: {str(e)}")
        raise
    


    
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
import os
import logging
import re
import secrets
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API keys dataset
API_KEYS_FILE = "api_keys.csv"

# Generate and store API keys
def generate_api_key():
    return secrets.token_hex(32)

def store_api_key(api_key):
    """Store a unique API key in a CSV file."""
    if os.path.exists(API_KEYS_FILE):
        df = pd.read_csv(API_KEYS_FILE)
    else:
        df = pd.DataFrame(columns=["api_key"])
    
    new_entry = pd.DataFrame({"api_key": [api_key]})
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(API_KEYS_FILE, index=False)
    logger.info(f"Stored API key: {api_key}")

def validate_api_key(api_key):
    """Check if the API key is in the dataset."""
    if not os.path.exists(API_KEYS_FILE):
        return False
    
    df = pd.read_csv(API_KEYS_FILE)
    return api_key in df["api_key"].values

# Generate and store a new API key
new_key = generate_api_key()
store_api_key(new_key)

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        api_key = request.headers.get("X-API-Key")
        if not validate_api_key(api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return await call_next(request)

# Initialize FastAPI application with middleware
app = FastAPI()
app.add_middleware(APIKeyMiddleware)

# Paths to model and vectorizer files
COMMENT_MODEL_PATH = "fake_comment_model.pkl"
COMMENT_VECTORIZER_PATH = "vectorizer.pkl"
SPAM_MODEL_PATH = "emailmodel.pkl"
SPAM_VECTORIZER_PATH = "spam_vectorizer.pkl"
REVIEW_MODEL_PATH = "reviewmodel.pkl"
REVIEW_VECTORIZER_PATH = "review_vectorizer.pkl"

def load_model_and_vectorizer(model_path, vectorizer_path, model_name):
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        logger.error(f"{model_name} files missing")
        raise FileNotFoundError(f"{model_name} files missing")
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        logger.error(f"Error loading {model_name}: {str(e)}")
        raise RuntimeError(f"Failed to load {model_name}: {str(e)}")

# Load models
comment_model, comment_vectorizer = load_model_and_vectorizer(COMMENT_MODEL_PATH, COMMENT_VECTORIZER_PATH, "Comment Detection")
spam_model, spam_vectorizer = load_model_and_vectorizer(SPAM_MODEL_PATH, SPAM_VECTORIZER_PATH, "Spam Detection")
review_model, review_vectorizer = load_model_and_vectorizer(REVIEW_MODEL_PATH, REVIEW_VECTORIZER_PATH, "Review Detection")

class CommentRequest(BaseModel):
    comment: str

class CommentResponse(BaseModel):
    comment: str
    is_fake: bool

@app.post("/detect_fake_comment", response_model=CommentResponse)
async def detect_fake_comment(request: CommentRequest):
    try:
        transformed_comment = comment_vectorizer.transform([request.comment])
        prediction = comment_model.predict(transformed_comment)
        result = bool(prediction[0])
        return CommentResponse(comment=request.comment, is_fake=result)
    except Exception as e:
        logger.error(f"Error processing comment: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    
    
    
    
    
    import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
import os
import logging
import secrets
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Key Storage File
API_KEYS_FILE = "api_keys.csv"

# Generate and store API keys securely
def generate_api_key():
    return secrets.token_hex(32)

def store_api_key(api_key):
    """Store a unique API key in a CSV file."""
    if os.path.exists(API_KEYS_FILE):
        df = pd.read_csv(API_KEYS_FILE)
    else:
        df = pd.DataFrame(columns=["api_key"])

    if api_key not in df["api_key"].values:
        new_entry = pd.DataFrame({"api_key": [api_key]})
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(API_KEYS_FILE, index=False)
        logger.info(f"Stored API key: {api_key}")

def validate_api_key(api_key):
    """Check if the API key exists in the dataset."""
    if not os.path.exists(API_KEYS_FILE):
        return False

    df = pd.read_csv(API_KEYS_FILE)
    return api_key in df["api_key"].values

# Generate and store a new API key
new_key = generate_api_key()
store_api_key(new_key)

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        api_key = request.headers.get("X-API-Key")
        if not validate_api_key(api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return await call_next(request)

# Initialize FastAPI application with middleware
app = FastAPI()
app.add_middleware(APIKeyMiddleware)

# Paths to model and vectorizer files
REVIEW_MODEL_PATH = "reviewmodel.pkl"
REVIEW_VECTORIZER_PATH = "review_vectorizer.pkl"
REVIEW_DATASET_PATH = "review_dataset.csv"

# Ensure vectorizer retraining if misaligned
def retrain_vectorizer():
    if not os.path.exists(REVIEW_DATASET_PATH):
        logger.error("Review dataset missing. Cannot retrain vectorizer.")
        raise FileNotFoundError("Review dataset is required for retraining.")

    dataset = pd.read_csv(REVIEW_DATASET_PATH)

    vectorizer = TfidfVectorizer(max_features=1768)
    vectorizer.fit(dataset["review"])
    joblib.dump(vectorizer, REVIEW_VECTORIZER_PATH)

    logger.info(f"Vectorizer retrained with {len(vectorizer.get_feature_names_out())} features.")

# Load models and vectorizers
def load_model_and_vectorizer(model_path, vectorizer_path, model_name):
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        logger.warning(f"{model_name} files missing, attempting vectorizer retraining.")
        retrain_vectorizer()

    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

        # Verify alignment
        sample_text = ["Test review"]
        test_vectorized = vectorizer.transform(sample_text)
        expected_features = 1768

        if test_vectorized.shape[1] != expected_features:
            logger.error(f"{model_name} feature mismatch: Found {test_vectorized.shape[1]}, expected {expected_features}")
            retrain_vectorizer()
            vectorizer = joblib.load(vectorizer_path)

        return model, vectorizer

    except Exception as e:
        logger.error(f"Error loading {model_name}: {str(e)}")
        raise RuntimeError(f"Failed to load {model_name}: {str(e)}")

# Load models
review_model, review_vectorizer = load_model_and_vectorizer(REVIEW_MODEL_PATH, REVIEW_VECTORIZER_PATH, "Review Detection")

class ReviewRequest(BaseModel):
    review: str

class ReviewResponse(BaseModel):
    review: str
    is_fake: bool

@app.post("/detect_fake_review", response_model=ReviewResponse)
async def detect_fake_review(request: ReviewRequest):
    try:
        if not request.review.strip():
            raise HTTPException(status_code=400, detail="Review cannot be empty.")

        # Transform the review
        transformed_review = review_vectorizer.transform([request.review])

        # Validate feature count
        expected_features = 1768
        if transformed_review.shape[1] != expected_features:
            logger.error(f"Feature mismatch detected: Expected {expected_features}, Found {transformed_review.shape[1]}")
            retrain_vectorizer()
            transformed_review = review_vectorizer.transform([request.review])

        # Predict with probability check
        prediction_proba = review_model.predict_proba(transformed_review)[0]
        is_fake = prediction_proba[1] > 0.5  # Threshold at 50%

        logger.info(f"Review: {request.review} -> Predicted: {'Fake' if is_fake else 'Real'} with probability {prediction_proba[1]:.4f}")
        return ReviewResponse(review=request.review, is_fake=is_fake)

    except Exception as e:
        logger.error(f"Error processing review: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
    
    
    import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
import os
import logging
import secrets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Key Storage File
API_KEYS_FILE = "api_keys.csv"

# Generate and store API keys securely
def generate_api_key():
    return secrets.token_hex(32)

def store_api_key(api_key):
    """Store a unique API key in a CSV file."""
    if os.path.exists(API_KEYS_FILE):
        df = pd.read_csv(API_KEYS_FILE)
    else:
        df = pd.DataFrame(columns=["api_key"])

    if api_key not in df["api_key"].values:
        new_entry = pd.DataFrame({"api_key": [api_key]})
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(API_KEYS_FILE, index=False)
        logger.info(f"Stored API key: {api_key}")

def validate_api_key(api_key):
    """Check if the API key exists in the dataset."""
    if not os.path.exists(API_KEYS_FILE):
        return False

    df = pd.read_csv(API_KEYS_FILE)
    return api_key in df["api_key"].values

# Generate and store a new API key
new_key = generate_api_key()
store_api_key(new_key)

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        api_key = request.headers.get("X-API-Key")
        if not validate_api_key(api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return await call_next(request)

# Initialize FastAPI application with middleware
app = FastAPI()
app.add_middleware(APIKeyMiddleware)

# Paths to model and vectorizer files
REVIEW_MODEL_PATH = "reviewmodel.pkl"
REVIEW_VECTORIZER_PATH = "review_vectorizer.pkl"
REVIEW_DATASET_PATH = "review_dataset.csv"

# Ensure vectorizer retraining if misaligned
def retrain_vectorizer():
    if not os.path.exists(REVIEW_DATASET_PATH):
        logger.error("Review dataset missing. Cannot retrain vectorizer.")
        raise FileNotFoundError("Review dataset is required for retraining.")

    dataset = pd.read_csv(REVIEW_DATASET_PATH)

    vectorizer = TfidfVectorizer(max_features=1768)
    vectorizer.fit(dataset["review"])
    joblib.dump(vectorizer, REVIEW_VECTORIZER_PATH)

    logger.info(f"Vectorizer retrained with {len(vectorizer.get_feature_names_out())} features.")

# Retrain model
def retrain_model():
    if not os.path.exists(REVIEW_DATASET_PATH):
        logger.error("Review dataset missing. Cannot retrain model.")
        raise FileNotFoundError("Review dataset is required for model training.")

    dataset = pd.read_csv(REVIEW_DATASET_PATH)

    vectorizer = joblib.load(REVIEW_VECTORIZER_PATH)
    X = vectorizer.transform(dataset["review"])
    y = dataset["label"].astype(int)  # Ensure labels are numeric

    model = LogisticRegression()
    model.fit(X, y)

    joblib.dump(model, REVIEW_MODEL_PATH)
    logger.info("Model retrained successfully.")

# Load models and vectorizers
def load_model_and_vectorizer(model_path, vectorizer_path, model_name):
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        logger.warning(f"{model_name} files missing, attempting retraining.")
        retrain_vectorizer()
        retrain_model()

    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

        # Verify alignment
        sample_text = ["Test review"]
        test_vectorized = vectorizer.transform(sample_text)
        expected_features = 1768

        if test_vectorized.shape[1] != expected_features:
            logger.error(f"{model_name} feature mismatch: Found {test_vectorized.shape[1]}, expected {expected_features}")
            retrain_vectorizer()
            retrain_model()
            vectorizer = joblib.load(vectorizer_path)
            model = joblib.load(model_path)

        return model, vectorizer

    except Exception as e:
        logger.error(f"Error loading {model_name}: {str(e)}")
        raise RuntimeError(f"Failed to load {model_name}: {str(e)}")

# Load models
review_model, review_vectorizer = load_model_and_vectorizer(REVIEW_MODEL_PATH, REVIEW_VECTORIZER_PATH, "Review Detection")

class ReviewRequest(BaseModel):
    review: str

class ReviewResponse(BaseModel):
    review: str
    is_fake: bool

@app.post("/detect_fake_review", response_model=ReviewResponse)
async def detect_fake_review(request: ReviewRequest):
    try:
        if not request.review.strip():
            raise HTTPException(status_code=400, detail="Review cannot be empty.")

        # Transform the review
        transformed_review = review_vectorizer.transform([request.review])
        logger.info(f"Transformed review feature count: {transformed_review.shape[1]}")

        # Validate feature count
        expected_features = 1768
        if transformed_review.shape[1] != expected_features:
            logger.error(f"Feature mismatch detected: Expected {expected_features}, Found {transformed_review.shape[1]}")
            retrain_vectorizer()
            retrain_model()
            transformed_review = review_vectorizer.transform([request.review])

        # Predict with probability check
        prediction_proba = review_model.predict_proba(transformed_review)[0]
        is_fake = prediction_proba[1] > 0.5  # Threshold at 50%

        logger.info(f"Review: {request.review} -> Predicted: {'Fake' if is_fake else 'Real'} with probability {prediction_proba[1]:.4f}")
        return ReviewResponse(review=request.review, is_fake=is_fake)

    except Exception as e:
        logger.error(f"Error processing review: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)

# Ensure vectorizer retraining if misaligned
def retrain_vectorizer():
    if not os.path.exists(REVIEW_DATASET_PATH):
        logger.error("Review dataset missing. Cannot retrain vectorizer.")
        raise FileNotFoundError("Review dataset is required for retraining.")

    dataset = pd.read_csv(REVIEW_DATASET_PATH)


    
    
    

except Exception as e:
logger.error(f"Error loading {model_name}: {str(e)}")
raise RuntimeError(f"Failed to load {model_name}: {str(e)}")


import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
import os
import logging
import secrets
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Key Storage File
API_KEYS_FILE = "api_keys.csv"

# Generate and store API keys securely
def generate_api_key():
    return secrets.token_hex(32)

def store_api_key(api_key):
    """Store a unique API key in a CSV file."""
    if os.path.exists(API_KEYS_FILE):
        df = pd.read_csv(API_KEYS_FILE)
    else:
        df = pd.DataFrame(columns=["api_key"])

    if api_key not in df["api_key"].values:
        new_entry = pd.DataFrame({"api_key": [api_key]})
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(API_KEYS_FILE, index=False)
        logger.info(f"Stored API key: {api_key}")

def validate_api_key(api_key):
    """Check if the API key exists in the dataset."""
    if not os.path.exists(API_KEYS_FILE):
        return False

    df = pd.read_csv(API_KEYS_FILE)
    return api_key in df["api_key"].values

# Generate and store a new API key
new_key = generate_api_key()
store_api_key(new_key)

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        api_key = request.headers.get("X-API-Key")
        if not validate_api_key(api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return await call_next(request)

# Initialize FastAPI application with middleware
app = FastAPI()
app.add_middleware(APIKeyMiddleware)

# Paths to model and vectorizer files
REVIEW_MODEL_PATH = "reviewmodel.pkl"
REVIEW_VECTORIZER_PATH = "review_vectorizer.pkl"
REVIEW_DATASET_PATH = "review_dataset.csv"

# Clean dataset to remove NaN values
dataset = pd.read_csv(REVIEW_DATASET_PATH)
dataset = dataset.dropna(subset=["review"])
dataset["review"] = dataset["review"].astype(str)
dataset.to_csv(REVIEW_DATASET_PATH, index=False)
logger.info("Dataset cleaned successfully!")

# Ensure vectorizer retraining if misaligned
def retrain_vectorizer():
    if not os.path.exists(REVIEW_DATASET_PATH):
        logger.error("Review dataset missing. Cannot retrain vectorizer.")
        raise FileNotFoundError("Review dataset is required for retraining.")

    dataset = pd.read_csv(REVIEW_DATASET_PATH)

    vectorizer = TfidfVectorizer(max_features=1768)
    vectorizer.fit(dataset["review"])
    joblib.dump(vectorizer, REVIEW_VECTORIZER_PATH)

    logger.info(f"Vectorizer retrained with {len(vectorizer.get_feature_names_out())} features.")

# Load models and vectorizer
def load_model_and_vectorizer(model_path, vectorizer_path, model_name):
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        logger.warning(f"{model_name} vectorizer missing, attempting retraining.")
        retrain_vectorizer()

    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

        # Verify alignment
        sample_text = ["Test review"]
        test_vectorized = vectorizer.transform(sample_text)
        expected_features = 1768

        if test_vectorized.shape[1] != expected_features:
            logger.error(f"{model_name} feature mismatch: Found {test_vectorized.shape[1]}, expected {expected_features}")
            retrain_vectorizer()
            vectorizer = joblib.load(vectorizer_path)

        return model, vectorizer

    except Exception as e:
        logger.error(f"Error loading {model_name}: {str(e)}")
        raise RuntimeError(f"Failed to load {model_name}: {str(e)}")

# Load models
review_model, review_vectorizer = load_model_and_vectorizer(REVIEW_MODEL_PATH, REVIEW_VECTORIZER_PATH, "Review Detection")

class ReviewRequest(BaseModel):
    review: str

class ReviewResponse(BaseModel):
    review: str
    is_fake: bool

@app.post("/detect_fake_review", response_model=ReviewResponse)
async def detect_fake_review(request: ReviewRequest):
    try:
        if not request.review.strip():
            raise HTTPException(status_code=400, detail="Review cannot be empty.")

        # Transform the review
        transformed_review = review_vectorizer.transform([request.review])
        logger.info(f"Transformed review feature count: {transformed_review.shape[1]}")

        # Validate feature count
        expected_features = 1768
        if transformed_review.shape[1] != expected_features:
            logger.error(f"Feature mismatch detected: Expected {expected_features}, Found {transformed_review.shape[1]}")
            retrain_vectorizer()
            transformed_review = review_vectorizer.transform([request.review])

        # Predict with probability check
        prediction_proba = review_model.predict_proba(transformed_review)[0]
        is_fake = prediction_proba[1] > 0.5  # Threshold at 50%

        logger.info(f"Review: {request.review} -> Predicted: {'Fake' if is_fake else 'Real'} with probability {prediction_proba[1]:.4f}")
        return ReviewResponse(review=request.review, is_fake=is_fake)

    except Exception as e:
        logger.error(f"Error processing review: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)