import os
import streamlit as st
import re
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Ensure NLTK stopwords are available and cache them
try:
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('english'))

# Load data (add simple error handling)
DATA_PATH = 'Fake.csv'
if not os.path.exists(DATA_PATH):
    st.error(f"Data file not found: {DATA_PATH}")
    st.stop()

news_df = pd.read_csv(DATA_PATH)
news_df = news_df.fillna(' ')

# Create a content field used for modelling
news_df['content'] = news_df['author'].astype(str) + ' ' + news_df['title'].astype(str)

# Define keywords and create labels:
# 1 => Rumour (e.g., 'fake', 'conspiracy', ...) ; 0 => Not Rumour
keywords = ['bias', 'conspiracy', 'fake', 'bs', 'hate']
# Normalize the 'type' column to lowercase string before checking
news_df['type'] = news_df['type'].astype(str).str.lower()
# Use substring match so values like 'fake' or 'fake news' are included
news_df['label'] = news_df['type'].apply(lambda t: 1 if any(k in t for k in keywords) else 0)

# Stemming / cleaning function (uses cached stop words)
ps = PorterStemmer()
def clean_and_stem(text: str) -> str:
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in STOP_WORDS]
    return ' '.join(text)

# Apply cleaning/stemming to the content column
news_df['content'] = news_df['content'].apply(clean_and_stem)

# Vectorize
X_texts = news_df['content'].values
y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(X_texts)
X = vector.transform(X_texts)

# Split and train
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Evaluate and show accuracy (optional but useful)
y_pred = model.predict(X_test)
acc = accuracy_score(Y_test, y_pred)

# Streamlit UI
st.title('Rumour Detector')
st.write(f"Model test accuracy: {acc:.4f}")

input_text = st.text_input('Enter the news')

def predict_input(text: str) -> int:
    # Apply same preprocessing as training data
    cleaned = clean_and_stem(text)
    input_vec = vector.transform([cleaned])
    pred = model.predict(input_vec)
    return int(pred[0])

if input_text:
    pred = predict_input(input_text)
    if pred == 1:
        st.write('It is a Rumour')
    else:
        st.write('It is not a Rumour')
