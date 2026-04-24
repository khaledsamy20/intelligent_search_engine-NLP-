# ehna hena bn3ml create l web application b streamlit 34an n3rf el model bta3na
# ehna hna5od el functions mn foo2 3alatol
import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def text_preprocessing(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = nltk.word_tokenize(text)
    text = [t for t in text if t not in stop_words]
    text = [lemmatizer.lemmatize(t) for t in text]
    return " ".join(text)

@st.cache_resource
def load_data_and_models():
    df = pd.read_pickle('data.pkl')
    vec = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))
    bert = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = pickle.load(open('sentence_embeddings.pkl', 'rb'))
    return df, vec, matrix, bert, embeddings

data, tfidf, tfidf_matrix, bert_model, sentence_embeddings = load_data_and_models()

def search_tfidf(query, k=5):
    processed_query = text_preprocessing(query)
    query_vec = tfidf.transform([processed_query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_k_idx = similarities.argsort()[-k:][::-1]
    result = data.iloc[top_k_idx][['Review Title', 'Review Text']].copy()
    result['Similarity Score'] = similarities[top_k_idx]
    return result

def search_bert(query, k=5):
    query_vec = bert_model.encode([query])
    similarities = cosine_similarity(query_vec, sentence_embeddings).flatten()
    top_k_idx = similarities.argsort()[-k:][::-1]
    result = data.iloc[top_k_idx][['Review Title', 'Review Text']].copy()
    result['Similarity Score'] = similarities[top_k_idx]
    return result

st.title("Intelligent Search Engine")

model_choice = st.radio("Choose Model:", ("TF-IDF", "BERT"))

query = st.text_input("Enter your search query:")

if st.button("Search"):
    if query:
        if model_choice == "TF-IDF":
            results = search_tfidf(query)
        else:
            results = search_bert(query)
        
        for _, row in results.iterrows():
            st.subheader(row['Review Title'])
            st.write(row['Review Text'])
            st.write(f"Similarity Score: {row['Similarity Score']:.4f}")
            st.divider()
