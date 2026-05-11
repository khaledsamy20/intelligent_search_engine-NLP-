# 🔍 Intelligent Search Engine: NLP Product Review Search

**Course:** Natural Language Processing  
**University:** Capital University (Faculty of Computing & Artificial Intelligence)  

## 📌 Project Overview
This project implements an **Intelligent Search Engine** for e-commerce product reviews and customer complaints. It compares traditional keyword-based search with intermediate and advanced semantic search models. 

When a user searches for a query (e.g., *"delivery problem"*), the search engine retrieves the most relevant reviews. The project demonstrates why modern AI models (Transformers/BERT) provide vastly superior search experiences compared to legacy statistical models (TF-IDF) and static embeddings (Word2Vec) by understanding the actual *meaning* and *context* of a user's query rather than just exact word matches.

## ✨ Features
* **Full NLP Preprocessing Pipeline:** Lowercasing, punctuation removal, tokenization, stop-word removal, and lemmatization using NLTK.
* **Lightning-Fast Query Correction:** Implements SymSpell to instantly fix user typos before processing the search (e.g., correcting "delivry problm" to "delivery problem").
* **Baseline Model (TF-IDF):** Uses Term Frequency-Inverse Document Frequency combined with Cosine Similarity for exact-keyword matching.
* **Intermediate Model (Word2Vec):** Uses Gensim to train custom Word2Vec embeddings on the dataset, calculating the average word vector per document to capture synonyms.
* **Advanced Model (BERT):** Uses the `all-MiniLM-L6-v2` Sentence Transformer model to generate dense semantic embeddings, capturing synonyms, paraphrases, and full sentence context.
* **Evaluation Metrics:** Includes Precision@k (Precision@5) and execution time analysis with visual comparisons via Matplotlib and Seaborn.
* **Interactive Web UI:** A user-friendly web interface built with Streamlit to test the models side-by-side in real-time.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **NLP & ML Libraries:** `nltk`, `scikit-learn`, `sentence-transformers` (HuggingFace), `gensim`, `symspellpy`
* **Data Manipulation:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Web Framework:** `streamlit`

## 📊 Dataset
The dataset (`Amazon_Reviews.csv`) consists of over 21,000 e-commerce product reviews and complaints. It covers a wide range of common customer service topics such as delivery delays, product quality, payment errors, and refund issues. 

## 🏆 Evaluation & Results (TF-IDF vs. Word2Vec vs. BERT)
We evaluated the Top-5 retrieved documents (Precision@5) using manual relevance checking across several complex queries:

| Query | TF-IDF Precision@5 | Word2Vec Precision@5 | BERT Precision@5 | Winner |
| :--- | :---: | :---: | :---: | :---: |
| *delivery problem* | 0.20 | 0.20 | 1.00 | 🟢 BERT |
| *item quality issue* | 0.00 | 0.20 | 0.80 | 🟢 BERT |
| *cannot get help from support* | 0.40 | 0.40 | 0.60 | 🟢 BERT |
| *payment charge error* | 0.80 | 0.80 | 1.00 | 🟢 BERT |
| **Overall Average** | **0.35** | **0.40** | **0.85** | 🟢 **BERT** |

### Key Takeaways:
1.  **Context & Synonyms:** BERT easily understands that *"Double charged my card"* means the exact same thing as *"payment charge error"*. TF-IDF completely misses this because the words don't match exactly.
2.  **The Limits of Word2Vec:** Word2Vec performs slightly better than TF-IDF because it understands basic synonyms, but it fails to understand full sentence grammar and context since it only averages individual words.
3.  **Handling Negation:** For queries like *"cannot get help"*, TF-IDF matches the word "help" and returns positive reviews (*"Always there to help"*). BERT understands the negative intent and retrieves actual customer complaints.
4.  **Speed vs. Accuracy:** While TF-IDF and Word2Vec are faster, BERT provides a fundamentally better user experience by retrieving highly accurate, semantically relevant results.
---

## 🚀 How to Run the Project

### 1. Install Dependencies
Open your terminal or command prompt and install the required Python libraries:
```bash
pip install pandas numpy nltk scikit-learn sentence-transformers matplotlib seaborn streamlit
```

go to folder from CLI then run 
like: 
```bash
cd /d D:\khaled\projects\NLP_ intelligent search engine
streamlit run ISE_application.py

