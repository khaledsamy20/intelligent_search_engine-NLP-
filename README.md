# 🔍 Intelligent Search Engine: NLP Product Review Search

**Course:** Natural Language Processing  
**University:** Capital University (Faculty of Computing & Artificial Intelligence)  

## 📌 Project Overview
This project implements an **Intelligent Search Engine** for e-commerce product reviews and customer complaints. It compares traditional keyword-based search with advanced semantic search. 

When a user searches for a query (e.g., *"delivery problem"*), the search engine retrieves the most relevant reviews. The project demonstrates why modern AI models (Transformers/BERT) provide vastly superior search experiences compared to legacy statistical models (TF-IDF) by understanding the actual *meaning* and *context* of a user's query rather than just exact word matches.

## ✨ Features
* **Full NLP Preprocessing Pipeline:** Lowercasing, punctuation removal, tokenization, stop-word removal, and lemmatization using NLTK.
* **Baseline Model (TF-IDF):** Uses Term Frequency-Inverse Document Frequency combined with Cosine Similarity for exact-keyword matching.
* **Advanced Model (BERT):** Uses the `all-MiniLM-L6-v2` Sentence Transformer model to generate dense semantic embeddings, capturing synonyms, paraphrases, and context.
* **Evaluation Metrics:** Includes Precision@k (Precision@5) and execution time analysis with visual comparisons via Matplotlib and Seaborn.
* **Interactive Web UI:** A user-friendly web interface built with Streamlit to test both models side-by-side in real-time.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **NLP & ML Libraries:** `nltk`, `scikit-learn`, `sentence-transformers` (HuggingFace)
* **Data Manipulation:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Web Framework:** `streamlit`

## 📊 Dataset
The dataset (`Amazon_Reviews.csv`) consists of over 21,000 e-commerce product reviews and complaints. It covers a wide range of common customer service topics such as delivery delays, product quality, payment errors, and refund issues. 

## 🏆 Evaluation & Results (TF-IDF vs. BERT)
We evaluated the Top-5 retrieved documents (Precision@5) using manual relevance checking across several complex queries:

| Query | TF-IDF Precision@5 | BERT Precision@5 | Winner |
| :--- | :---: | :---: | :---: |
| *delivery problem* | 0.20 | 1.00 | 🟢 BERT |
| *item quality issue* | 0.00 | 0.80 | 🟢 BERT |
| *cannot get help from support* | 0.40 | 0.60 | 🟢 BERT |
| *payment charge error* | 0.60 | 1.00 | 🟢 BERT |

### Key Takeaways:
1.  **Context & Synonyms:** BERT easily understands that *"Double charged my card"* means the exact same thing as *"payment charge error"*. TF-IDF completely misses this because the words don't match exactly.
2.  **Handling Negation:** For queries like *"cannot get help"*, TF-IDF matches the word "help" and returns positive reviews (*"Always there to help"*). BERT understands the negative intent and retrieves actual customer complaints.
3.  **Speed vs. Accuracy:** While TF-IDF is slightly faster (sparse matrix multiplication), BERT provides a fundamentally better user experience by retrieving semantically relevant results.

---

## 🚀 How to Run the Project

### 1. Install Dependencies
Open your terminal or command prompt and install the required Python libraries:
```bash
pip install pandas numpy nltk scikit-learn sentence-transformers matplotlib seaborn streamlit

go to folder from CLI then run 
streamlit run ISP_application.py"# intelligent_search_engine-NLP-" 
