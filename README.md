# Program 3 — IMDB Sentiment Analysis (TF-IDF, Doc2Vec, Logistic Regression, Random Forest)

NLP text classification pipeline on the IMDB Movie Review dataset. Compares two text representation strategies (TF-IDF and Doc2Vec) combined with Logistic Regression and Random Forest classifiers for binary sentiment prediction.

---

## Overview

This notebook implements a full NLP preprocessing and classification workflow for sentiment analysis. Raw movie reviews are cleaned, tokenized, and represented as feature vectors before being passed to supervised classifiers. 

---

## Dataset

**File:** `IMDB Dataset.csv`

50,000 movie reviews labeled `positive` or `negative` (balanced classes).

---

## Pipeline

### 1. Data Exploration (EDA)

- Check shape, columns, missing values, duplicate entries
- Inspect class distribution (`positive` vs `negative`)
- Analyze review length distribution
- Examine raw review text for noise (HTML tags, punctuation)

### 2. Train/Test Split

80/20 split performed **before any fit-based preprocessing** to prevent data leakage.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 3. Text Preprocessing

Applied to training data; transformations replayed on test data:

- Lowercase conversion
- HTML tag removal (regex)
- Punctuation and special character removal
- Stop word removal (`nltk.corpus.stopwords`)
- Tokenization (`nltk.word_tokenize`)
- Stemming (`PorterStemmer`) and Lemmatization (`WordNetLemmatizer`)
- POS tagging for lemmatization context (`averaged_perceptron_tagger`)

### 4. Feature Representations

| Method | Library | Description |
|--------|---------|-------------|
| TF-IDF | `sklearn.feature_extraction.text.TfidfVectorizer` | Sparse bag-of-words weighted by inverse document frequency |
| Doc2Vec | `gensim.models.doc2vec.Doc2Vec` | Dense paragraph-level embeddings |

### 5. Classification Models

| Model | Representation |
|-------|---------------|
| Logistic Regression | TF-IDF |
| Random Forest | TF-IDF |
| Logistic Regression | Doc2Vec |
| Random Forest | Doc2Vec |

### 6. Evaluation

Each combination evaluated on:
- Accuracy
- Precision
- Recall
- F1-score (macro)

---

## Key Libraries

```bash
pip install pandas nltk spacy gensim scikit-learn
python -m nltk.downloader stopwords punkt wordnet averaged_perceptron_tagger
python -m spacy download en_core_web_sm
```

## Usage

```bash
# Place IMDB Dataset.csv in the working directory
jupyter notebook Program3.ipynb
```

---

