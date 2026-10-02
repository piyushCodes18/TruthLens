# 🔎 TruthLens

**TruthLens** is a simple machine-learning based fake news detection project built as a BTech CSE 3rd-semester project.

It takes a news article as input and classifies it as **REAL** or **FAKE** using a **Bag of Words (BoW)** text representation and a **Logistic Regression** classifier.

The project includes a lightweight Flask backend and a cartoon-style HTML/CSS/JavaScript frontend.

---

## ✨ Features

- 📰 Enter or paste a news article
- 🤖 Classify the article as **REAL** or **FAKE**
- 📚 Built-in News Library containing actual articles from the dataset
- 🎨 Interactive cartoon-style user interface
- 🔴 Red visual effect for FAKE predictions
- 🟢 Green visual effect for REAL predictions
- ⏳ Loading animation while the model processes the article
- 🧹 Clear button for resetting the input
- 📱 Responsive frontend design
- 🐍 Flask backend for connecting the frontend with the ML model

---

## 🧠 Machine Learning Approach

The project uses the following workflow:

```text
News Dataset
     ↓
Data Cleaning
     ↓
Combine Title + Article Text
     ↓
Train/Test Split
     ↓
Bag of Words (CountVectorizer)
     ↓
Logistic Regression
     ↓
Prediction
     ↓
REAL / FAKE
```

### 1. Data Cleaning

The news text is cleaned before being processed by the model.

The preprocessing includes:

- Converting text to lowercase
- Removing URLs
- Removing HTML tags
- Removing punctuation
- Removing extra spaces
- Removing empty articles
- Removing duplicate records

The Flask application applies the same basic text-cleaning process before prediction. The backend then transforms the cleaned text using the saved vectorizer and sends it to the saved model. 

### 2. Feature Extraction

TruthLens uses **Bag of Words (BoW)** with `CountVectorizer`.

```python
CountVectorizer(
    max_features=5000,
    stop_words="english"
)
```

The model therefore works with numerical word-frequency features rather than raw text.

### 3. Classification

The classifier used is **Logistic Regression**.

```python
LogisticRegression(max_iter=1000)
```

---

## 📊 Model Performance

The model was evaluated on a test set containing **20% of the dataset**.

### Accuracy

**99.67%**

### Confusion Matrix

| Actual / Predicted | FAKE | REAL |
|---|---:|---:|
| **FAKE** | 3516 | 7 |
| **REAL** | 15 | 3167 |

The model therefore performed very well on the held-out test split used during development.

> **Note:** This accuracy is specific to the dataset and train/test split used during development. It should not be interpreted as proof that the model can determine the factual truth of every new article on the internet.

---

## 🖥️ Application Architecture

```text
                 ┌─────────────────────┐
                 │      User Input     │
                 │    News Article     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   HTML / CSS / JS   │
                 │     Frontend        │
                 └──────────┬──────────┘
                            │
                     POST /predict
                            │
                            ▼
                 ┌─────────────────────┐
                 │    Flask Backend    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    Text Cleaning    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   BoW Vectorizer    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Logistic Regression │
                 │       Model         │
                 └──────────┬──────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  REAL / FAKE  │
                    └───────────────┘
```

The Flask backend exposes a `/predict` POST endpoint that receives the article text, cleans it, transforms it with the saved BoW vectorizer, and obtains the prediction from the saved Logistic Regression model. 

---

## 📁 Project Structure

```text
TruthLens/
│
├── news_detect_app.py
├── fake_news_model.pkl
├── bow_vectorizer.pkl
├── requirements.txt
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── README.md
```

### File Description

| File | Purpose |
|---|---|
| `news_detect_app.py` | Flask backend and prediction API |
| `fake_news_model.pkl` | Trained Logistic Regression model |
| `bow_vectorizer.pkl` | Saved Bag of Words vectorizer |
| `templates/index.html` | Main frontend interface |
| `static/style.css` | Frontend styling and animations |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

---

## 🛠️ Technologies Used

### Programming & Backend
- Python
- Flask

### Machine Learning
- Scikit-learn
- Logistic Regression
- CountVectorizer / Bag of Words

### Data Processing
- Pandas
- Regular Expressions

### Frontend
- HTML5
- CSS3
- JavaScript
- Google Fonts

### Development Tools
- Google Colab
- Visual Studio Code
- GitHub

---

## 📚 Dataset

The project was trained using the **Fake and Real News Dataset**, containing separate collections of fake and real news articles.

The dataset records contain fields such as:

- `title`
- `text`
- `subject`
- `date`

The News Library in the frontend uses actual records from the supplied dataset rather than invented demonstration paragraphs.

---

## 🚀 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/TruthLens.git
cd TruthLens
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Flask application

```bash
python news_detect_app.py
```

### 4. Open the application

Open the local address shown by Flask, normally:

```text
http://127.0.0.1:5000
```

---

## 📦 Required Dependencies

A basic `requirements.txt` can contain:

```text
Flask
joblib
scikit-learn==1.6.1
```

If the project environment also requires the data-analysis libraries used during model development:

```text
pandas
numpy
```

---

## 🔬 Model Development

The model development process was carried out in Google Colab.

The main steps were:

1. Load the fake and real news datasets.
2. Assign labels to the two classes.
3. Combine the datasets.
4. Clean the article data.
5. Combine article title and text.
6. Split the data into training and testing sets.
7. Convert text into Bag of Words features.
8. Train Logistic Regression.
9. Evaluate the model.
10. Save the trained model and vectorizer as `.pkl` files.
11. Connect the saved model to the Flask application.

---

## ⚠️ Limitations

TruthLens is a **machine-learning text classifier**, not a complete fact-checking system.

The prediction is based on patterns learned from the training dataset. Therefore:

- A real article can potentially be classified as FAKE.
- A fake article can potentially be classified as REAL.
- The model does not independently verify claims against authoritative sources.
- Performance on modern or completely different news sources may differ from the development test results.
- The reported accuracy is based on the particular dataset and test split used for this project.

For this reason, a prediction should be treated as a **model classification**, not as definitive proof that a news article is true or false.

---

## 🎯 Project Objective

The main objective of TruthLens is to demonstrate how **Natural Language Processing and Machine Learning** can be used to classify news articles into fake and real categories.

The project also demonstrates the integration of:

```text
Machine Learning
       +
Natural Language Processing
       +
Flask
       +
HTML / CSS / JavaScript
```

into a simple working application.

---

## 👨‍💻 Project

**TruthLens — Fake News Detection System**

Built as a BTech CSE 3rd-semester machine learning project.

---

## 📌 Future Improvements

Possible future improvements include:

- Training with a larger and more diverse dataset
- Comparing multiple classification algorithms
- Using TF-IDF or word embeddings
- Adding a confidence/probability display
- Integrating live news sources
- Adding claim verification using external reliable sources
- Improving multilingual news detection
- Deploying the Flask application online
