# TruthLens: Fake News Detection System

## 1. Aim
To develop a machine-learning web application that analyses submitted news text and predicts whether it is **REAL** or **FAKE**.

## 2. Objectives
- Build a text-classification model using the WELFake dataset.
- Convert news text into numerical features using TF-IDF.
- Train a Logistic Regression classifier to predict the dataset label.
- Integrate the trained model into a Flask web application.
- Evaluate model performance using accuracy, precision, recall, F1-score, and a confusion matrix.

## 3. Technologies Used
- **Python** — application and machine-learning implementation
- **Flask** — web server and prediction endpoint
- **Pandas** — dataset loading and preprocessing
- **scikit-learn** — TF-IDF vectorization, Logistic Regression, train/test split, and evaluation metrics
- **Joblib** — saving and loading the trained model and vectorizer
- **HTML and CSS** — web interface
- **Requests** — HTTP requests for the optional Google Fact Check API integration

## 4. Dataset
**Dataset:** WELFake Dataset (`WELFake_Dataset.csv`)

The dataset contains news headlines, article text, and binary labels:
- `0` = REAL
- `1` = FAKE

### Preprocessing workflow
1. Load the CSV dataset using Pandas.
2. Remove the unnecessary `Unnamed: 0` column, if present.
3. Remove rows with missing title, article text, or label.
4. Convert labels to integers and retain only labels `0` and `1`.
5. Combine the headline and article text into one text field.
6. Remove duplicate combined articles.
7. Remove entries whose combined text is shorter than 20 characters.
8. Split the cleaned data into 80% training data and 20% testing data, using stratification to preserve the class proportions.

### Dataset statistics
The evaluation script `evaluate_welfake_proper_new.py` previously reported these cleaned-data statistics:

| Measure | Count |
|---|---:|
| Total cleaned articles | 63,114 |
| REAL (label 0) | 34,790 |
| FAKE (label 1) | 28,324 |
| Training split (80%) | 50,491 |
| Testing split (20%) | 12,623 |

These are the statistics printed by the evaluation script for the version of the dataset used during testing. They may differ if the CSV file changes.

## 5. Model and Evaluation
The project uses **TF-IDF** to represent news text as numerical features and **Logistic Regression** for binary classification.

The vectorizer configuration used by the training/evaluation workflow includes:
- `max_features=10000`
- `ngram_range=(1, 1)`
- `min_df=3`
- `max_df=0.98`
- `sublinear_tf=True`
- `stop_words="english"`

The fresh train/test evaluation previously reported **95.33% accuracy**. This is the result from `evaluate_welfake_proper_new.py`'s separately trained evaluation model on its held-out split; it is not a guarantee that every real-world article will be classified correctly, nor should it be described as the exact live accuracy of the saved web model.

Evaluation measures:
- **Accuracy:** fraction of test examples predicted correctly.
- **Precision:** proportion of predictions for a class that are correct.
- **Recall:** proportion of actual examples of a class correctly detected.
- **F1-score:** harmonic mean of precision and recall.
- **Confusion matrix:** counts correct and incorrect predictions by actual and predicted class.

## 6. Overall Workflow
1. The user enters a headline and/or article in the TruthLens interface.
2. Flask receives the submitted text at the `/predict` endpoint.
3. The saved TF-IDF vectorizer transforms the text into numerical features.
4. The saved Logistic Regression model predicts label `0` or `1`.
5. The application maps label `0` to REAL and label `1` to FAKE.
6. The interface displays the prediction.
7. If `GOOGLE_FACT_CHECK_API_KEY` is configured, the backend can also query the Google Fact Check Tools API for related claim reviews. This integration is optional and separate from the machine-learning prediction.

## 7. Project Files
- `news_detect_app_fixed.py` — Flask application, prediction logic, class mapping, and optional fact-check API request.
- `templates/index.html` — main webpage and prediction interface.
- `static/style.css` — webpage styling.
- `pkl_files/fake_news_welfake_model.pkl` — saved trained classifier loaded by the application.
- `pkl_files/welfake_tfidf_vectorizer.pkl` — saved TF-IDF vectorizer paired with the classifier.
- `WELFake_Dataset.csv` — source dataset used by training and evaluation scripts.
- `welfake_train_test_eval/train_truthlens_welfake.py` — trains the classifier and saves the model and vectorizer.
- `welfake_train_test_eval/test_welfake_model.py` — tests a headline/article using the saved model and prints a prediction.
- `welfake_train_test_eval/evaluate_welfake_proper_new.py` — trains a separate evaluation model on a train/test split and prints evaluation metrics. It does not overwrite the model used by the website.
- `requirements.txt` — Python dependencies.

## 8. Commands to Run the Project and View Model Statistics

Run these commands in the terminal from the main `TruthLens` project directory.

### Start the web application
```powershell
python news_detect_app_fixed.py
```
Then open `http://127.0.0.1:5000` in a browser.

### Display evaluation statistics
```powershell
python welfake_train_test_eval/evaluate_welfake_proper_new.py
```
This prints the dataset counts, training/testing split sizes, accuracy, classification report, and confusion matrix. **It trains a fresh evaluation model each time**, so these are evaluation statistics for that run, not live monitoring statistics from the already-saved website model.

### Test a news item with the saved model
```powershell
python welfake_train_test_eval/test_welfake_model.py
```
Enter the headline and article text when prompted.

### Retrain and save the website model
```powershell
python welfake_train_test_eval/train_truthlens_welfake.py
```
Run this only when you intend to replace the saved model and vectorizer. Confirm that the script saves them into `pkl_files/`; otherwise update the save paths in the script to match the application's paths.

## 9. Important Note
TruthLens is a machine-learning classifier, not a definitive fact-checking authority. It can misclassify genuine or false articles, especially news that differs from its training data. A prediction should not be treated as proof that a claim is true or false.
