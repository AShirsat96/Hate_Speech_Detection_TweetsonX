# Hate Speech Detection in Tweets

## Project Overview

This project implements a machine learning pipeline to detect hate speech in tweets. The solution classifies tweets as either hate speech (racist/sexist content) or non-hate speech using natural language processing techniques and machine learning algorithms.

## Problem Statement

The task is to detect hate speech in tweets. For this project, a tweet is considered to contain hate speech if it has racist or sexist sentiment associated with it. The goal is to classify tweets into two categories:
- Label 1: Tweet contains hate speech (racist/sexist content)
- Label 0: Tweet does not contain hate speech

## Dataset

The project uses two datasets:
- `train_E6oV3lV.csv`: Training dataset with 31,962 labeled tweets
  - Columns: id, label, tweet
- `test_tweets_anuFYb8.csv`: Test dataset with 17,197 unlabeled tweets
  - Columns: id, tweet

## Requirements

```
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.2.2
nltk==3.8.1
matplotlib==3.7.1
seaborn==0.12.2
xgboost==1.7.3
wordcloud==1.9.2
```

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Download NLTK resources (this is handled in the code, but can be done manually):
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('punkt')
   ```

## Usage

### Running the Complete Pipeline

```bash
python hate_speech_detection.py
```

The script will:
1. Load and preprocess the data
2. Extract features using TF-IDF
3. Train multiple models and compare them
4. Perform model fine-tuning
5. Create an ensemble model
6. Generate predictions for the test set
7. Save feature importance visualizations

### Output Files

- `hate_speech_predictions.csv`: Predictions for the test dataset
- `tweet_length_distribution.png`: Visualization of tweet lengths by class
- `wordcloud_class_0.png`: Word cloud for non-hate speech tweets
- `wordcloud_class_1.png`: Word cloud for hate speech tweets
- `model_comparison.png`: F1 scores for all models
- `feature_importance.png` or `feature_coefficients.png`: Visualization of feature importance

## Pipeline Components

### 1. Data Preprocessing

- Text cleaning: URL removal, mention removal, hashtag removal, punctuation removal
- Text normalization: lowercasing, whitespace standardization
- NLP processing: tokenization, stopword removal, lemmatization (with fallbacks if NLTK is unavailable)

### 2. Feature Extraction

- TF-IDF Vectorization with n-grams (unigrams and bigrams)
- Parameters: max_features=10000, min_df=5, max_df=0.8, ngram_range=(1, 2)

### 3. Model Training and Evaluation

Four different algorithms are trained and compared:
- Logistic Regression: Linear model with L2 regularization
- Random Forest: Ensemble of decision trees
- Support Vector Machine (SVM): Linear kernel
- XGBoost: Gradient boosting framework

### 4. Model Tuning

Grid search is performed on the Random Forest model to find the optimal hyperparameters.

### 5. Ensemble Model

A voting classifier combines the predictions from all models using soft voting.

### 6. Feature Importance Analysis

Detailed analysis of the most important features (words/phrases) that contribute to classification:
- For tree-based models: Feature importance scores
- For linear models: Coefficient values
- Visualizations and rankings of top features

## Feature Importance Analysis

The feature importance analysis is implemented in a separate module that:
1. Extracts importance values from the trained model
2. Creates appropriate visualizations based on model type
3. Ranks and displays the most influential features
4. Offers additional analysis through a DataFrame

To use the feature importance analysis separately:

```python
from hate_speech_feature_analysis import analyze_feature_importance, create_feature_importance_df

# After training your model
importance_results = analyze_feature_importance(final_model, tfidf_vectorizer)
importance_df = create_feature_importance_df(final_model, tfidf_vectorizer)

# Save results
importance_df.to_csv('feature_importance.csv', index=False)
```

## Evaluation Metric

The model performance is evaluated using the F1 score, which is the harmonic mean of precision and recall. This metric is particularly useful for imbalanced classification problems.

## Error Handling

The code includes robust error handling to manage potential issues with:
- NLTK resource availability
- Data loading and processing
- Model training and evaluation
- Visualization creation

## Customization

To adapt this code for your own hate speech detection task:
1. Replace the input data files with your own labeled datasets
2. Adjust the preprocessing steps if needed for your specific text data
3. Modify the hyperparameters or add new models to the comparison
4. Change the evaluation metrics if F1 score is not appropriate for your case
