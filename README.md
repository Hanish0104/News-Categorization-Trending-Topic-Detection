AI News Categorization and Trending Topic Detection

A machine learning project that automatically classifies news articles into categories and detects trending topics using NLP techniques on the Huffington Post News Category Dataset.

Author
Venkata Hanish Talluri

Project Overview
Objective
Develop machine learning models that accurately:

Predict the category of a news article based on its date, headline, and short description.
Detect trending topics from news content in real time.

Motivation

Manual categorization of online news is inefficient at scale.
Automated classification improves content delivery, searchability, and user engagement.
Trending topic detection can make news applications more dynamic and responsive in real time.

Key Research Question

Can we predict a news article's category using only its headline and description?


Dataset

Source: Huffington Post News Category Dataset – Kaggle
Size: ~200,000 labeled entries
Features: Headlines and short descriptions across multiple news categories


Tools & Libraries
CategoryToolsEnvironmentGoogle Colab (Jupyter Notebook)LanguagePythonData ProcessingPandas, NLTKVisualizationMatplotlibML / ModelingScikit-learn, XGBoost, MLP (Neural Network)Text RepresentationTF-IDF Vectorizer, Sentence Transformers (all-MiniLM-L6-v2)Class BalancingSMOTE

Data Preprocessing
Cleaning Steps

Combined published date, headline, and short description into a single unified text column
Dropped null values and properly converted date fields
Unified category labels (e.g., "ARTS & CULTURE" → "ARTS")
Removed URLs, special characters, mentions/hashtags, and excess whitespace
Converted all text to lowercase

Linguistic Preprocessing (NLTK)

Tokenization
Stopword removal
Lemmatization with WordNet Lemmatizer
Filtered out short tokens (< 3 characters)


Feature Engineering & Class Balancing
Feature Representation

TF-IDF Vectorization — up to 10,000 features
Sentence Transformers (all-MiniLM-L6-v2) for semantic embeddings

Class Balancing

Applied SMOTE to handle highly imbalanced category distributions
Ensured equal representation across all categories

Train / Validation / Test Split

80% Training + Validation / 20% Test (stratified)
Training set further split: 80% Train / 20% Validation


Models
1. MLP Classifier — News Categorization

Multi-Layer Perceptron neural network
Handles high-dimensional, sparse text data effectively
Balanced with SMOTE

MetricScoreValidation Accuracy 87% Test Accuracy 87% Test F1-Score (macro) 87%
2. Logistic Regression — Trending Topic Detection (Best Model)

Simple, fast, and effective on high-dimensional sparse data
Balanced class weights for fairness across categories
Max iterations: 1,000

MetricScoreValidation Accuracy ~ 60% Test Accuracy ~ 55.2% Test F1-Score (macro) ~ 51%
3. XGBoost Classifier — Trending Topic Detection

Gradient boosting decision tree algorithm
Captures non-linear feature interactions
More prone to overfitting on this dataset

MetricScoreTraining Accuracy87% Validation Accuracy~51.7% Test Accuracy~51.9%

Performance Summary
MetricLogistic Regression (Trending)XGBoost (Trending)MLP (Categorization)Train Accuracy 54.8% 87.0% 87% Validation Accuracy~60% 51.7% 87% Test Accuracy57.2% 51.9% 87% F1-Score (macro) 57% 51% 87% VerdictMore stableOverfitted Best for categorization

Conclusions

MLP is the best model for news categorization.
Logistic Regression outperformed XGBoost on all major metrics for trending topic detection.
Combining TF-IDF and Sentence Transformers (all-MiniLM-L6-v2) yielded strong results.
SMOTE oversampling significantly improved learning across underrepresented categories.

Limitations

Dataset contains noisy labels with subtle class differences.
Class overlap in topics can confuse simpler models.

Future Work

Experiment with BERT and other transformer-based models for richer semantic understanding.
Apply attention mechanisms to focus on key parts of text.
Use ensemble techniques to combine the strengths of multiple models.
Explore topic modeling (e.g., LDA) to improve category distinction.
