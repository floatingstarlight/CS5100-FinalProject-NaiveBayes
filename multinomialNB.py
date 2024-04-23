import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from text_preprocessor import TextPreprocessor


def load_and_process_data():
    # Read data input
    reddit_data = pd.read_csv('./data/Reddit_Data.csv').dropna()
    twitter_data = pd.read_csv('./data/Twitter_Data.csv').dropna()

    reddit_data.columns = ['messages', 'labels']
    twitter_data.columns = ['messages', 'labels']

    # Concatenate data from Reddit and Twitter
    data = pd.concat([reddit_data, twitter_data], ignore_index=True)

    # Preprocess messages
    preprocessor = TextPreprocessor()
    data['messages1'] = data['messages'].apply(preprocessor.clean_text)

    # Finding X and y
    X = data['messages1']
    y = data['labels']
    return X, y

def plot_confusion_matrix(y_true, y_pred):
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Calculate percentages
    conf_matrix_percent = conf_matrix.astype('float') /  np.sum(conf_matrix)

    # Plot confusion matrix with percentages
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_percent, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=['Negative', 'Neutral', 'Positive'], 
                yticklabels=['Negative', 'Neutral', 'Positive'], annot_kws={"size": 18})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Percentage)')
    plt.savefig('confusion_matrix_multimonimalNB.png')

def main():
    # Preprocess tweets
    comments, classes = load_and_process_data()

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(comments, classes, test_size=0.1, random_state=42)

    # Convert lists of comments to strings
    X_train_str = [' '.join(comment) for comment in X_train]
    X_test_str = [' '.join(comment) for comment in X_test]

    # preprocess and transform the raw text data into a numerical format 
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train_str)
    X_test_vectorized = vectorizer.transform(X_test_str)

    # Train Naive Bayes classifier
    nb_classifier = MultinomialNB(alpha=0.1)
    nb_classifier.fit(X_train_vectorized, y_train)

    # Evaluate classifier
    y_pred = nb_classifier.predict(X_test_vectorized)
    print(classification_report(y_test, y_pred))

	#Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)


if __name__ == "__main__":
    main()


