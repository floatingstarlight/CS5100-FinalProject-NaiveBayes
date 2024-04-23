import numpy as np 
import pandas as pd
from text_preprocessor import TextPreprocessor
from NaiveBayesClassifier import NaiveBayesClassifier
from random import shuffle
from math import log2
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Read data input and return list of texts + numeric labels
def read_input():
    # Preprocessing Twitter data
    twitter_data = pd.read_csv('./data/Twitter_Data.csv')
    twitter_data = twitter_data.dropna()

    twitter_texts = list(twitter_data['clean_text'])
    twitter_labels = []

	# Encoding sentiment label
    for label in twitter_data["category"]:
        if label == -1:
            twitter_labels.append(-1)
        elif label == 0:
            twitter_labels.append(0)
        elif label == 1:
            twitter_labels.append(1)
        else:
            print(label)

    # Preprocessing Reddit data
    reddit_data = pd.read_csv('./data/Reddit_Data.csv')
    reddit_data = reddit_data.dropna()

    reddit_texts = list(reddit_data['clean_comment'])
    reddit_labels = []

	# Encoding sentiment label
    for label in reddit_data["category"]:
        if label == -1:
            reddit_labels.append(-1)
        elif label == 0:
            reddit_labels.append(0)
        elif label == 1:
            reddit_labels.append(1)
        else:
            print(label)

    return (twitter_texts, twitter_labels), (reddit_texts, reddit_labels)


def evaluate_model(train_list, test_list):
    vocab_size = 5000
    classes=[-1, 0, 1]
    model = NaiveBayesClassifier(classes, vocab_size=vocab_size)
    model.train(train_list)

    actual_labels = []
    predicted_labels = []

    for sample in test_list:
        y = sample[1]
        y_pred = model.test(sample[0])

        # Update evaluation metrics
        actual_labels.append(y)
        predicted_labels.append(y_pred)


    # Calculate 4 evaluation metrics
    accuracy = accuracy_score(actual_labels, predicted_labels)
    precision = precision_score(actual_labels, predicted_labels, average='macro')
    recall = recall_score(actual_labels, predicted_labels, average='macro')
    f1 = f1_score(actual_labels, predicted_labels, average='macro')

    print('Precision: %f' % precision)
    print('Recall: %f' % recall)
    print('F1-score: %f' % f1)
    print('Accuracy: %f' % accuracy)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(actual_labels, predicted_labels)

	# Calculate percentages
    conf_matrix_percent = conf_matrix.astype('float') / np.sum(conf_matrix)

    # Plot confusion matrix with percentages
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_percent, annot=True, fmt='.2%', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'], annot_kws={"size": 18})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (%)')
    plt.savefig('confusion_matrix_naiveNB_percent.png')

def preprocess_and_split_data(twitter_data, reddit_data, config):
	# Use TextPreprocessor class for preprocessing
    preprocessor = TextPreprocessor()
	
	# Initialize empty lists
    processed_reddit_data = []
    processed_twitter_data = []

	#Unigram + No preprocessing
    if 'noProcessing' in config:
        processed_reddit_data = [[preprocessor.clean_text(data), label] for data, label in zip(reddit_data[0], reddit_data[1])]
        processed_twitter_data = [[preprocessor.clean_text(data), label] for data, label in zip(twitter_data[0], twitter_data[1])]
	#Unigram + Add abbreivation
    elif 'abbrev' in config:
        processed_reddit_data = [[preprocessor.clean_text(data, replace_abbrev=True), label] for data, label in zip(reddit_data[0], reddit_data[1])]
        processed_twitter_data = [[preprocessor.clean_text(data, replace_abbrev=True), label] for data, label in zip(reddit_data[0], reddit_data[1])]
    #Unigram + Remove Stopwords
    elif 'stopWords' in config:
        processed_reddit_data = [[preprocessor.clean_text(data, remove_stopwords=True), label] for data, label in zip(reddit_data[0], reddit_data[1])]
        processed_twitter_data = [[preprocessor.clean_text(data, remove_stopwords=True), label] for data, label in zip(reddit_data[0], reddit_data[1])]
    #Unigram + Lemmitization
    elif 'lemma' in config:
        processed_reddit_data = [[preprocessor.clean_text(data, lemmatization=True), label] for data, label in zip(reddit_data[0], reddit_data[1])]
        processed_twitter_data = [[preprocessor.clean_text(data, lemmatization=True), label] for data, label in zip(reddit_data[0], reddit_data[1])]
    #Unigram + Stemming
    elif 'stem' in config:
        processed_reddit_data = [[preprocessor.clean_text(data, stemming=True), label] for data, label in zip(reddit_data[0], reddit_data[1])]
        processed_twitter_data = [[preprocessor.clean_text(data,stemming=True), label] for data, label in zip(reddit_data[0], reddit_data[1])]
    #Unigram + bigram
    elif 'bigram' in config:
        processed_reddit_data = [[preprocessor.clean_text_with_bigrams(data), label] for data, label in zip(reddit_data[0], reddit_data[1])]
        processed_twitter_data = [[preprocessor.clean_text_with_bigrams(data), label] for data, label in zip(reddit_data[0], reddit_data[1])]
    #Unigram + All Processing implement
    if 'all' in config:
        processed_reddit_data = [[preprocessor.clean_text(data, remove_all_punct=True,remove_url=True, remove_html=True,replace_abbrev=True,
               lemmatization=True, stemming=True, remove_stopwords=True, lower_case=True), label] for data, label in zip(reddit_data[0], reddit_data[1])]
        processed_twitter_data = [[preprocessor.clean_text(data, remove_all_punct=True,remove_url=True, remove_html=True,replace_abbrev=True,
               lemmatization=True, stemming=True, remove_stopwords=True, lower_case=True), label] for data, label in zip(twitter_data[0], twitter_data[1])]
	    

    # Combine Reddit and Twitter data
    processed_data = processed_reddit_data + processed_twitter_data
    
    # Shuffle data and split into train/test with 90:10 ratio
    shuffle(processed_data)
    train_size = int(len(processed_data) * 0.9)
    train_list = processed_data[:train_size]
    test_list = processed_data[train_size:]

    return train_list, test_list


def main():
	#Read two original list
    twitter_data, reddit_data = read_input()
    processed_data = []

	# Instantiate the TextPreprocessor object
    preprocessor = TextPreprocessor()

	# Variations/Experiments:
	# Define different preprocessing configurations
	# all: Include all preprocessing 
	# noProcessing: Only Unigram + no preprocessing at all
	# Abbrev: Unigram + Abbrevation replace
	# Lemma: test for lemmatization
	# Stemming: test for removing stemming word
	# stopWords: test for remove stopWords
	# bigram: test for bigram feature selection
    preprocessing_configs = [
        {'all': True},
        {'abbrev': True},
        {'lemma': True},
        {'stem': True},
        {'stopWords': True},
        {'bigram': True},
		{'noProcessing': True},
    ]

	# Train and evaluate models for each preprocessing configuration
    for config in preprocessing_configs:
        print(f"Evaluating model with preprocessing configuration: {config}")
        train_list, test_list = preprocess_and_split_data(twitter_data, reddit_data, config)
        print("---------------")
		# Train and evaluate the model
        evaluate_model(train_list, test_list)
        print("---------------")

if __name__ == "__main__":
    main()


