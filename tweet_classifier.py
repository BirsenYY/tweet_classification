#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import pandas as pd
import numpy as np
import gensim.downloader
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# Function to load GloVe embeddings from a file
def load_glove_embeddings(path):
    """
    Load GloVe embeddings from a file into a dictionary.
    
    Parameters:
    path (str): Path to the GloVe embeddings file.
    
    Returns:
    dict: A dictionary mapping words to their embedding vectors.
    """
    embeddings_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict

# Function to convert tweets to embeddings
def from_word_to_embeddings(tweet, glove_vectors):
    """
    Convert a tweet into an embedding by averaging the embeddings of its words.
    
    Parameters:
    tweet (str): The tweet text.
    glove_vectors (dict): A dictionary of word embeddings.
    
    Returns:
    np.array: The average embedding of the tweet.
    """
    word_list = tweet.split()
    word_set = set(word_list)
    unique_word_list = list(word_set)
    embedding_list = []
    
    for word in unique_word_list:
        if word.isalpha() and word in glove_vectors:
            embedding_list.append(glove_vectors[word])
    
    if embedding_list:
        average_embedding = np.mean(embedding_list, axis=0)
    else:
        average_embedding = np.zeros(200)  # Assuming 200-dimensional embeddings
    
    return average_embedding

# Main script starts here
if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv('labeled_data.csv')

    # Load GloVe vectors
    glove_path = 'glove.twitter.27B.200d.txt'  # Update with the correct path
    glove_vectors = load_glove_embeddings(glove_path)
    #Alternative to the above, the Glove embeddings can be loaded directly as follow (if the IOPub message rate limit is sufficient):
    # glove_vectors = gensim.downloader.load('glove-twitter-100')
    #If using the line above, you need to update the code accordingly.
    # Convert tweets to embeddings
    embedding_list = df['tweet'].apply(lambda x: from_word_to_embeddings(x, glove_vectors))
    X = np.vstack(embedding_list)
    y = df['class'].to_numpy()

    # Split the dataset and train the SVM classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = SVC(kernel='linear', class_weight='balanced')
    clf.fit(X_train, y_train)

    # Evaluate the model
    predictions = clf.predict(X_test)
    print(classification_report(y_test, predictions))

    # Save the trained model
    with open('model_200.pkl', 'wb') as f:
        pickle.dump(clf, f)
