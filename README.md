# tweet_classification

This project offers a straightforward machine learning approach to sort tweets into three categories: hate speech, offensive language, and neither. It uses Support Vector Machines (SVM) and GloVe (Global Vectors for Word Representation) embeddings, particularly the glove-twitter-200 dataset, to perform a basic analysis of what tweets are about. By running tweets through a trained SVM model, it becomes possible to automatically classify the tone of tweets, which helps in managing and understanding conversations on social media. This project not only demonstrates how natural language processing (NLP) techniques can be applied but also acts as a practical tool for those new to machine learning, providing them with a chance to try out simple models.

## Project Structure

- `app.py`: The main Flask application script for running the web interface.
- `model.pkl`: The pre-trained SVM model (not included in the repository, see setup below).
- `templates/`: Folder containing HTML templates for the Flask application.
- `requirements.txt`: File listing the necessary Python packages.
- 'tweet_classifier.py': Python script loading and preprocessing data, training and saving an SVM model 

## Setup

1. **Clone the Repository**

git clone https://github.com/birsenyy/tweet-classifier.git
cd tweet-classifier

2. **Install Dependencies**

pip install -r requirements.txt

3. **Download Glove Vectors**

The Glove vectors can be loaded directly in Python script:

glove_vectors = gensim.downloader.load('glove-twitter-200'

However, if using Jupyter Notebook, the IOPub message rate limit may not be sufficient when loading embeddings with higher dimensions. 

In this case, apply the following steps:

Dowload glove.twitter.27B.zip from https://nlp.stanford.edu/projects/glove/

Unzip the folder. 

The Python script 'tweet-classifier.py" was written applying the second method. Please make necessary changes if you load the embeddings directly.


## Running the Application

1. Start the Flask application:

python app.py

2. Open a web browser and navigate to `http://127.0.0.1:5001` to access the application.

## Usage and extra information:

Enter a tweet into the form on the web page and submit it to classify the tweet into categories based on its content.

model_25.pkl and model_200 are pre-trained models that were trained by Glove embeddings with size 25 and 200. 

