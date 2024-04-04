# Tweet Sentiment Analysis: Classifying Hate Speech and Offensive Language

This project offers a straightforward machine learning approach to sort tweets into three categories: hate speech, offensive language, and neither. It uses Support Vector Machines (SVM) and GloVe (Global Vectors for Word Representation) embeddings, particularly the glove-twitter-200 dataset, to perform a basic analysis of what tweets are about. The initiative is based on a dataset sourced from https://data.world/thomasrdavidson/hate-speech-and-offensive-language, which comprises labeled tweets that have been manually categorized to facilitate the training and evaluation of the model. By running tweets through a trained SVM model, it becomes possible to automatically classify the tone of tweets, which helps in managing and understanding conversations on social media. This project not only demonstrates how natural language processing (NLP) techniques can be applied but also acts as a practical tool for those new to machine learning, providing them with a chance to try out simple models.

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

3. **Dowload dataset**

Dowload the dataset from:

https://data.world/thomasrdavidson/hate-speech-and-offensive-language/workspace/file?filename=labeled_data.csv

3. **Download Glove Vectors**

The Glove vectors can be loaded directly in Python script:

glove_vectors = gensim.downloader.load('glove-twitter-200'

However, if using Jupyter Notebook, the IOPub message rate limit may not be sufficient when loading embeddings with higher dimensions. 

In this case, apply the following steps:

Dowload glove.twitter.27B.zip from https://nlp.stanford.edu/projects/glove/

Unzip the folder. 

The Python script 'tweet-classifier.py" was written applying the second method. Please make necessary changes if you load the embeddings directly.

model_25.pkl and model_200 in the repository are pre-trained models that were trained by Glove embeddings with size 25 and 200. 

## Running the Application

1. Start the Flask application:

python app.py

2. Open a web browser and navigate to `http://127.0.0.1:5001` to access the application.

## Usage:

Enter a tweet into the form on the web page and submit it to classify the tweet into categories based on its content.

## Dataset Acknowledgment

This project utilizes the "Hate Speech and Offensive Language" dataset originally compiled by Thomas Davidson, Dana Warmsley, Michael Macy, and Ingmar Weber. The dataset is made available under the Creative Commons Attribution (CC-BY) license, allowing for both academic and commercial use with proper attribution.

The dataset can be accessed and downloaded from [data.world](https://data.world/thomasrdavidson/hate-speech-and-offensive-language).

### License

This dataset is distributed under the terms of the Creative Commons Attribution (CC-BY) license. Users are free to share and adapt the dataset as long as appropriate credit is given, a link to the license is provided, and it is indicated if changes were made. More details about the CC-BY license can be found [here](https://creativecommons.org/licenses/by/4.0/).

### How to Cite

If you use this dataset in your project, please provide the following attribution:

> Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). Hate Speech and Offensive Language. Retrieved from https://data.world/thomasrdavidson/hate-speech-and-offensive-language

We express our gratitude to the authors for their work and for making this valuable resource available to the community.
