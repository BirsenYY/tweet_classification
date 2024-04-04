
from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import gensim.downloader

app = Flask(__name__)

# Load the pre-trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load GloVe vectors
glove_vectors = gensim.downloader.load('glove-twitter-25')
embedding_dim = 25  # Dimensionality of the GloVe vectors you're using

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])

def classify_tweet():
    data = request.form['tweet']
    words = data.lower().split()
    embeddings = [glove_vectors[word] for word in words if word in glove_vectors]
    
    if embeddings:
        tweet_embedding = np.mean(embeddings, axis=0)
    else:
        tweet_embedding = np.zeros(embedding_dim)
    
    tweet_embedding = tweet_embedding.reshape(1, -1)
    
    # Predict class using the loaded model
    prediction = model.predict(tweet_embedding)
    prediction_text = ''
    predicted_class = prediction[0]
    if predicted_class == 0:
       prediction_text = " The tweet contains hate speech."
    elif predicted_class == 1:
       prediction_text = "The tweet contains offensive language."
    elif predicted_class == 2:
       prediction_text = "The tweet is accepted as neither hate speech nor offensive language."
    
    return render_template('index.html', prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True, port=5001)






