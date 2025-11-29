import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

import re
import pickle
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('all')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request, redirect, session


app = Flask(__name__)




with open("person.pkl", "rb") as f:
    tokenizer = pickle.load(f)


datas = pickle.load(open('person.pkl','rb'))



porter_stem = PorterStemmer()

def tokenization(text):
  res= text.lower()
  res= re.sub(r'/d+', '', text)
  res= re.sub(r'/W', '', text)
  res= re.sub(r'/w/s', '', text)
  res= re.sub(r'^a-zA-Z0-9', '', text)
  res= re.sub(r'\s+', ' ', text).strip()
  res= [porter_stem.stem(word) for word in res if not word in stopwords.words('english')]
  res = ''.join(res)

  return res



def cosine_similarity_matrix(text1, text2):
    # Initialize the TF-IDF Vectorizer for english words
    vectorizer = TfidfVectorizer(tokenizer=tokenization)

    # Fit and transform the new text feature to create the TF-IDF matrix
    text_vectors = vectorizer.fit_transform([text1, text2])

    # Calculate the cosine similarity matrix between all 'texts'  
    cosine_sim_matrix = cosine_similarity(text_vectors)

    similarity_between_texts =  cosine_sim_matrix

    return similarity_between_texts[0][1]




def get_recommendations(search_query):
  search_query = search_query.lower()
  processed_query = tokenization(search_query)

  def calculate_similarity(product_text):
    return cosine_similarity_matrix(processed_query, product_text)

  datas['similarity_scores'] = datas['title'].apply(calculate_similarity)

  best_matches = datas.sort_values(by=['similarity_scores'], ascending=False).head(10)

  data = []
  for i in best_matches.index:
    data.append(best_matches['title'][i])
    data.append(best_matches['category'][i])
    data.append(best_matches['imgs'][i])

  return data


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html',
                           title= list(datas['title']),
                           category=list(datas['category']),
                           imgs=list(datas['imgs']))


# @app.route('/recommend', methods=['GET', 'POST'])
# def recommend():
#     result = None
#     error=None
   
#     if request.method == 'POST':
#        query = request.form.get('query')

#        if not query:
#           return render_template("res.html", error="no query")

#        if len(query) < 2:
#           return render_template("res.html", error="Text too short")
       
#        if query:
#           query = query.lower()
#           processed_query = tokenization(query)

#           def calculate_similarity(product_text):
#             return cosine_similarity_matrix(processed_query, product_text)

#           datas['similarity_scores'] = datas['title'].apply(calculate_similarity)

#           best_matches = datas.sort_values(by=['similarity_scores'], ascending=False).head(10)

#           results = []
#           for i in best_matches.index:
#             results.append(best_matches['title'][i])
#             results.append(best_matches['category'][i])
#             results.append(best_matches['imgs'][i])

    
#     return render_template("rec.html", results=results, 
#                            title= list(datas['title']),
#                            category=list(datas['category']),
#                            imgs=list(datas['imgs']))
          



@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    result = None
    error=None
   
    if request.method == 'POST':
       query = request.form.get('query')

       if not query:
          return render_template("res.html", error="no query")

       if len(query) < 2:
          return render_template("res.html", error="Text too short")
       
       if query:
          query = query.lower()
          processed_query = tokenization(query)

          def calculate_similarity(product_text):
            return cosine_similarity_matrix(processed_query, product_text)

          datas['similarity_scores'] = datas['title'].apply(calculate_similarity)

          best_matches = datas.sort_values(by=['similarity_scores'], ascending=False).head(10)

          results = []
          for i in best_matches.index:
            product_tuple = (
               best_matches['title'][i],
               best_matches['category'][i],
               best_matches['imgs'][i]
               
            )

            results.append(product_tuple)

    
    return render_template("rec.html",error=error, results=results, 
                           title= list(datas['title']),
                           category=list(datas['category']),
                           imgs=list(datas['imgs']))
          





if __name__ == "__main__":
    app.run(debug=True)