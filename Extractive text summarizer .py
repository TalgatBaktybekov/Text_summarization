#!/usr/bin/env python
# coding: utf-8

# In[8]:


import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import pandas as pd
from nltk.tokenize import word_tokenize
import string
def read_article(file_name):
    file = open(file_name, "r")
    filedata = file.readlines()
    filedata = ''.join(filedata)
    report = filedata.split(". ")
    sentences = []
    lm = WordNetLemmatizer()
    for sentence in report:
        sentence = sentence.replace("\n", " ")
        sentence = sentence.replace("[^a-zA-Z]", " ")
        sentence = [lm.lemmatize(w.lower().strip(string.punctuation)) for w in sentence.split(" ") if w.lower().strip(string.punctuation) != '']
        if len(sentence) > 0:
            sentences.append(sentence)
    sentences.pop()

    
    return sentences
def tfidf(report, word):
    count = 0
    for sentence in report:  
        if word in sentence:
            count += 1
    return np.log(len(report)/count)        
def sentence_similarity(report, sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    vector3 = list(map(lambda x: x[0]/len(sent1) * tfidf(report, x[1]), zip(vector1, all_words)))
    vector4 = list(map(lambda x: x[0]/len(sent1) * tfidf(report, x[1]), zip(vector2, all_words)))
    return 1 - cosine_distance(vector3, vector4)

def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences))) 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences, sentences[idx1], sentences[idx2], stop_words)
    return similarity_matrix


def generate_summary(file_name, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    # Read text anc split it
    sentences =  read_article(file_name)

    # Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)       
    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))

    # output the summarize text
    print("Summarize Text: \n", ". ".join(summarize_text))

# let's begin
generate_summary( "The_Necklace.txt", 5)


# In[ ]:




