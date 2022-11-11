from dataclasses import replace
from multiprocessing.resource_sharer import stop
import pandas as pd
import string
import re
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import collections
import contractions
from sklearn.feature_extraction.text import CountVectorizer
#from keras.preprocessing.text import Tokenizer
import language_tool_python  
from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.corpus import stopwords
from string import punctuation as punc

from heapq import nlargest

stopwords = nltk.corpus.stopwords.words('english')
stopwords = [contractions.fix(word) for word in stopwords]
stopwords.append('know')
stopwords.append('like')
stopwords.append('it')
keep = ["i", 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them']
for word in keep:
    if word in stopwords:
        stopwords.remove(word)


def expand_contractions_lower(text):
    return contractions.fix(text.lower())

def remove_punctuation(text):
    for i in range(len(text)):
        text[i] = re.sub(r'[^\w\s]', '', text[i])
    return text

def tokenize(text):
    return sent_tokenize(text)

def remove_stop_words(text):
    return " ".join([word for word in text.split() if word not in stopwords])

def lemmetization(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    return " ".join([wordnet_lemmatizer.lemmatize(word) for word in text.split()])

def fix_grammar_spelling(text):
    # using the tool
    text = text.replace(",", "")
    my_tool = language_tool_python.LanguageTool('en-US')  

    text = my_tool.correct(text)
    newtext = ""
    for word in text.split():
        if word != "i":
            newtext += word
        if word == text[len(text)-1]:
            newtext += "."
        else:
            newtext += " "
    return newtext

def summarize(text):
    tokens = word_tokenize(text)
    punctuation = punc + '\n'
    word_frequencies = {}
    for word in tokens:    
        if word.lower() not in stopwords:
            if word.lower() not in punctuation:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1
    max_freq = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_freq
    sent_token = sent_tokenize(text)
    sentence_scores = {}
    for sent in sent_token:
        sentence = sent.split(" ")
        for word in sentence:        
            if word.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.lower()]
    select_length = int(len(sent_token)*0.3)
    summary = nlargest(select_length, sentence_scores, key = sentence_scores.get)
    final_summary = [word for word in summary]
    summary = ' '.join(final_summary)
    return summary

def remove_nums(text):
    return re.sub(r'[0-9]+', '', text)

def clean(text):
    text = remove_nums(text)
    text = expand_contractions_lower(text)
    text = remove_stop_words(text)
    text = lemmetization(text) 
    text = tokenize(text)
    text = remove_punctuation(text)
    return text

def processSocial():
    file = open('NewData/s_raw.txt')
    file_data = file.read().replace("\n", ' ')
    print("Schizophrenia Data Set Size:", len(file_data))
    file_data = clean(file_data)
    write_to = open('NewData/s_processed.txt', 'w')
    for line in file_data:
        write_to.write(line)
        write_to.write(". ")
    file.close()
    write_to.close()

    file = open('NewData/sa_raw.txt')
    file_data = file.read().replace("\n", ' ')
    print("Schizoaffective Data Set Size:", len(file_data))
    file_data = clean(file_data)
    write_to = open('NewData/sa_processed.txt', 'w')
    for line in file_data:
        write_to.write(line)
        write_to.write(". ")
    file.close()
    write_to.close()

    file = open('NewData/c_raw.txt')
    file_data = file.read().replace(".", ' ')
    print("Control Data Set Size Pre Summarize:", len(file_data))
    file_data = summarize(file_data)
    print("Control Data Set Size:", len(file_data))
    file_data = clean(file_data)
    write_to = open('NewData/c_processed.txt', 'w')
    for line in file_data:
        write_to.write(line)
        write_to.write(". ")
    file.close()
    write_to.close()

    file = open('NewData/bi_raw.txt')
    file_data = file.read().replace("\n", ' ')
    print("Bipolar Data Set Size:", len(file_data))
    file_data = clean(file_data)
    write_to = open('NewData/bi_processed.txt', 'w')
    for line in file_data:
        write_to.write(line)
        write_to.write(". ")
    file.close()
    write_to.close()

def processPodcast():
    s = open("cleaned_data/schizophrenia.txt", "r")
    c = open("cleaned_data/control.txt", "r")
    sa = open("cleaned_data/schizoaffective.txt", "r")

    data1 = s.read()
    data2 = c.read()
    data3 = sa.read()

    #data1 = data1.replace(". ", " ")
    #data2 = data2.replace(". ", " ")
    #data3 = data3.replace(". ", " ")

    data1 = clean(data1)
    data2 = clean(data2)
    data3 = clean(data3)

    file = open("processed_data/schizophrenia.txt", "w")
    for line in data1:
        file.write(line)
        file.write(". ")
    file.close()

    file = open("processed_data/control.txt", "w")
    for line in data2:
        file.write(line)
        file.write(". ")

    file = open("processed_data/schizoaffective.txt", "w")
    for line in data3:
        file.write(line)
        file.write(". ")

    s.close()
    c.close()
    sa.close()

def main():
    #processSocial()
    processPodcast()

main()