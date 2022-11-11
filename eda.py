from operator import index
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
import pytextrank

import pandas as pd

#from wordcloud import WordCloud
from textwrap import wrap

from heapq import nlargest
    

def generate_wordcloud(data,title):
    wc = WordCloud(width=400, height=330, max_words=150,colormap="Dark2").generate_from_frequencies(data)
    plt.figure(figsize=(10,8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(title + ' wordcloud',fontsize=13)
    plt.show()
    # Transposing document term matrix


def wordclouds(text, tag):
    df_grouped = text.split("\n")
    cv=CountVectorizer(analyzer='word')
    data=cv.fit_transform(df_grouped)
    df_dtm = pd.DataFrame(data.toarray(), columns=cv.get_feature_names())

    df_dtm=df_dtm.transpose()
    print(df_dtm)

    generate_wordcloud(df_dtm.sort_values(ascending=False),tag)

    # Plotting word cloud for each product
    for index,product in enumerate(df_dtm.columns):
        generate_wordcloud(df_dtm[product].sort_values(ascending=False),tag)

def get_top_ngram(c1, c2, c3, tag, n=None):
    words_freq = freq_dataframe(c1, c2, c3, n)
    top_ngrams = words_freq.sort_values(by=[tag], ascending=False)[tag][:10]
    x,y=top_ngrams.index, top_ngrams.values
    sns.barplot(x=y,y=x)
    plt.title(tag)
    plt.show()


def freq_dataframe(c1, c2, c3, n=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(c1)
    bag_of_words = vec.transform(c1)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq1 = [(word, sum_words[0, idx]) 
                  for word, idx in vec.vocabulary_.items()]
    words_freq1 =sorted(words_freq1, key = lambda x: x[1], reverse=True)
    top_ngrams = words_freq1[:10]
    x1,y1=map(list,zip(*top_ngrams))

    vec = CountVectorizer(ngram_range=(n, n)).fit(c2)
    bag_of_words = vec.transform(c2)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq2 = [(word, sum_words[0, idx]) 
                  for word, idx in vec.vocabulary_.items()]
    words_freq2 =sorted(words_freq2, key = lambda x: x[1], reverse=True)
    top_ngrams = words_freq2[:10]
    x2,y2=map(list,zip(*top_ngrams))

    vec = CountVectorizer(ngram_range=(n, n)).fit(c3)
    bag_of_words = vec.transform(c3)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq3 = [(word, sum_words[0, idx]) 
                  for word, idx in vec.vocabulary_.items()]
    words_freq3 =sorted(words_freq3, key = lambda x: x[1], reverse=True)
    top_ngrams = words_freq3[:10]
    x3,y3=map(list,zip(*top_ngrams))
    
    bigrams = set()
    for x in x1:
        bigrams.add(x)
    for x in x2:
        bigrams.add(x)
    for x in x3:
        bigrams.add(x)
    bigrams = list(bigrams)
    bigram_df = pd.DataFrame(index=bigrams, columns=['s','sa','c'])
    for word, freq in words_freq1[:50]:
        bigram_df.loc[word, "s"] = freq
    for word, freq in words_freq2[:50]:
        bigram_df.loc[word, "sa"] = freq
    for word, freq in words_freq3[:50]:
        bigram_df.loc[word, "c"] = freq
    bigram_df = bigram_df.sort_values(by=['s','sa','c'], ascending=False)
    return bigram_df.head(20)


def main():
    s = open("processed_data/schizophrenia.txt", "r").read()
    sa = open("processed_data/schizoaffective.txt", "r").read()
    c = open("processed_data/control.txt", "r").read()

    data_s = s.split("\n")
    data_sa = sa.split("\n")
    data_c = c.split("\n")

    get_top_ngram(data_s, data_sa, data_c, "s", 2)
    get_top_ngram(data_s, data_sa, data_c, "sa", 2)
    get_top_ngram(data_s, data_sa, data_c, "c", 2)

    #wordclouds(s, "schizophrenia")
    #wordclouds(sa, "schizoaffective")
    #wordclouds(c, "control")
    

main()

#https://github.com/DivakarPM/NLP/blob/master/Text_Summarization/Text_Summarization.ipynb
