import pandas as pd
import os
import importlib

import gensim
from gensim.utils import simple_preprocess
import nltk

from nltk.corpus import stopwords

import gensim.corpora as corpora

from pprint import pprint

import pyLDAvis
import pyLDAvis.gensim_models
import pickle 

df = pd.read_csv('data.csv').drop(['Unnamed: 0'], axis=1)
df = df.dropna()

bigrams_measures = nltk.collocations.BigramAssocMeasures()
finder = nltk.collocations.BigramCollocationFinder.from_documents([line.split() for line in df.sentence])
finder.apply_freq_filter(10)
bigram_scores = finder.score_ngrams(bigrams_measures.pmi)
print(bigram_scores)





    

"""

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]

data = df.sentence.values.tolist()
data_words = list(sent_to_words(data))
# remove stop words
#data_words = remove_stopwords(data_words)

# Create Dictionary
id2word = corpora.Dictionary(data_words)
# Create Corpus
texts = data_words
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View

# number of topics
num_topics = 5
# Build LDA model
if __name__ == '__main__':
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=num_topics)
    # Print the Keyword in the 4 topics
    topics = lda_model.print_topics()
    doc_lda = lda_model[corpus]
    pprint(lda_model.print_topics())
    

    for disease in topics:
        for scores in disease[1].split("+"):
            score, word = scores.split("*")
            word = word[1:len(word)-2]
            if float(score) < .010:
                filter = df[df['id'] == disease[0]]
                #print(filter)

"""