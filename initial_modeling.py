import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import liwc
import pprint

from nltk.corpus import stopwords
from transformers import pipeline

from liwc import Liwc
from collections import Counter
from wordcloud import WordCloud 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

stopwords = nltk.corpus.stopwords.words('english')
stopwords.append('know')
stopwords.append('like')
stopwords.append('it')
keep = ["i", 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them']
for word in keep:
    if word in stopwords:
        stopwords.remove(word)

#def forest():

#def logisticRegression():

def linearSVC(df, features, labels):
    model = LinearSVC()
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    y_pred = model.predict(X_test)
    score = (y_pred == y_test['id'].values).sum()
    print("Linear Support Vector Machine Score:", int(score / len(y_pred) * 100),"%")

def naive_bayes(df):
    X_train, X_test, y_train, y_test = train_test_split(df['sentence'], df['disease'], random_state = 0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, y_train.values.ravel())

    prediction = clf.predict(count_vect.transform(X_test.array))
    score = (prediction == y_test.array).sum()
    print("Naive Bayes Score:", int(score / len(prediction) * 100),"%")

def ngramCorrelation(tfidf, df, features, labels):
    N = 2
    category_id_df = df[['disease', 'id']].drop_duplicates().sort_values('id')
    category_to_id = dict(category_id_df.values)

    for disease, id in category_to_id.items():
        target = (labels == id)['id'].array
        features_chi2 = chi2(features, target)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names_out())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        print("#'{}':".format(disease))
        print("Most correlated unigrams:\n {}".format('\n '.join(unigrams[-N:])))
        print("Most correlated bigrams:\n {}\n".format('\n '.join(bigrams[-N:])))

def statistics(features, labels):
    models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
    ]
    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, features, labels.values.ravel(), scoring='accuracy', cv=CV)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    sns.boxplot(x='model_name', y='accuracy', data=cv_df)
    sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
                size=8, jitter=True, edgecolor="gray", linewidth=2)
    print(cv_df.groupby('model_name').accuracy.mean())
    plt.show()

def createDataFrame(input):
    data = {"disease":[],"sentence":[], "id":[]}
    i = 0
    sentiment = []
    sentiment_pipeline = pipeline("sentiment-analysis")
    for tag, lines in input.items():
        for line in sentiment_pipeline(lines.split('. ')):
            if line['label'] == 'POSITIVE':
                sentiment.append("+" + str(line['score']))
            else:
                sentiment.append("-" + str(line['score']))
        for line in lines.split('. '):
            data.get("disease").append(tag)
            data.get("sentence").append(line)
            data.get("id").append(i)
        i += 1
    df = pd.DataFrame.from_dict(data)
    df['Sentiment'] = sentiment

    df = df.dropna()

    return df

def createDataFrame2(input):
    data = {"disease":[],"sentence":[], "id":[]}
    i = 0
    for tag, lines in input.items():
        for line in lines.split('. '):
            data.get("disease").append(tag)
            data.get("sentence").append(line)
            data.get("id").append(i)
        i += 1
    df = pd.DataFrame.from_dict(data)

    df = df.dropna()

    return df


def liwc(text):
    liwc = Liwc()
    print(liwc.search('happy'))
    print(liwc.parse('I love ice cream.'.split(' ')))
    
def generateWordCloud(text):
    stopwords.append("br")
    stopwords.append("href")
    for tag, val in text.items():
        wordcloud = WordCloud(stopwords=stopwords).generate(val)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(tag + " WordCloud")
        plt.show()

def sentimentAnalysis(text):
    data = [w for w in text.split() if (w != ".") and (len(w.split('.')) < 2)]
    text = text.split('.')
    #pprint.pprint(nltk.word_tokenize(text), width=79, compact=True)
    fd = nltk.FreqDist(data)
    print(fd.tabulate(4))
    finder = nltk.collocations.TrigramCollocationFinder.from_words(data)
    print(finder.ngram_fd.most_common(2))
    sentiment_pipeline = pipeline("sentiment-analysis")
    sentiment = []
    for i, line in enumerate(sentiment_pipeline(text)):
        if line['label'] == 'POSITIVE':
            sentiment.append("+" + str(line['score']))
        else:
            sentiment.append("-" + str(line['score']))
    return sentiment

def main():
    # discussion forum
    '''
    s = open('NewData/s_processed.txt', 'r')
    s_data = s.read()
    s.close()
    c = open('NewData/c_processed.txt', 'r')
    c_data = c.read()
    c.close()
    bi = open('NewData/bi_processed.txt', 'r')
    bi_data = bi.read()
    bi.close()
    sa = open('NewData/sa_processed.txt', 'r')
    sa_data = sa.read()
    sa.close()
    '''
    # podcasts
    s = open('processed_data/schizophrenia.txt', 'r')
    s_data = s.read()
    s.close()
    c = open('processed_data/control.txt', 'r')
    c_data = c.read()
    c.close()
    sa = open('processed_data/schizoaffective.txt', 'r')
    sa_data = sa.read()
    sa.close()
    #input = {"Schizophrenia":s_data , "Bipolar":bi_data , "Schizoaffective":sa_data , "Control":c_data }
    input = {"Schizophrenia":s_data , "Schizoaffective":sa_data , "Control":c_data }
    df = createDataFrame2(input)
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(df.sentence).toarray()
    labels = df['id'].to_frame()

    #ngramCorrelation(tfidf, df, features, labels)
    #statistics(features, labels)
    #linearSVC(df, features, labels)
    #naive_bayes(df)
    #generateWordCloud(input)
    df.to_csv('dataPod.csv')
    print(df)

main()


    

