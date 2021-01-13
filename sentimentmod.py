'''
Bu modül sentiment analizi yapmamızı sağlar.

'''

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#File: sentiment_mod.py

import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize


#Kullanacağımız VoteClassifier classımız nltk.classify'daki ClassifierI'dan miras alınarak oluşturulur.
#bu classı kullandığımız çeşitli classifierlar için mode almakta ve her bir classifier için ayrı bir confidence değeri döndürebilme amaçlı tanımlarız
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

#short reviewsdan aldığımız ve önceden pickling yaptığımız dosyalarımızı read in bytes modunda açarız ve bu pickle dosyalarını kullanarak her bir classifierımız için işlemlerimizi yaparız
documents_f = open("short_data_pickled/save_doc.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


word_features5k_f = open("short_data_pickled/word_feature_save.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()


featuresets_f = open("short_data_pickled/featuresets.pickle", "rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

random.shuffle(featuresets)#featuresetsleri kararız
print(len(featuresets))
#ilk 10000 değeri testing için on 10000 değeri training için kullanmak üzere atama yaparız
testing_set = featuresets[10000:]
training_set = featuresets[:10000]



open_file = open("short_data_pickled/short_reviews.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()


open_file = open("short_data_pickled/short_reviews_MNB.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()



open_file = open("short_data_pickled/short_reviews_Bernoulli.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()


open_file = open("short_data_pickled/short_reviews_LogisticReg.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()


open_file = open("short_data_pickled/short_reviews_LinearSVC.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()






#VoteClassifier classını kullanarak pickling ile okuduğumuz dosyalarımızdan gelen returnleri bir değişkene atama yaparız
voted_classifier = VoteClassifier(classifier,LinearSVC_classifier,MNB_classifier,BernoulliNB_classifier,LogisticRegression_classifier)



#sentimentmod modülümüz için live_graphing ve connection_twitter'da kullanacağımız fonksiyonumuzu tanımlarız
def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)

