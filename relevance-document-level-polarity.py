#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import langdetect
import xml.etree.ElementTree as ET
import numpy as np

from nltk.corpus import stopwords

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# adapted from:
# https://marcobonzanini.com/2015/03/09/mining-twitter-data-with-python-part-2/

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs
    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?u)\b\w\w+\b'  # other words
    # r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)


def read_xml(input_file, x, y):
    tree = ET.parse(input_file)
    root = tree.getroot()
    for doc in root.findall('Document'):
        text = doc.find('text').text
        relevance = doc.find('relevance').text
        sentiment = doc.find('sentiment').text
        x.append((doc.attrib, text))
        y.append((relevance, sentiment))
        """
        print doc.attrib
        print text
        print "relevance :", relevance
        print "sentiment :", sentiment
        opinions = doc.find('Opinions')
        if opinions:
            for opinion in opinions.findall('Opinion'):
                print opinion.attrib
        """


def predict_relevance(x_texts, y_classes):

    # train classifiers for predicting relevance
    data_x = [x[1] for x in x_texts]
    data_y = [y[0] for y in y_classes]

    data_x = np.array(data_x)
    data_y = np.array(data_y)

    stratified_split = StratifiedShuffleSplit(n_splits=2, test_size=0.33, random_state=42)
    for train_index, test_index in stratified_split.split(data_x, data_y):
        x_train, x_test = data_x[train_index], data_x[test_index]
        y_train, y_test = data_y[train_index], data_y[test_index]

    stop_words = stopwords.words('german')

    # Naive Bayes
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=stop_words, tokenizer=pre_process)),
        ('clf', MultinomialNB(fit_prior=True, class_prior=None)),
    ])
    parameters = {
        'tfidf__max_df': (0.25, 0.5, 0.75),
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'clf__alpha': (1e-2, 1e-3)
    }
    """

    # Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=stop_words, tokenizer=pre_process)),
        ('clf', LogisticRegression(penalty='l2', dual=False, tol=0.0001,
                                   fit_intercept=True, intercept_scaling=1,
                                   class_weight=None, random_state=None,
                                   solver='sag', max_iter=100,
                                   multi_class='ovr', verbose=0,
                                   warm_start=False, n_jobs=1))
    ])
    parameters = {
        'tfidf__max_df': (0.25, 0.5, 0.75),
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'clf__C': (10, 1, 1e-1, 1e-2, 1e-3)
    }

    grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=1, verbose=4)
    grid_search_tune.fit(x_train, y_train)
    print
    print("Best parameters set:")
    print grid_search_tune.best_estimator_.steps
    print

    # measuring performance on test set
    print "Applying best classifier on test data:"
    best_clf = grid_search_tune.best_estimator_
    predictions = best_clf.predict(x_test)

    print classification_report(y_test, predictions)

    """
                precision    recall  f1-score   support

        false       0.86      0.56      0.68      1474
        true        0.91      0.98      0.94      6363

    avg / total     0.90      0.90      0.89      7837


    Best parameters set:
[('tfidf', TfidfVectorizer(analyzer=u'word', binary=False, decode_error=u'strict',
        dtype=<type 'numpy.int64'>, encoding=u'utf-8', input=u'content',
        lowercase=True, max_df=0.75, max_features=None, min_df=1,
        ngram_range=(1, 1), norm=u'l2', preprocessor=None, smooth_idf=True,
        stop_words=[u'aber', u'alle', u'allem', u'allen', u'aller', u'alles', u'als', u'also', u'am', u'an', u'ander', u'andere', u'anderem', u'anderen', u'anderer', u'anderes', u'anderm', u'andern', u'anderr', u'anders', u'auch', u'auf', u'aus', u'bei', u'bin', u'bis', u'bist', u'da', u'damit', u'dann', u'...u'wo', u'wollen', u'wollte', u'w\xfcrde', u'w\xfcrden', u'zu', u'zum', u'zur', u'zwar', u'zwischen'],
        strip_accents=None, sublinear_tf=False,
        token_pattern=u'(?u)\\b\\w\\w+\\b',
        tokenizer=<function pre_process at 0x3f27d70>, use_idf=True,
        vocabulary=None)), ('clf', LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='sag', tol=0.0001,
          verbose=0, warm_start=False))]

Applying best classifier on test data:
             precision    recall  f1-score   support

      false       0.87      0.61      0.72      1474
       true       0.91      0.98      0.95      6363

avg / total       0.91      0.91      0.90      7837
    """


def predict_sentiment(x_texts, y_classes):

    # train classifiers for predicting relevance
    data_x = [x[1] for x in x_texts]
    data_y = [y[1] for y in y_classes]

    data_x = np.array(data_x)
    data_y = np.array(data_y)

    stratified_split = StratifiedShuffleSplit(n_splits=2, test_size=0.33)
    for train_index, test_index in stratified_split.split(data_x, data_y):
        x_train, x_test = data_x[train_index], data_x[test_index]
        y_train, y_test = data_y[train_index], data_y[test_index]

    stop_words = stopwords.words('german')
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=stop_words, tokenizer=pre_process)),
        ('clf', MultinomialNB(fit_prior=True, class_prior=None)),
    ])

    parameters = {
        'tfidf__max_df': (0.25, 0.5, 0.75),
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'clf__alpha': (1e-2, 1e-3)
    }
    grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=3, verbose=4)
    grid_search_tune.fit(x_train, y_train)

    print
    print("Best parameters set:")
    print grid_search_tune.best_estimator_.steps
    print

    # measuring performance on test set
    print "Applying best classifier on test data:"
    best_clf = grid_search_tune.best_estimator_
    predictions = best_clf.predict(x_test)
    print classification_report(y_test, predictions)

    """
             precision    recall  f1-score   support

   negative       0.62      0.52      0.57      1945
    neutral       0.81      0.88      0.84      5431
   positive       0.58      0.36      0.44       461

avg / total       0.75      0.76      0.75      7837

    """


def text_analysis(x_texts):
    # How many different sources ?
    for x in x_texts:
        print x[0], len(x[1])
        """
        lang = langdetect.detect(x[1])
        if lang != 'de':
            print x[1]
            print lang
            print
        """


def pre_process(text):
    tokens = tokens_re.findall(text)
    tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens


def main():

    x_texts = []
    y_classes = []

    print "Reading data..."
    read_xml('data_xml/train.xml', x_texts, y_classes)
    read_xml('data_xml/trial.xml', x_texts, y_classes)
    read_xml('data_xml/dev.xml', x_texts, y_classes)
    print len(x_texts), "documents read"

    # predict_relevance(x_texts, y_classes)
    # predict_sentiment(x_texts, y_classes)
    text_analysis(x_texts)

if __name__ == "__main__":
    main()
