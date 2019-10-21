#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import xml.etree.ElementTree as ET

from spacy.de import German
from collections import defaultdict

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

categories = defaultdict(list)
categories['Allgemein']
categories['Atmosphäre']
categories['Lautstärke']
categories['Beleuchtung']
categories['Fahrgefühl']
categories['Temperatur']
categories['Sauberkeit allgemein']
categories['Geruch']
categories['Connectivity']
categories['WLAN/Internet']
categories['Telefonie/Handyempfang']
categories['ICE Portal']
categories['Design']

print "Initializing spaCy German parser..."
parser = German()


def read_xml(input_file):
    tree = ET.parse(input_file)
    root = tree.getroot()
    for doc in root.findall('Document'):
        text = doc.find('text').text
        relevance = doc.find('relevance').text
        sentiment = doc.find('sentiment').text

        """
        print text.encode("utf8")
        print doc.attrib
        print "relevance :", relevance
        print "sentiment :", sentiment
        """
        opinions = doc.find('Opinions')
        if opinions is not None:
            print text.encode("utf8")
            for opinion in opinions.findall('Opinion'):
                # print opinion.attrib
                print opinion.attrib['category'].split("#")[0].encode("utf8"),
                print "\t",
                print opinion.attrib['target'].encode("utf8"),
                print opinion.attrib['polarity'].encode("utf8")

            print

            if isinstance(text, str):
                text = text.decode("utf8")

            parsed = parser(text, tag=True, parse=True, entity=True)

            print "entities:", parsed.ents
            print
            print "noun_chunks: "
            for noun in parsed.noun_chunks:
                print noun
            print "sentences: "
            print
            for s in parsed.sents:
                print s
                print "part-of-speech: "
                for token in s:
                    print(token.orth_, token.pos_, token.lemma_)
                print
                print "dependencies: "
                print('\n'.join(
                    '{child:<8} <{label:-^7} {head}'.format(child=t.orth_.encode("utf8"),
                                                            label=t.dep_.encode("utf8"),
                                                            head=t.head.orth_.encode("utf8"))
                    for t in s))
                print

            print "=================================="
            print


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
    print "Reading data...\n"
    read_xml('data_xml/train.xml')
    read_xml('data_xml/trial.xml')
    read_xml('data_xml/dev.xml')

if __name__ == "__main__":
    main()
