import random
import re
from collections import defaultdict
from itertools import combinations

import requests
import spacy
import stanza
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from ordered_set import OrderedSet
from postal.parser import parse_address
from sentence_transformers import SentenceTransformer
from spacy_stanza import StanzaLanguage
from summarizer import Summarizer

snlp = stanza.Pipeline(lang="en")
stanza_nlp = StanzaLanguage(snlp)
spacy_nlp = spacy.load('en_core_web_lg')
bert_sent_model = SentenceTransformer('roberta-base-nli-mean-tokens')
stopwords = set(stopwords.words('english'))
model = Summarizer()


def __parse_address(match_text):
    address = defaultdict(OrderedSet)
    for match in match_text:
        _addrs = OrderedSet(
            parse_address(address=' '.join(list(match))))
        if bool(_addrs):
            for item in _addrs:
                address[item[1]].add(item[0])
    return address


def main_call(base_url_list, client_name):
    corpus_token_list = []
    for url in base_url_list:
        clean_text_list = parse_article(requests.get(url).text)
        if bool(clean_text_list):
            clean_corpus = text_cleaning(' '.join(clean_text_list))
            corpus_token_list.append(clean_corpus)

    match_text = get_jaccard_sim_multi(' '.join(corpus_token_list[0]), ' '.join(corpus_token_list[1]),
                                       ' '.join(corpus_token_list[2]), ' '.join(corpus_token_list[3]), client_name)
    address = __parse_address(match_text)
    return address


def get_jaccard_sim_multi(str1, str2, str3, str4, client_name):
    a = OrderedSet(str1.lower().split())
    b = OrderedSet(str2.lower().split())
    c = OrderedSet(str3.lower().split())
    d = OrderedSet(str4.lower().split())

    all_combinations = list(combinations([a, b, c, d], 2))
    e = []
    for x in all_combinations:
        e.append(
            OrderedSet(x[0].intersection(x[1])) - set(client_name.split()))
    return e


def text_cleaning(raw_text):
    raw_text_list = raw_text.split('\n')
    raw_text_list = [
        token for token in raw_text_list if token not in stopwords
    ]
    clean_sent_list = [
        re.sub('[^A-Za-z0-9]+\.-/', '', token) for token in raw_text_list
        if bool(token)
    ]
    clean_sent = ' '.join(clean_sent_list)
    clean_sent = ' '.join(clean_sent.split())
    doc = stanza_nlp(clean_sent)
    spacy_text_list = []
    for sent in doc.sents:
        spacy_text_list.append(sent.text)

    spacy_text_list = random.sample(spacy_text_list, len(spacy_text_list))
    return spacy_text_list


def tag2text(tag):
    if tag.name == 'p':
        return tag.text


def parse_article(text):
    soup = BeautifulSoup(text, 'html.parser')
    # find the article title
    h1 = soup.find('h1')
    # find the common parent for <h1> and all <p>s.
    root = h1
    while root.name != 'body':
        if root.parent == None:
            break
        root = root.parent
    # find all the content elements.
    ps = root.find_all(['h2', 'h3', 'h4', 'h5', 'h6', 'p', 'pre'])
    ps.insert(0, h1)
    content = [tag2text(p) for p in ps]
    content = [x for x in content if bool(x)]
    return content
