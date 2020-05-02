import re
from collections import defaultdict

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
    #     spacy_text_list = random.sample(spacy_text_list, len(spacy_text_list))
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


def gather_content_data(url_list):
    assert len(url_list) > 0
    corpus = []
    for url in url_list:
        content = parse_article(requests.get(url).text)
        if bool(content):
            corpus.append(' '.join(content))
    spacy_text_list = text_cleaning(' '.join(corpus))
    return ' '.join(spacy_text_list)


def __parse_address(match_text):
    address = defaultdict(OrderedSet)
    for match in match_text:
        _addrs = OrderedSet(
            parse_address(address=' '.join(list(match))))
        if bool(_addrs):
            for item in _addrs:
                address[item[1]].add(item[0])

    def set_default(obj):
        if isinstance(obj, OrderedSet):
            return list(obj)
        raise TypeError

    import json
    result = json.dumps(address, default=set_default)
    return result
