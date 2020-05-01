import re
import requests
import spacy
import stanza
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from spacy_stanza import StanzaLanguage
from summarizer import Summarizer
from textblob import TextBlob

stop_words = stopwords.words('english')
snlp = stanza.Pipeline(lang="en")
stanza_nlp = StanzaLanguage(snlp)
spacy_nlp = spacy.load('en_core_web_lg')


def text_cleaning(raw_text):
    raw_text_list = raw_text.split('\n')
    raw_text_list = [
        token for token in raw_text_list if token not in stop_words
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


def calculate_similarity_score(full, kyc_doc):
    extraction = spacy_nlp(full)
    benchmark = spacy_nlp(kyc_doc)
    similarity_score = extraction.similarity(benchmark)
    return similarity_score


def bert_summarizer(source_url_list, kyc_doc):
    '''
    :param source_url_list: list of valid urls to scrape
    :param kyc_doc: profile summary of client from documentum
    :return: similarity score , polarity
    '''
    corpus = gather_content_data(url_list=source_url_list)
    model = Summarizer()
    result = model(corpus, min_length=30, ratio=0.5, max_length=len(corpus))
    full = ''.join(result)
    similarity_score = calculate_similarity_score(full, kyc_doc)
    testimonial = TextBlob(full)
    polarity = testimonial.sentiment.polarity
    return full, similarity_score, polarity
