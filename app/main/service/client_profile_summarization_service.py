import requests
from flask_restplus import abort
from summarizer import Summarizer
from textblob import TextBlob
from .base_services import text_cleaning, parse_article

import spacy
nlp = spacy.load("en_core_web_lg")
def gather_content_data(url_list):
    if len(url_list) > 0:
        corpus = []
        for url in url_list:
            content = parse_article(requests.get(url).text)
            if bool(content):
                corpus.append(' '.join(content))
            else:
                print('No content present: ', url)
        spacy_text_list = text_cleaning(' '.join(corpus), remove_stopwords=False)
        return ' '.join(spacy_text_list)
    else:
        abort(400, 'Empty Url List')


def calculate_similarity_score(full, kyc_doc):
    """
    :param full:
    :param kyc_doc:
    :return:
    """
    extraction = nlp(full)
    benchmark = nlp(kyc_doc)
    similarity_score = extraction.similarity(benchmark)
    return similarity_score


def bert_summarizer(source_url_list, kyc_doc):
    """
    :param source_url_list: list of valid urls to scrape
    :param kyc_doc: profile summary of client from documentum
    :return: similarity score , polarity
    """
    corpus = gather_content_data(url_list=source_url_list)
    model = Summarizer()
    result = model(corpus, min_length=10, ratio=0.5, algorithm='gmm', max_length=200)
    full = ''.join(result)
    similarity_score = calculate_similarity_score(full, kyc_doc)
    testimonial = TextBlob(full)
    polarity = testimonial.sentiment.polarity
    return full, similarity_score, polarity
