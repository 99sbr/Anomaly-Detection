import spacy
from summarizer import Summarizer
from textblob import TextBlob

from .base_services import gather_content_data

spacy_nlp = spacy.load('en_core_web_lg')


# noinspection PySingleQuotedDocstring
def calculate_similarity_score(full, kyc_doc):
    """
    :param full:
    :param kyc_doc:
    :return:
    """
    extraction = spacy_nlp(full)
    benchmark = spacy_nlp(kyc_doc)
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
    result = model(corpus, min_length=30, ratio=0.5, max_length=len(corpus))
    full = ''.join(result)
    similarity_score = calculate_similarity_score(full, kyc_doc)
    testimonial = TextBlob(full)
    polarity = testimonial.sentiment.polarity
    return full, similarity_score, polarity
