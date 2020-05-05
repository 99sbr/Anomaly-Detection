from itertools import combinations

import requests
from flask_restplus import abort
from ordered_set import OrderedSet

from .base_services import text_cleaning, __parse_address, parse_article


def main_call(base_url_list, client_name):
    """

    :param base_url_list:
    :param client_name:
    :return:
    """
    corpus_token_list = []
    if len(base_url_list) > 0:
        for url in base_url_list:
            clean_text_list = parse_article(requests.get(url).text)
            if bool(clean_text_list):
                clean_corpus = text_cleaning(' '.join(clean_text_list), remove_stopwords=True)
                corpus_token_list.append(clean_corpus)

        match_text = get_jaccard_sim_multi(' '.join(corpus_token_list[0]), ' '.join(corpus_token_list[1]),
                                           ' '.join(corpus_token_list[2]), ' '.join(corpus_token_list[3]), client_name)
        address = __parse_address(match_text)
        return address
    else:
        abort(400, 'Empty URL List')


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
