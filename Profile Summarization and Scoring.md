### Import Libraries and Models


```python
from spacy_stanza import StanzaLanguage
import stanza
import spacy
import nltk
import tensorflow as tf
import re
import random
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import requests
from bs4 import BeautifulSoup
import html2text
from summarizer import Summarizer
from textblob import TextBlob
stop_words = stopwords.words('english')
snlp = stanza.Pipeline(lang="en")
stanza_nlp = StanzaLanguage(snlp)
spacy_nlp = spacy.load('en_core_web_lg')
```

    /Users/subir/pythonenv/default/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:541: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint8 = np.dtype([("qint8", np.int8, 1)])
    /Users/subir/pythonenv/default/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:542: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
    /Users/subir/pythonenv/default/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:543: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint16 = np.dtype([("qint16", np.int16, 1)])
    /Users/subir/pythonenv/default/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:544: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
    /Users/subir/pythonenv/default/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:545: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint32 = np.dtype([("qint32", np.int32, 1)])
    /Users/subir/pythonenv/default/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:550: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      np_resource = np.dtype([("resource", np.ubyte, 1)])
    2020-05-04 11:04:11 INFO: Loading these models for language: en (English):
    =========================
    | Processor | Package   |
    -------------------------
    | tokenize  | ewt       |
    | pos       | ewt       |
    | lemma     | ewt       |
    | depparse  | ewt       |
    | ner       | ontonotes |
    =========================
    
    2020-05-04 11:04:11 INFO: Use device: cpu
    2020-05-04 11:04:11 INFO: Loading: tokenize
    2020-05-04 11:04:11 INFO: Loading: pos
    2020-05-04 11:04:12 INFO: Loading: lemma
    2020-05-04 11:04:12 INFO: Loading: depparse
    2020-05-04 11:04:14 INFO: Loading: ner
    2020-05-04 11:04:14 INFO: Done loading processors!


### Defining Client Profile Summary from Documentum


```python
fromkyc = "Holding company activities and collection center for FICOFI which is engaged in Import, Distribution and Sales of wines\
and spirits. The group has also centralized its operations in Singapore and setup a global treasury/ collection centre based here.\
The primary reason for this decision was that the group has lot of suppliers and clients who are commoon across various entities. When the\
cliens make payments they usually make one lumpsum payment for various invoices. To ensure that they streamline the process for their clients,\
ficofi has decides that they will start with centraliznig the collecttion process- collect funds from clients into accounts with SG."

kyc_doc = spacy_nlp(fromkyc.strip())
```

### Source URL list to crawl


```python
source_url_list = [
    "https://recordowl.com/company/ficofi-partners-holding-pte-ltd",
    "https://www.emis.com/php/company-profile/SG/Ficofi_Partners_Holding_Pte_Ltd_en_6690179.html",
    "https://www.timesbusinessdirectory.com/companies/ficofi-partners-holding-pte-ltd",
    "https://opengovsg.com/corporate/201309826H"
]
```

### HTML Parsing and Text Cleaning


```python
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
    import random
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
```

### Creating Corpus of Information from WEB


```python
def gather_content_data(url_list):
    assert len(url_list) > 0
    corpus = []
    for url in url_list:
        content = parse_article(requests.get(url).text)
        if bool(content):
            corpus.append(' '.join(content))
    spacy_text_list = text_cleaning(' '.join(corpus))
    return ' '.join(spacy_text_list)
```


```python
corpus = gather_content_data(url_list=source_url_list)
```

### BERT based Text Summarization


```python
model = Summarizer()
```


```python
result = model(corpus, min_length=30, algorithm='gmm',ratio=0.5, max_length=len(corpus))
full = ''.join(result)
print(full)
testimonial = TextBlob(full)
print('\n Polarity of Article:', testimonial.sentiment.polarity)
```

    Unique Entity Number: 201309826H FICOFI PARTNERS HOLDING PTE. The company was registered / incorporated on 12 April 2013 (Friday), 7 years ago The address of this company registered office is 25 INTERNATIONAL BUSINESS PARK #03-01/02 GERMAN CENTRE SINGAPORE 609916 The company has 7 officers / owners / shareholders. The company secondary activity is MANAGEMENT CONSULTANCY SERVICES N.E.C.. Ficofi Partners Holding Pte. Ltd. is an enterprise located in Singapore, with the main office in Singapore. It was incorporated on April 12, 2013. Ltd. report to view the information. EMIS company profiles are part of a larger information service which combines company, industry and country data and analysis for over 145 emerging markets. Marshall Cavendish Business Information Pte Ltd. All Rights Reserved. The entity status is Live Company. Please comment or provide details below to improve the information on .
    
     Polarity of Article: 0.054004329004329006


### Similarity Score Calculation using spaCy


```python
extraction = spacy_nlp(full)
similarity_score = extraction.similarity(kyc_doc)
print('The Similarity Score of Summarized Text is: ', similarity_score*100)
```

    The Similarity Score of Summarized Text is:  93.13271240399268



```python

```
