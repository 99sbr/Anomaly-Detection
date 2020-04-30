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
result = model(corpus, min_length=30, ratio=0.5, max_length=len(corpus))
full = ''.join(result)
print(full)
testimonial = TextBlob(full)
print('\n Polarity of Article:', testimonial.sentiment.polarity)
```

    Unique Entity Number: 201309826H FICOFI PARTNERS HOLDING PTE. The company was registered / incorporated on 12 April 2013 (Friday), 7 years ago The address of this company registered office is 25 INTERNATIONAL BUSINESS PARK #03-01/02 GERMAN CENTRE SINGAPORE 609916 The company has 7 officers / owners / shareholders. The company secondary activity is MANAGEMENT CONSULTANCY SERVICES N.E.C.. Ficofi Partners Holding Pte. It operates in the Management of Companies and Enterprises sector. It was incorporated on April 12, 2013. Ltd. report to view the information. EMIS company profiles are part of a larger information service which combines company, industry and country data and analysis for over 145 emerging markets. Marshall Cavendish Business Information Pte Ltd. All Rights Reserved. The entity status is Live Company. Each entity is registered with unique entity number (UEN), entity name, entity time, UEN issue date, location, etc.
    
     Polarity of Article: 0.08376623376623378


### Similarity Score Calculation using spaCy


```python
extraction = spacy_nlp(full)
similarity_score = extraction.similarity(kyc_doc)
print('The Similarity Score of Sumaarized Text is: %d', similarity_score)
```

    The Similarity Score of Sumaarized Text is: %d 0.930997122566286



```python

```
