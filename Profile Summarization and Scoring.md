### Import Libraries and Models


```python
from pprint import pprint
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

    2020-05-05 16:12:23 INFO: Loading these models for language: en (English):
    =========================
    | Processor | Package   |
    -------------------------
    | tokenize  | ewt       |
    | pos       | ewt       |
    | lemma     | ewt       |
    | depparse  | ewt       |
    | ner       | ontonotes |
    =========================
    
    2020-05-05 16:12:23 INFO: Use device: cpu
    2020-05-05 16:12:23 INFO: Loading: tokenize
    2020-05-05 16:12:23 INFO: Loading: pos
    2020-05-05 16:12:24 INFO: Loading: lemma
    2020-05-05 16:12:24 INFO: Loading: depparse
    2020-05-05 16:12:25 INFO: Loading: ner
    2020-05-05 16:12:26 INFO: Done loading processors!


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
    "https://www.emis.com/php/company-profile/SG/Ficofi_Partners_Holding_Pte_Ltd_en_6690179.html",
    "https://www.timesbusinessdirectory.com/companies/ficofi-partners-holding-pte-ltd",
    "https://opengovsg.com/corporate/201309826H",
    "https://recordowl.com/company/ficofi-partners-holding-pte-ltd",
    "https://www.singaporecontacts.com/companies/ficofi-partners-holding-pte-ltd-singapore/84b81fc9-5b2b-e993-a43e-5b4acb7e0366",
    "https://singapore-corp.com/co/ficofi-partners-holding-pte-ltd"
]
```

### HTML Parsing and Text Cleaning


```python
def text_cleaning(raw_text):
    raw_text_list = raw_text.split('\n')
#     raw_text_list = [
#         token for token in raw_text_list if token not in stop_words
#     ]
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
        print(url)
        content = parse_article(requests.get(url).text)
        if bool(content):
            corpus.append(' '.join(content))
    spacy_text_list = text_cleaning(' '.join(corpus))
    return ' '.join(spacy_text_list)
```


```python
corpus = gather_content_data(url_list=source_url_list)
```

    https://www.emis.com/php/company-profile/SG/Ficofi_Partners_Holding_Pte_Ltd_en_6690179.html
    https://www.timesbusinessdirectory.com/companies/ficofi-partners-holding-pte-ltd
    https://opengovsg.com/corporate/201309826H
    https://recordowl.com/company/ficofi-partners-holding-pte-ltd
    https://www.singaporecontacts.com/companies/ficofi-partners-holding-pte-ltd-singapore/84b81fc9-5b2b-e993-a43e-5b4acb7e0366
    https://singapore-corp.com/co/ficofi-partners-holding-pte-ltd



```python
corpus
```




    'Ficofi Partners Holding Pte. Ltd. is an enterprise located in Singapore, with the main office in Singapore. It operates in the Management of Companies and Enterprises sector. It was incorporated on April 12, 2013. Headquarters 25 International Business Park #03-01/02 German Centre Singapore 609916 Singapore; Singapore; Postal Code: 609916 Contact Details: Purchase the Ficofi Partners Holding Pte. Ltd. report to view the information. EMIS company profiles are part of a larger information service which combines company, industry and country data and analysis for over 145 emerging markets. To view more information, Request a demonstration of the EMIS service FICOFI PARTNERS HOLDING PTE LTD 25 International Business Park #03-01/02 German Centre Singapore 609916 Copyright © 2020. Marshall Cavendish Business Information Pte Ltd. All Rights Reserved. FICOFI PARTNERS HOLDING PTE. LTD. (UEN ID 201309826H) is a corporate entity registered with Accounting and Corporate Regulatory Authority. The UEN issue date is April 13, 2013. The entity status is Live Company. Please comment or provide details below to improve the information on . This dataset includes 1.5 million corporate entities registered with Accounting and Corporate Regulatory Authority (ACRA), Singapore. Each entity is registered with unique entity number (UEN), entity name, entity time, UEN issue date, location, etc. Registration No. / Unique Entity Number: 201309826H FICOFI PARTNERS HOLDING PTE. LTD. (the "Company") The Company is a PRIVATE COMPANY LIMITED BY SHARES and it\'s current status is Live Company. The company was registered / incorporated on 12 April 2013 (Friday), 7 years ago The address of this company registered office is 25 INTERNATIONAL BUSINESS PARK #03-01/02 GERMAN CENTRE SINGAPORE 609916 The company has 7 officers / owners / shareholders. The company principal activity is OTHER HOLDING COMPANIES. The company secondary activity is MANAGEMENT CONSULTANCY SERVICES N.E.C.. HOT LEADS / LISTS HR Managers eMail Lists HR Managers Database HR-VP, Directors Database Finance Managers eMail Lists Finance Managers Database IT Managers eMail Lists IT Managers Database CIO / CTO Database CFO / Director - Finance eMail Lists CFO / Director - Finance Database Sales Managers eMail Lists Sales Managers Database Sales-VP, Director Database CEO, President, MD eMail Lists Directors (Share Holding) eMail Lists Marketing Managers eMail Lists Marketing Managers Database Marketing-VP, Director Database CEO, President, MD Database Directors (Share Holding) Database Home | Enquiry | FAQ | Contact Copyright © 2020 Singapore Contacts. All Rights Reserved.'



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

    Ltd. is an enterprise located in Singapore, with the main office in Singapore. It operates in the Management of Companies and Enterprises sector. It was incorporated on April 12, 2013. EMIS company profiles are part of a larger information service which combines company, industry and country data and analysis for over 145 emerging markets. To view more information, Request a demonstration of the EMIS service FICOFI PARTNERS HOLDING PTE LTD 25 International Business Park #03-01/02 German Centre Singapore 609916 Copyright © 2020. Marshall Cavendish Business Information Pte Ltd. All Rights Reserved. The entity status is Live Company. Each entity is registered with unique entity number (UEN), entity name, entity time, UEN issue date, location, etc. The company secondary activity is MANAGEMENT CONSULTANCY SERVICES N.E.C.. HOT LEADS / LISTS HR Managers eMail Lists HR Managers Database HR-VP, Directors Database Finance Managers eMail Lists Finance Managers Database IT Managers eMail Lists IT Managers Database CIO / CTO Database CFO / Director - Finance eMail Lists CFO / Director - Finance Database Sales Managers eMail Lists Sales Managers Database Sales-VP, Director Database CEO, President, MD eMail Lists Directors (Share Holding) eMail Lists Marketing Managers eMail Lists Marketing Managers Database Marketing-VP, Director Database CEO, President, MD Database Directors (Share Holding) Database Home | Enquiry | FAQ | Contact Copyright © 2020 Singapore Contacts.
    
     Polarity of Article: 0.12533670033670033


### Similarity Score Calculation using spaCy


```python
extraction = spacy_nlp(full)
similarity_score = extraction.similarity(kyc_doc)
print('The Similarity Score of Summarized Text is: ', similarity_score*100)
```

    The Similarity Score of Summarized Text is:  88.67057066260796



```python
"""
China Sci-Tech Holdings Ltd
American express singapore
DHL Express singapore
Hitachi captial asia pacific pvt ltd
"""
```




    '\nChina Sci-Tech Holdings Ltd\nAmerican express singapore\nDHL Express singapore\nHitachi captial asia pacific pvt ltd\n'



### Hitting Profile Summarizarion API


```python
request_url ="http://0.0.0.0:1234/client/get-summary"
```


```python
input_data = {
    "ProfileSummaryBenchmark":
    "Holding company activities and collection center for FICOFI which is engaged in Import, Distribution and Sales of wines and spirits. The group has also centralized its operations in Singapore and setup a global treasury/ collection center based here.The primary reason for this decision was that the group has lot of suppliers and clients who are commoon across various entities. When thecliens make payments they usually make one lumpsum payment for various invoices. To ensure that they streamline the process for their clients,ficofi has decides that they will start with centraliznig the collecttion process- collect funds from clients into accounts with SG.",
    "SearchUrlList": [
        "https://recordowl.com/company/ficofi-partners-holding-pte-ltd",
        "https://www.emis.com/php/company-profile/SG/Ficofi_Partners_Holding_Pte_Ltd_en_6690179.html",
        "https://www.timesbusinessdirectory.com/companies/ficofi-partners-holding-pte-ltd",
        "https://opengovsg.com/corporate/201309826H",
        "https://www.singaporecontacts.com/companies/ficofi-partners-holding-pte-ltd-singapore/84b81fc9-5b2b-e993-a43e-5b4acb7e0366",
        "https://singapore-corp.com/co/ficofi-partners-holding-pte-ltd"
    ]
}
```


```python
print('Hitting Profile Summarization API')
response = requests.post(url=request_url,json=input_data)
```

    Hitting Profile Summarization API



```python
print('The Profile Summary of FICOFI PARTNERS HOLDING PTE extracted from web: \n', str(response.json()['Summary']))
```

    The Profile Summary of FICOFI PARTNERS HOLDING PTE extracted from web: 
     Unique Entity Number: 201309826H FICOFI PARTNERS HOLDING PTE. The company was registered / incorporated on 12 April 2013 (Friday), 7 years ago The address of this company registered office is 25 INTERNATIONAL BUSINESS PARK #03-01/02 GERMAN CENTRE SINGAPORE 609916 The company has 7 officers / owners / shareholders. Ltd. is an enterprise located in Singapore, with the main office in Singapore. It operates in the Management of Companies and Enterprises sector. It was incorporated on April 12, 2013. Ltd. report to view the information. EMIS company profiles are part of a larger information service which combines company, industry and country data and analysis for over 145 emerging markets. To view more information, Request a demonstration of the EMIS service FICOFI PARTNERS HOLDING PTE LTD 25 International Business Park #03-01/02 German Centre Singapore 609916 Copyright © 2020. Marshall Cavendish Business Information Pte Ltd. All Rights Reserved. UEN ID 201309826H) is a corporate entity registered with Accounting and Corporate Regulatory Authority. The entity status is Live Company. Please comment or provide details below to improve the information on . HOT LEADS / LISTS HR Managers eMail Lists HR Managers Database HR-VP, Directors Database Finance Managers eMail Lists Finance Managers Database IT Managers eMail Lists IT Managers Database CIO / CTO Database CFO / Director - Finance eMail Lists CFO / Director - Finance Database Sales Managers eMail Lists Sales Managers Database Sales-VP, Director Database CEO, President, MD eMail Lists Directors (Share Holding) eMail Lists Marketing Managers eMail Lists Marketing Managers Database Marketing-VP, Director Database CEO, President, MD Database Directors (Share Holding) Database Home | Enquiry | FAQ | Contact Copyright © 2020 Singapore Contacts.



```python
print('The Profile Summary of FICOFI PARTNERS HOLDING PTE extracted from web: \n', str(response.json()['Similarity Score']))
```

    The Profile Summary of FICOFI PARTNERS HOLDING PTE extracted from web: 
     0.8905983419196403



```python
print('The Profile Summary of FICOFI PARTNERS HOLDING PTE extracted from web: \n', str(response.json()['Polarity']))
```

    The Profile Summary of FICOFI PARTNERS HOLDING PTE extracted from web: 
     0.11900252525252523



```python

```
