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
    2020-05-15 14:27:35 INFO: Loading these models for language: en (English):
    =========================
    | Processor | Package   |
    -------------------------
    | tokenize  | ewt       |
    | pos       | ewt       |
    | lemma     | ewt       |
    | depparse  | ewt       |
    | ner       | ontonotes |
    =========================
    
    2020-05-15 14:27:35 INFO: Use device: cpu
    2020-05-15 14:27:35 INFO: Loading: tokenize
    2020-05-15 14:27:35 INFO: Loading: pos
    2020-05-15 14:27:36 INFO: Loading: lemma
    2020-05-15 14:27:37 INFO: Loading: depparse
    2020-05-15 14:27:38 INFO: Loading: ner
    2020-05-15 14:27:39 INFO: Done loading processors!



```python
"""
shv holdings
"""
```




    '\nWINSON OIL TRADING PTE. LTD\n'



### Defining Client Profile Summary from Documentum


```python
fromkyc = "SHV is a family-owned, decentralised company active in energy distribution, cash-and-carry wholesale, heavy lifting and transport activities, industrial services, animal nutrition and aquafeed, exploration, development and production of oil and gas and providing private equity through its seven companies: SHV Energy, Makro, Mammoet, ERIKS, Nutreco, ONE-Dyas and NPM Capital. SHV employs more than 60,000 people and is present in 58 countries."

kyc_doc = spacy_nlp(fromkyc.strip())
```

### Source URL list to crawl


```python
source_url_list = [
    "https://www.referenceforbusiness.com/history2/94/SHV-Holdings-N-V.html",
    "https://www.fis.com/fis/companies/details.asp?l=e&company_id=158503",
    "https://www.wikiwand.com/en/SHV_Holdings"
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

    try:
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
    except Exception as e:
        print(e)
        return None
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

    https://www.referenceforbusiness.com/history2/94/SHV-Holdings-N-V.html
    https://www.fis.com/fis/companies/details.asp?l=e&company_id=158503
    'NoneType' object has no attribute 'name'
    https://www.wikiwand.com/en/SHV_Holdings



```python
corpus
```




    'Rijnkade 13511 LC Utrecht The Netherlands Company Perspectives: SHV is a privately held company and wishes to remain so. SHV is a decentralised company. Great trust is placed in our people in the field. This decentralisation provides an excellent opportunity for individual development. Mutual respect and trust provide the basis for happiness at work. SHV\'s most important values are integrity and loyalty. Integrity means being honest, genuine, and totally open in communications about all matters that concern the company. Good news may travel slowly, bad news should travel quickly. Loyalty means putting your best effort into your work for the company and its development. Based on the integrity and loyalty of our people, SHV wishes to continue to grow both for the benefit of our shareholders, our employees, and for the well-being of the society in which we live and work. SHV Holdings N.V. is one of the Netherlands\' largest private companies. The family-owned concern serves a holding company for its interests in liquid petroleum gas, wholesale foods and consumer goods, private equity, recycling materials and products, and oil and gas exploration. Paris-based SHV Gas overseas the company\'s LPG interests, with operations in 23 countries, under the Primagaz, Calor, and other names. Makro represents the group\'s interests in the wholesale cash-and-carry market, focused on the Southeast Asian and South American markets (the company sold its European chain to Germany\'s Metro in the 1990s). NPM Capital is SHV\'s equity investment arm, with a focus on European venture capital investments. The company also owns one of the world\'s largest recycling businesses, through U.S.-based David J. Joseph Company and TSR, in Germany, formerly part of the Thyssen group. Lastly, SHV has been involved in the oil and gas exploration business off the Netherlands\' coastline through Dyas, a "non-operator" that provides investment and partnership to other companies. SHV Holdings is owned by the Fentener van Vlissingen family, which remains committed to the company\'s privately owned status. The company\'s exposure to slumping South American currencies caused its sales to drop to EUR 9.4 billion in 2002 (from 10.4 billion in 2001). Coal Cartel at the Turn of the Century The Fentener van Vlissingen family was already one of the Netherlands\' most prominent coal trading families when, together with eight other Dutch trading families, it formed the Steenkolen Handels-Vereeniging ("Coal Trading Association") in Utrecht in 1896. A number of the original partners in SHV had been active in the coal trade since the 18th century. In addition to the formation of SHV, the families formed bonds through a number of marriages--this was the case, for example, between the Fentener van Vlissingen and van Beuningen families, which gained control of the SHV by the beginning of World War II. In 1904, SHV acquired the exclusive trading rights for the coal produced by the Rheinisch-Westfälisches Kohlen Syndikat, which controlled the important Ruhr region coal production. Two years later, the SHV gained the exclusive rights to coal transports along the Netherlands\' inland waterways, including the Rhine river, setting up the first of its subsidiary businesses to conduct those operations. SHV inland waterways fleet, later regrouped under the name Nederlandsche Rijnvaart Vereniging, grew into the world\'s largest by the 1950s. Much of the growth of SHV came under the leadership of Frederik "Frits" Fentener van Vlissingen II, who joined the company in 1904 at the age of 22, taking over from his ailing father. By 1911, the younger Fentener van Vlissingen had been named the company\'s director. In that same year, Fentener van Vlissingen led the company in a new direction, setting up the Hollandsche Kunstzijde Industrie, a producer of synthetic fabrics, in part because the heavy coal consumption needed by that industry made the new company an important customer for SHV\'s coal interests. Financial backing for the Hollandsche Kunstzijde came from the German company Vereignigte Glanzstoff, which itself was owned by a member of the SHV partnership. Under Fenterner van Vlissingen--later called one of the fathers of the Dutch economy--SHV established its own investment capital company, called Unitas, which provided funding for the creation of new companies. As such, SHV became the motor behind the establishment of such major Dutch companies as KLM, Hoogovens, Akzo, and the Nederlandsche Vliegtuigenfabriek, as well as Germany\'s Vereinigte Stahlwerke. SHV grew strongly, adding offices in Rotterdam and Amsterdam; in 1913, the company transferred its headquarters to a new office in Utrecht, which was to remain the company\'s home into the 21st century. The Netherlands\' neutral status during World War I enabled it to continue trading, particularly given its close ties with Germany\'s coal and other industries. In 1917, SHV, through Unitas, itself moved into mining in Germany, setting up Nemos (Netherlandsche Maatschappij tot Ontginning van de Steenkolenvelden) to operate the Sophia Jacoba anthracite mine. That operation led SHV, which had functioned primarily as an importer, to begin exporting coal as well. From its start in 1924, SHV\'s coal export operation grew to become one of the world\'s leaders in that market. The rising importance of oil did not go unnoticed by SHV, however. By the late 1930s, the company had begun trading in that area as well, setting up a fuel facility; the company also took a majority share of the VEM (Verenigde Exploitatiemij) shipping company--a merger of three fishing companies--the trawler fleet of which became an early customer for SHV\'s oil. Nonetheless, SHV remained reliant on its coal interests into the 1950s, when the rising dominance of petroleum forced it to look elsewhere for its future growth. By then, the Fentener van Vlissingen family had bought out the share in SHV held by the van Beuningen family, becoming the company\'s sole owner. Transforming in the 1970s In adapting to the new market conditions, SHV showed a flexibility that was to become a company hallmark. By the end of the 1950s, SHV had embraced the oil industry, acquiring oil interests in Austria and Italy while building up an important oil bunker business. The company adopted a new trade name for its oil activities, PAM, under which SHV began selling petroleum products. SHV also built up a network of gasoline stations, which extended from the Netherlands into Germany and Austria, under the PAM name. In addition, under a joint venture, Calpam, SHV began providing heating oil for the home and industrial markets in Belgium, Demark, and Luxembourg. SHV\'s fuel interests soon turned toward a new type, however. A large natural gas deposit had been discovered in the Netherland\'s northern Groningen region at the end of the 1950s. In 1963, SHV set up a new subsidiary, Dyas--taken from the name given to the rock layer beneath which natural gas deposits were often found--to begin exploration activities. Dyas preferred the role of "non-operator," providing financial and other support services to its operator partners. One of the first of these was formed with Amoco and Gelsenkirchen (later Veba), which discovered its first gas deposit in the so-called Bergen Concession, near Amsterdam, in 1964. That same year, Paul Fentener van Vlissingen joined the company, by then led by brother Frits Fentener van Vlissingen III. The younger Fentener van Vlissingen quickly showed himself an energetic leader and was largely credited with transforming the company after the collapse of the coal industry by leading it on a diversification drive. This process began in earnest at the end of the 1960s. Electronics became an important new area of interest for SHV with the acquisition of Geveke & Groenpol NV in 1970. SHV split up Geveke & Groenpol into two operations. The first part took over the Geveke & Groenpol installation business, which was placed, alongside other acquisitions, including into a new subsidiary, Group Technical Installation, or GTI. Meanwhile, Geveke & Groenpol\'s sales and distribution operation were reformed under the name Groenpol Industriële Verkoop, which later became the basis of Geveke Electronics (later known as Getronics). SHV\'s foray into electronics lasted only until the early 1980s, when it spun off GTI and Geveke in management buyouts by 1983. Instead, SHV had begun building a number of other growing operations. The first of these was the Makro wholesale cash-and-carry chain, which the company had launched, in partnership with Germany\'s Metro supermarket group, in 1968. Makro was inspired by the American example of self-service warehouse stores and opened its first branch in Amsterdam. The concept caught on quickly, and Makro began expanding throughout the Netherlands, then throughout Europe, in conjunction with partner Metro. Makro also entered the South American market in the early 1970s. By the 1980s, Makro had entered the United States as well, although in 1988 SHV sold 51 percent of Makro Inc. to the Kmart Corporation. The success of the Makro concept encouraged SHV to try its hand at other retailing concepts, acquiring, in 1970, the Otto Reichelt chain--which was to form the company\'s largest retail operation into the mid-1990s--as well as the Netherlands formats Kijkshop and Xenos. Another diversification area was recycling, which the company entered with the 1975 purchase of the David J. Joseph Company, which had stemmed from a company founded in Ohio in 1863 and which had grown into the largest scrap metal and recycling firm in the United States. At the same time, SHV began building a dry bulk transport and logistics wing, holding interests in such companies as Europees Massagoed Overslagbedrijf, EKOM, and Goudriaan & Co. Re-focusing for the New Century By the end of the 1980s, SHV had found a new interest. In 1982, the company acquired a major share in France\'s Primagaz, which had been founded as Société Liotard Frères in 1857. In 1928, that company had the idea of bottling butane--which was a byproduct of the petroleum distillation process--and selling it to households and industries. By 1934, the company had adopted the brand name Primagaz, which became the company\'s name itself in 1938. Primagaz had added propane in 1951, and then began an international expansion, spreading throughout most of Europe. The company also developed Liquid Petroleum Gas as a automotive fuel by the end of the 1970s, a market which began in earnest in the mid-1980s. SHV continued to build up its stake in Primagaz. At the same time, it sought other acquisition targets. A major addition came in the mid-1980s when SHV began building a stake in the United Kingdom\'s Calor Gas. Formed in 1935, that company spread throughout the United Kingdom by the outbreak World War II. Calor grew strongly in the years following the war, becoming the country\'s leading bottled gas provider; it met with more limited success in its own vehicle LPG operations. SHV initially acquired a nearly 30 percent share in Calor Gas, becoming that company\'s largest shareholder. In 1987, the company attempted to force a merger between Calor Gas and Burmah Oil, a British oil company. That bid was rejected by Calor Gas, however. Instead, SHV continued to build up its position in Calor Gas, reaching 40 percent by 1988. SHV unsuccessfully tried to marry Burmah and Calor again at the end of the decade and finally gave up on this scheme in 1991. Instead, SHV used its positions within Primagaz and Calor to take the two companies on an expansion drive into Eastern Europe, setting up partnerships in Poland and Slovakia in 1991 and in Hungary in 1992. SHV also entered the South American market, buying up Brazil\'s Supergasbras in 1995. Meanwhile, SHV continued to build its interest in Calor, gaining majority control; in 1996, SHV launched an offer to take over full control of Calor Gas. That deal was completed in 1997. Two years later, SHV acquired full control of Primagaz as well. Makro too had continued to expand, opening its first branches in Southeast Asia in Thailand in 1989, followed by Indonesia in 1992, Malaysia in 1993, and China and the Philippines in 1996. Makro had also significantly expanded its presence in the Brazilian market, with 20 stores by 1990 and 40 stores by the beginning of the next decade. Yet, faced with bruising competition in Europe, and hampered in its ability to compete by its private status, SHV decided to exit the cash-and-carry market in 1997, selling off its entire Makro Europe operation--by then posting sales of some $8 billion per year--to Germany\'s Metro. SHV kept its South American and Asian Makro operations. At the same time, the company, which had in the meantime built up a variety of other retailing interests, ranging from office supplies in Beligum to wholesale goods in South Africa, began divesting these operations as well. Flush with cash from the Makro sale, SHV began eyeing new acquisition candidates. In 1998, the company purchased a 60 percent stake in the scrap metal operations of Germany\'s Thyssen group, which were renamed TSR. The following year, SHV took full control of TSR. In 2000, SHV turned to another direction, buying up NPM Venture Capital Group, paying nearly EUR 2 billion for one of the largest venture capital companies in the Netherlands, a move made in part to allow SHV to enter the Internet and e-commerce markets. By 2001, SHV had rebuilt its annual revenues past EUR 10 billion. In that year, the company, which still maintained interests in coal trading, sold off the last of those operations. By the end of 2002, however, the company\'s exposure to fluctuations in South American currencies pushed its annual sales back to EUR 9.4 million. Nonetheless, SHV remained one of the Netherlands\' largest--and most flexible--holding companies. Principal Subsidiaries: Bizimgaz Ticaret Ve Sanayi A.S. (Turkey); Butan Plin d.d. (Slovenia); Calor Gas Ltd. (United Kingdom); Calor Gas Northern Ireland Ltd (Northern Ireland); Calor Teoranta (United Kingdom); Compagnie des Gaz de Pétrole Primagaz S.A.; CTA Makro Commercial Co., Ltd. (China); Dyas B.V.; Gaspol S.A. (Poland); Guangzhou Chia Tai Makro (JiaJing) Co. Ltd. (China); HKS Metals B.V. (Netherlands); Ipragaz A.S. (Turkey); Joseph Transportation Inc. (United States); Liquigas S.p.A. (Italy); Liquigaz Philippines Corporation (Philippines); Makro Asia Thailand ; Makro Atacadista S.A. (Brazil); Makro Cash & Carry Distribution (M) Sdn. Bhd. (Malaysia); Makro Comercializadora S.A. (Venezuela); Makro de Colombia S.A. (Colombia); Makro Office Centre Limited (Thailand); Makro Taiwan Ltd.; Minasgás S.A. Distribuidora de Gás Combustível (Brazil); Orkam Asia Holding AG (Switzerland); Orkam South America Holding AG (Switzerland); Pilipinas Makro, Inc. (Philippines); Primagas GmbH (Germany); Primagaz Belgium N.V.; Primagaz Central Europe GmbH (Austria); Primagaz Danmark A/S; Primagaz Distribución S.A. (Spain); Primagaz GmbH (Austria); Prímagáz Hungária Rt. (Hungary); Primagaz Nederland B.V.; Primaplyn Spol. s.r.o. (Czech Republic); Probugas a.s. (Slovakia); PT Makro Indonesia (Indonesia); SHV Energy India Ltd (India); SHV Energy Pakistan (Pvt) Ltd. (Pakistan); SHV Gas China (P.R. China); Siam Makro Public Company Limited (Thailand); Supergasbras Distribuidora de Gás S.A. (Brazil); Supermercados Mayoristas Makro S.A. (Argentina); The David J. Joseph Company (United States); Tissir Primagaz (Morocco); TSR Recycling GmbH & Co KG (Germany); World Gas (Thailand) Company Limited (Thailand). Principal Competitors: Carrefour SA; Wal-Mart Stores, Inc.; TOTAL S.A.; CFF Recycling; Royal Dutch/Shell Group of Companies. You can help our automatic cover photo selection by reporting an unsuitable photo. Your input will affect cover photo selection, along with input from other users. Would you like to suggest this photo as the cover photo for this article?'



### BERT based Text Summarization


```python
model = Summarizer()
```


```python
result = model(corpus, min_length=10, max_length=100,algorithm='gmm',ratio=0.5)
full = ''.join(result)
print(full)
testimonial = TextBlob(full)
print('\n Polarity of Article:', testimonial.sentiment.polarity)
```

    SHV is a decentralised company. This decentralisation provides an excellent opportunity for individual development. SHV's most important values are integrity and loyalty. Good news may travel slowly, bad news should travel quickly. By 1911, the younger Fentener van Vlissingen had been named the company's director. SHV's fuel interests soon turned toward a new type, however. This process began in earnest at the end of the 1960s. SHV split up Geveke & Groenpol into two operations. Instead, SHV had begun building a number of other growing operations. SHV continued to build up its stake in Primagaz. At the same time, it sought other acquisition targets. Formed in 1935, that company spread throughout the United Kingdom by the outbreak World War II. That bid was rejected by Calor Gas, however. Two years later, SHV acquired full control of Primagaz as well. Nonetheless, SHV remained one of the Netherlands' largest--and most flexible--holding companies. Principal Subsidiaries: Bizimgaz Ticaret Ve Sanayi A.S. (Turkey); Butan Plin d.d. ( Hungary); Primagaz Nederland B.V.; Primaplyn Spol. Your input will affect cover photo selection, along with input from other users. Would you like to suggest this photo as the cover photo for this article?
    
     Polarity of Article: 0.1777935606060606


### Similarity Score Calculation using spaCy


```python
extraction = spacy_nlp(full)
similarity_score = extraction.similarity(kyc_doc)
print('The Similarity Score of Summarized Text is: ', similarity_score*100)
```

    The Similarity Score of Summarized Text is:  91.93679763169389



```python

```
