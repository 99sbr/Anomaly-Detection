## Load all libraries


```python
from nltk.corpus import stopwords
from summarizer import Summarizer
import tensorflow_hub as hub
import tensorflow as tf
import pandas as pd
import umap
import nltk
import random
from bs4 import BeautifulSoup
from ordered_set import OrderedSet
from collections import defaultdict
from itertools import combinations
import usaddress
import requests
import html2text
import spacy
from spacy_stanza import StanzaLanguage
import stanza
import re

from streetaddress import StreetAddressFormatter, StreetAddressParser
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import cufflinks
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import time
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
snlp = stanza.Pipeline(lang="en")
stanza_nlp = StanzaLanguage(snlp)
spacy_nlp = spacy.load('en_core_web_lg')
bert_sent_model = SentenceTransformer('roberta-base-nli-mean-tokens')
stopwords = set(stopwords.words('english'))
model = Summarizer()
# Standard plotly imports
# Using plotly + cufflinks in offline mode
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)
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
    /Users/subir/pythonenv/default/lib/python3.7/site-packages/statsmodels/tools/_testing.py:19: FutureWarning:
    
    pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
    
    2020-05-04 11:04:33 INFO: Loading these models for language: en (English):
    =========================
    | Processor | Package   |
    -------------------------
    | tokenize  | ewt       |
    | pos       | ewt       |
    | lemma     | ewt       |
    | depparse  | ewt       |
    | ner       | ontonotes |
    =========================
    
    2020-05-04 11:04:33 INFO: Use device: cpu
    2020-05-04 11:04:33 INFO: Loading: tokenize
    2020-05-04 11:04:33 INFO: Loading: pos
    2020-05-04 11:04:35 INFO: Loading: lemma
    2020-05-04 11:04:35 INFO: Loading: depparse
    2020-05-04 11:04:37 INFO: Loading: ner
    2020-05-04 11:04:38 INFO: Done loading processors!



<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-latest.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>




<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-latest.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>




```python
Counterparty = 'ficofi holdings pvt ltd.'
RegisteredAddress = "25 INTERNATIONAL BUSINESS PARK #03-01/02 GERMAN CENTRE SINGAPORE (609916)"
```

## URL list


```python
base_url_list = [
    "https://recordowl.com/company/ficofi-partners-holding-pte-ltd",
    "https://www.emis.com/php/company-profile/SG/Ficofi_Partners_Holding_Pte_Ltd_en_6690179.html"
    "https://www.timesbusinessdirectory.com/companies/ficofi-partners-holding-pte-ltd",
    "https://opengovsg.com/corporate/201309826H"
]
```

## PreProcesing URL and Raw Text


```python
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
```

## Jaccard Similarity technique


```python
corpus_token_list = []
for url in base_url_list:
    clean_text_list = parse_article(requests.get(url).text)
    if bool(clean_text_list):
        clean_corpus = text_cleaning(' '.join(clean_text_list))
        corpus_token_list.append(clean_corpus)
```


    ---------------------------------------------------------------------------

    TimeoutError                              Traceback (most recent call last)

    ~/pythonenv/default/lib/python3.7/site-packages/urllib3/connection.py in _new_conn(self)
        158             conn = connection.create_connection(
    --> 159                 (self._dns_host, self.port), self.timeout, **extra_kw)
        160 


    ~/pythonenv/default/lib/python3.7/site-packages/urllib3/util/connection.py in create_connection(address, timeout, source_address, socket_options)
         79     if err is not None:
    ---> 80         raise err
         81 


    ~/pythonenv/default/lib/python3.7/site-packages/urllib3/util/connection.py in create_connection(address, timeout, source_address, socket_options)
         69                 sock.bind(source_address)
    ---> 70             sock.connect(sa)
         71             return sock


    TimeoutError: [Errno 60] Operation timed out

    
    During handling of the above exception, another exception occurred:


    NewConnectionError                        Traceback (most recent call last)

    ~/pythonenv/default/lib/python3.7/site-packages/urllib3/connectionpool.py in urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, **response_kw)
        599                                                   body=body, headers=headers,
    --> 600                                                   chunked=chunked)
        601 


    ~/pythonenv/default/lib/python3.7/site-packages/urllib3/connectionpool.py in _make_request(self, conn, method, url, timeout, chunked, **httplib_request_kw)
        342         try:
    --> 343             self._validate_conn(conn)
        344         except (SocketTimeout, BaseSSLError) as e:


    ~/pythonenv/default/lib/python3.7/site-packages/urllib3/connectionpool.py in _validate_conn(self, conn)
        838         if not getattr(conn, 'sock', None):  # AppEngine might not have  `.sock`
    --> 839             conn.connect()
        840 


    ~/pythonenv/default/lib/python3.7/site-packages/urllib3/connection.py in connect(self)
        300         # Add certificate verification
    --> 301         conn = self._new_conn()
        302         hostname = self.host


    ~/pythonenv/default/lib/python3.7/site-packages/urllib3/connection.py in _new_conn(self)
        167             raise NewConnectionError(
    --> 168                 self, "Failed to establish a new connection: %s" % e)
        169 


    NewConnectionError: <urllib3.connection.VerifiedHTTPSConnection object at 0x20b6ef710>: Failed to establish a new connection: [Errno 60] Operation timed out

    
    During handling of the above exception, another exception occurred:


    MaxRetryError                             Traceback (most recent call last)

    ~/pythonenv/default/lib/python3.7/site-packages/requests/adapters.py in send(self, request, stream, timeout, verify, cert, proxies)
        448                     retries=self.max_retries,
    --> 449                     timeout=timeout
        450                 )


    ~/pythonenv/default/lib/python3.7/site-packages/urllib3/connectionpool.py in urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, **response_kw)
        637             retries = retries.increment(method, url, error=e, _pool=self,
    --> 638                                         _stacktrace=sys.exc_info()[2])
        639             retries.sleep()


    ~/pythonenv/default/lib/python3.7/site-packages/urllib3/util/retry.py in increment(self, method, url, response, error, _pool, _stacktrace)
        398         if new_retry.is_exhausted():
    --> 399             raise MaxRetryError(_pool, url, error or ResponseError(cause))
        400 


    MaxRetryError: HTTPSConnectionPool(host='recordowl.com', port=443): Max retries exceeded with url: /company/ficofi-partners-holding-pte-ltd (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x20b6ef710>: Failed to establish a new connection: [Errno 60] Operation timed out'))

    
    During handling of the above exception, another exception occurred:


    ConnectionError                           Traceback (most recent call last)

    <ipython-input-34-4c7b480c2566> in <module>
          1 corpus_token_list = []
          2 for url in base_url_list:
    ----> 3     clean_text_list = parse_article(requests.get(url).text)
          4     if bool(clean_text_list):
          5         clean_corpus = text_cleaning(' '.join(clean_text_list))


    ~/pythonenv/default/lib/python3.7/site-packages/requests/api.py in get(url, params, **kwargs)
         74 
         75     kwargs.setdefault('allow_redirects', True)
    ---> 76     return request('get', url, params=params, **kwargs)
         77 
         78 


    ~/pythonenv/default/lib/python3.7/site-packages/requests/api.py in request(method, url, **kwargs)
         59     # cases, and look like a memory leak in others.
         60     with sessions.Session() as session:
    ---> 61         return session.request(method=method, url=url, **kwargs)
         62 
         63 


    ~/pythonenv/default/lib/python3.7/site-packages/requests/sessions.py in request(self, method, url, params, data, headers, cookies, files, auth, timeout, allow_redirects, proxies, hooks, stream, verify, cert, json)
        528         }
        529         send_kwargs.update(settings)
    --> 530         resp = self.send(prep, **send_kwargs)
        531 
        532         return resp


    ~/pythonenv/default/lib/python3.7/site-packages/requests/sessions.py in send(self, request, **kwargs)
        641 
        642         # Send the request
    --> 643         r = adapter.send(request, **kwargs)
        644 
        645         # Total elapsed time of the request (approximately)


    ~/pythonenv/default/lib/python3.7/site-packages/requests/adapters.py in send(self, request, stream, timeout, verify, cert, proxies)
        514                 raise SSLError(e, request=request)
        515 
    --> 516             raise ConnectionError(e, request=request)
        517 
        518         except ClosedPoolError as e:


    ConnectionError: HTTPSConnectionPool(host='recordowl.com', port=443): Max retries exceeded with url: /company/ficofi-partners-holding-pte-ltd (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x20b6ef710>: Failed to establish a new connection: [Errno 60] Operation timed out'))



```python
def get_jaccard_sim_multi(str1, str2, str3, str4):
    a = OrderedSet(str1.lower().split())
    b = OrderedSet(str2.lower().split())
    c = OrderedSet(str3.lower().split())
    d = OrderedSet(str4.lower().split())

    Combinations = list(combinations([a, b, c, d], 2))
    e = []
    for x in Combinations:
        e.append(
            OrderedSet(x[0].intersection(x[1])) - set(Counterparty.split()))
    return e
```


```python
match_text = get_jaccard_sim_multi(' '.join(corpus_token_list[0]), ' '.join(corpus_token_list[1]), ' '.join(corpus_token_list[2]), ' '.join(corpus_token_list[3]))
```


```python
from postal.parser import parse_address
```


```python
Address = defaultdict(OrderedSet)
for match in match_text:
    _addrs = OrderedSet(
        parse_address(address=' '.join(list(match))))
    if bool(_addrs):
        for item in _addrs:
            for sub_item in item[0].split():
                Address[item[1]].add(sub_item)
```


```python
Address
```

## USE Experiment (Not in Use)


```python
def get_embeddings(text_list):
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        textual_embeddings = session.run(embed(text_list))
    return textual_embeddings
    
def get_bert_embeddings(text_list):
    return bert_sent_model.encode(text_list,convert_to_numpy=True, show_progress_bar=True)
```


```python
def plot_sentence_embeddings(sentence_embeddings,token_list):
    embedding = umap.UMAP(metric="correlation", n_components=2, random_state=42).fit_transform(sentence_embeddings)
    df_se_emb = pd.DataFrame(embedding, columns=["x", "y"])
    df_se_emb['tokens']=token_list
    df_emb_sample = df_se_emb
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.scatter(
        df_emb_sample["x"].values, df_emb_sample["y"].values, s=50
    )
    plt.setp(ax, xticks=[], yticks=[])
    plt.title("Sentence embeddings embedded into two dimensions by UMAP", fontsize=18)
    plt.show()
    return df_emb_sample
```


```python
%%time
sentence_embeddings = get_bert_embeddings(corpus_token_list[0])
df_emb_sample_0 = plot_sentence_embeddings(sentence_embeddings,corpus_token_list[0])
```


```python
%%time
sentence_embeddings = get_bert_embeddings(corpus_token_list[1])
df_emb_sample_1 = plot_sentence_embeddings(sentence_embeddings,corpus_token_list[1])
```


```python
%%time
sentence_embeddings = get_bert_embeddings(corpus_token_list[2])
df_emb_sample_2 = plot_sentence_embeddings(sentence_embeddings, corpus_token_list[2])
```


```python
%%time
sentence_embeddings = get_bert_embeddings(corpus_token_list[3])
df_emb_sample_3 = plot_sentence_embeddings(sentence_embeddings,corpus_token_list[3])
```


```python
df_emb_sample_0['tag']='url0'
df_emb_sample_0.head()
```


```python

df_emb_sample_1['tag']='url1'
df_emb_sample_1.head()
```


```python

df_emb_sample_2['tag']='url2'
df_emb_sample_2.head()
```


```python

df_emb_sample_3['tag']='url3'
df_emb_sample_3.head()
```


```python
df_emb_sample = df_emb_sample_0.append([df_emb_sample_1, df_emb_sample_2, df_emb_sample_3])
```


```python
df_emb_sample.dropna(inplace=True,subset=['tokens'])
```


```python
df_emb_sample.shape
```


```python
import plotly_express as px
px.colors.qualitative.D3
```


```python
import plotly.express as px
df_emb_sample['size']=1
fig = px.scatter(df_emb_sample, x="x", y="y",size='size',color='tag', hover_name=df_emb_sample['tokens'])
fig.show()
```


```python
embedding_0 = get_bert_embeddings(corpus_token_list[0])
embedding_1 = get_bert_embeddings(corpus_token_list[1])
embedding_2 = get_bert_embeddings(corpus_token_list[2])
embedding_3 = get_bert_embeddings(corpus_token_list[3])
```


```python
Sum_of_squared_distances = []
K = range(1,20)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(df_emb_sample[['x','y']])
    Sum_of_squared_distances.append(km.inertia_)
    
    
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
```


```python
from sklearn.cluster import KMeans

num_clusters = 5
```


```python
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(df_emb_sample[['x','y']])
cluster_assignment = clustering_model.labels_
df_embd_cluster = pd.DataFrame({'token':df_emb_sample['tokens'], 'cluster':cluster_assignment,'url':df_emb_sample['tag']})
```


```python
# clustering_model = KMeans(n_clusters=4)
# clustering_model.fit(embedding_0)
# cluster_assignment = clustering_model.labels_
# df_embd_0_cluster = pd.DataFrame({'token':corpus_token_list[0], 'cluster':cluster_assignment})
```


```python
df_embd_cluster[df_embd_cluster.cluster==0].url.value_counts()
```


```python
df_embd_cluster[df_embd_cluster.cluster==1].url.value_counts()
```


```python
df_embd_cluster[df_embd_cluster.cluster==2].url.value_counts()
```


```python
df_embd_cluster[df_embd_cluster.cluster==3].url.value_counts()
```


```python
df_embd_cluster[df_embd_cluster.cluster==4].url.value_counts()
```


```python
df_embd_cluster[df_embd_cluster.cluster==4].values
```


```python

```


```python

```
