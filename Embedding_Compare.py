import networkx as nx
import matplotlib.pyplot as plt
import json
import numpy as np
from karateclub import GLEE
from karateclub import NetMF
from karateclub import RandNE
from karateclub import SocioDim
from karateclub import Node2Vec
from karateclub import deepwalk


with open('all.json', encoding='utf-8-sig') as f:
  data = json.load(f)

G = nx.Graph()
for i in range(0,len(data)):
    info = data[i]['p']

    start_node = info['start']
    end_node = info['end']

    G.add_node(start_node['identity'], labels=start_node['labels'], properties=start_node['properties'])
    G.add_node(end_node['identity'], labels=end_node['labels'], properties=end_node['properties'])

    relation = info['segments'][0]['relationship']
    G.add_edge (relation['start'], relation['end'], type = relation['type'],properties = relation['properties'])
    # G.add_edge (relation['start'], relation['end'], type = relation['type'], properties = relation['properties']， relation['identity'])

mapping = {node: i for i, node in enumerate( G. nodes())}
G = nx. relabel_nodes( G, mapping)

print(G.nodes)
print(G.nodes(data=True))
print(G.edges(data=True))
print( len(G.nodes()) )
print(G.nodes[2767])


y = []
for node in G.nodes:
    if G.nodes[node]['labels'] == ['FacePhenotype']: 
        y.append('FacePhenotype')
    elif G.nodes[node]['labels'] == ['Sample']:
        y.append('Sample')
    elif G.nodes[node]['labels'] == ['Disease']: 
        y.append('Disease')
    elif G.nodes[node]['labels'] == ['Article']: 
        y.append('Article')
    elif G.nodes[node]['labels'] == ['Variation']: 
        y.append('Variation')
    elif G.nodes[node]['labels'] == ['Genotype']: 
        y.append('Genotype')

from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  f1_score
from sklearn.model_selection import train_test_split


def score_metric(embeddings):
    X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train) 
    y_pred = rf.predict(X_test)

    f1 = f1_score(y_test, y_pred, average='weighted')
    return f1




def embedding_estimator(embeddings):
    X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.2, random_state=42)
    print(f"train set:{len(X_train)}，test set:{len(X_test)}")


    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train) 
    y_pred = rf.predict(X_test)

    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    print("weighted_f1 Score:", weighted_f1)
    print("micro_f1 Score:", micro_f1)
    print("macro_f1 Score:", macro_f1)


model_deepwalk = deepwalk.DeepWalk(walk_length=70, walk_number=5)
model_deepwalk.fit(G)
X_deepwalk = model_deepwalk.get_embedding()
embedding_estimator(X_deepwalk)

model_node2vec = Node2Vec()
model_node2vec.fit(G)
X_node2vec = model_node2vec.get_embedding()
embedding_estimator(X_node2vec)

model_socioDim = SocioDim()  
model_socioDim.fit(G)
X_socioDim = model_socioDim.get_embedding()
embedding_estimator(X_socioDim)

model_randne = RandNE() 
model_randne.fit(G)
X_randne = model_randne.get_embedding()
embedding_estimator(X_randne)

model_netmf = NetMF()
model_netmf.fit(G)
X_netmf = model_netmf.get_embedding()
embedding_estimator(X_netmf)

model_glee = GLEE()
model_glee.fit(G)
X_glee = model_glee.get_embedding()
embedding_estimator(X_glee)

