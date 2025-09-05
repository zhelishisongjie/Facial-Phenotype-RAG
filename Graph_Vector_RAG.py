from langchain.chat_models import ChatOpenAI
import os
import openai
import pandas as pd
from tqdm import tqdm
import time
from openai import OpenAI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from karateclub import NodeSketch, GLEE
import networkx as nx
import matplotlib.pyplot as plt
import json
import numpy as np

# question = "There is a 19 years old female German patient with a facial phenotype of Wide nasal bridge, Broad philtrum. What disease might this patient have?"
# FRONTONASAL DYSPLASIA 1(FND1)
questions = pd.read_csv("./questions/GMDB_patient_question.csv")
questions = questions['Patient_question']


# ==================== QA template ====================
Extraction_TEMPLATE="""
You are a professional biomedical information extraction assistant. Please accurately identify and extract the following entity information from the given sentence:  
- age  
- gender  
- race  
- mutation_gene  
- mutation_hgvs (mutation description in HGVS format)  
- facial_phenotypes  
- disease  

Requirements:  
1. Output MUST be a SINGLE VALID JSON object.
2. DO NOT wrap the output in ```json ``` or any other Markdown.
3. If a field is not found, set it to `null`.
4. Never add comments or text outside the JSON.

Input sentence: {question} 

Example output format:  
{{ 
  "age": "extracted age information",  
  "gender": "extracted gender information",  
  "race": "extracted race information",  
  "mutation_gene": "extracted gene name",  
  "mutation_hgvs": "extracted HGVS mutation description",  
  "facial_phenotypes": ["facial feature 1", "facial feature 2"],  
  "disease": ["disease 1", "disease 2"]
}}  

Usage example:  
Input sentence: "There is a 12.0-year-and-0.0-month-old female European patient with a mutation in NFIX, namely, NM_002501.3:c.59T>C, p.(Leu20Pro) and with facial phenotypes of Tall stature, Narrow mouth, Open mouth, Everted lower lip vermilion, Long face, Mandibular prognathia, Triangular face, Anteverted nares, Strabismus, Deeply set eye, Downslanted palpebral fissures, Atypical behavior, Intellectual disability, Hypotonia, Global developmental delay, Motor delay, obsolete Joint laxity, Ventriculomegaly, Chiari malformation, Prominent forehead. What disease might this patient have? Choose from the following options: A.SOTOS SYNDROME; SOTOS    B.LATERAL MENINGOCELE SYNDROME; LMNS    C.Oculocutaneous albinism    D.SPASTIC PARAPLEGIA 50, AUTOSOMAL RECESSIVE; SPG50"  
Output:  
{{ 
  "age": "12.0",  
  "gender": "female",  
  "race": "European",  
  "mutation_gene": "NFIX",  
  "mutation_hgvs": "NM_002501.3:c.59T>C, p.(Leu20Pro)",  
  "facial_phenotypes": ["Tall stature", "Narrow mouth", "Open mouth", "Everted lower lip vermilion", "Long face", "Mandibular prognathia", "Triangular face", "Anteverted nares", "Strabismus", "Deeply set eye", "Downslanted palpebral fissures", "Atypical behavior", "Intellectual disability", "Hypotonia", "Global developmental delay", "Motor delay", "obsolete Joint laxity", "Ventriculomegaly", "Chiari malformation", "Prominent forehead"],  
  "disease": ["SOTOS SYNDROME; SOTOS", "LATERAL MENINGOCELE SYNDROME; LMNS", "Oculocutaneous albinism", "SPASTIC PARAPLEGIA 50", "AUTOSOMAL RECESSIVE; SPG50"]  
}} 
"""

# LLMs
LLM_MODEL = 'gpt-4o'

# api key
os.environ["OPENAI_API_BASE"] =  "https://api.openai.com/v1/"
os.environ["OPENAI_API_KEY"] = "your api key"
API_KEY = os.environ.get('API_KEY')
API_VERSION = os.environ.get('API_VERSION')
API_BASE = os.environ.get('OPENAI_API_BASE')
openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = API_BASE
openai.api_version = API_VERSION


# ==================== LLM ====================
temperature = 0.1
chat_model = ChatOpenAI(model_name = LLM_MODEL, temperature = temperature)

questions = pd.read_csv("./questions/GMDB_patient_question.csv")
questions = questions['Patient_question']

import json
import re

# entities

with open('records_25.05.08.json', encoding='utf-8-sig') as f:
    graph_data = json.load(f)

client = OpenAI(
    base_url=os.environ.get('OPENAI_API_BASE'),
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def filter_fields(node_data):
    target_fields = ['labels', 'phenotypeName',
                     'with_disease', 'sid',
                     'geneName',
                     'details', 'variation_gene_name',
                     'dname']
    filtered = {}
    for field in target_fields:
        if field in node_data:
            filtered[field] = node_data[field]
        elif 'properties' in node_data and field in node_data['properties']:
            filtered[field] = node_data['properties'][field]
    return filtered


from fuzzywuzzy import fuzz


def match(str1, str2):
    str1 = str1.replace(" ", "")
    str2 = str2.replace(" ", "")
    match_percentage = fuzz.ratio(str1, str2)
    if match_percentage > 80:
        return True
    else:
        return False


def trim_strings_to_max_length(string_list, max_length):
    total_length = sum(len(s) for s in string_list)
    if total_length > max_length:
        excess_length = total_length - max_length
        for i in range(len(string_list) - 1, -1, -1):
            current_string_length = len(string_list[i])
            if current_string_length > excess_length:

                string_list[i] = string_list[i][:current_string_length - excess_length]
                break
            else:
                string_list[i] = ""
                excess_length -= current_string_length
    return string_list


context_list = []
for k in tqdm(range(len(questions))):
    question = questions[k]
    print(question)

    # construct graph
    G = nx.Graph()
    for i in range(0, len(graph_data)):
        info = graph_data[i]['p']
        start_node = info['start']
        end_node = info['end']

        G.add_node(start_node['identity'], labels=start_node['labels'], properties=start_node['properties'])
        G.add_node(end_node['identity'], labels=end_node['labels'], properties=end_node['properties'])

        relation = info['segments'][0]['relationship']
        # G.add_edge (relation['start'], relation['end'], type = relation['type'],properties = relation['properties'])
        G.add_edge(relation['start'], relation['end'], type=relation['type'], properties=relation['properties'],
                   key=relation['identity'])


    def graph_sort(G):
        index_map = {}
        index = 0
        for node in G.nodes():
            index_map[str(node)] = index
            index += 1
        # print(index_map)

        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        # print(G.nodes)
        # print(f" lenth: {len(G.nodes())}  last node:{ G.nodes[len(G.nodes())-1] }")
        return G, index_map


    G, index_map = graph_sort(G)

    # entity extraction
    data = {"question": question}
    content = Extraction_TEMPLATE.format(**data)

    # Employing an LLM for entity extraction may lead to better results.
    max_retries = 5
    retries = 0
    while retries < max_retries:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                model=LLM_MODEL,
            )
            response = chat_completion.choices[0].message.content
            entity = json.loads(response)

            break
        except Exception as e:
            retries += 1
            print(f"An error occurred: {e}")
            print(f"retry：{retries}...")

    age = entity.get('age')
    gender = entity.get('gender')
    race = entity.get('race')
    mutation_gene = entity.get('mutation_gene')
    mutation_detail = entity.get('mutation_hgvs')
    facial_phenotypes = entity.get('facial_phenotypes')
    disease = entity.get('disease')

    # Add node to KG
    new_node_id = len(G.nodes)
    G.add_node(len(G.nodes), labels=['New nodes'], properties={})
    connection_rules = [
        ('Variation', 'details', mutation_detail or []),
        ('FacePhenotype', 'phenotypeName', facial_phenotypes or []),
        ('Genotype', 'geneName', mutation_gene or []),
        ('Disease', 'dname', disease or [])
    ]
    for n in G.nodes:
        node_labels = G.nodes[n].get('labels', [])
        if not node_labels:
            continue
        node_props = G.nodes[n].get('properties', {})

        for label, prop_key, match_value in connection_rules:
            if node_labels[0] == label:
                prop_value = node_props.get(prop_key, '')

                if any(match(prop_value, p) for p in match_value):
                    G.add_edge(n, new_node_id, type='Connected')
                    # print(f"Connected {prop_value}  to new node")

    # Embedding
    model_GLEE = GLEE(seed=100)
    model_GLEE.fit(G)
    embeddings = model_GLEE.get_embedding()
    new_node_vector = embeddings[new_node_id]
    embeddings = embeddings[:-1, :]
    similarities = cosine_similarity(new_node_vector.reshape(1, -1), embeddings)
    most_similar_idx = similarities[0].argsort()[::-1]
    neo4j_index = [list(index_map.keys())[idx] for idx in most_similar_idx]

    top_similar_nodes = most_similar_idx[0:5]

    # for topnode in top_similar_nodes:
    #    print(similarities[0][topnode], G.nodes[topnode])

    # subgraph of top similar nodes
    context = []
    for topnode in top_similar_nodes:
        neighbors = list(G.neighbors(topnode))[:-1]
        for neighbor in neighbors:
            edge_data = G.get_edge_data(topnode, neighbor)
            subgraph = str(filter_fields(G.nodes[topnode])) + "   " + edge_data['type'] + "   " + str(
                filter_fields(G.nodes[neighbor]))
            # print(f"***********Retrieved subgraph: {subgraph}")
            context.append(subgraph)

    context = trim_strings_to_max_length(context, 10000)
    print(context)
    context_list.append(context)



# ==================== QA template ====================
QA_TEMPLATE="""
You are a medical genetics assistant specializing in analyzing facial phenotypes to identify rare genetic diseases, and interpret the relationships between genes, facial features, and associated diseases.

Instructions: 
The information section provides some knowledge based on the patient's symptoms and genetic data. You should refer to this knowledge and make the most likely diagnosis.
Make the answer sound as a response to the question.
Do not mention that you got this result based on the information provided, but ensure the explanation is medically sound and justifiable.
If the information provided is empty, answer the question normally using medical reasoning based on typical symptoms and known genetic associations.

Information: {context}
Question: {question}
"""

# LLMs
#LLM_MODEL = 'gpt-3.5-turbo'
#LLM_MODEL = 'gpt-4-turbo-2024-04-09'
LLM_MODEL = 'gpt-4o'
#LLM_MODEL = 'claude-3-opus-20240229'
#LLM_MODEL = 'claude-3-sonnet-20240229'
#LLM_MODEL = 'claude-3-haiku-20240307'
#LLM_MODEL = 'gemini-1.0-pro-latest'



filename = "Vector-" + LLM_MODEL + "-1"

API_KEY = os.environ.get('API_KEY')
API_VERSION = os.environ.get('API_VERSION')
API_BASE = os.environ.get('OPENAI_API_BASE')
openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = API_BASE
openai.api_version = API_VERSION


# ==================== LLM ====================
temperature = 0.1
chat_model = ChatOpenAI(model_name = LLM_MODEL, temperature = temperature)

questions = pd.read_csv("./questions/GMDB_patient_question_withoutchoices.csv")
questions = questions['Patient_question']

# ============================================================ openai ============================================================
client = OpenAI(
    base_url=os.environ.get('OPENAI_API_BASE'),
    api_key=os.environ.get("OPENAI_API_KEY"),
)

max_retries = 5
response_rag_list = []
for i in tqdm(range(len(response_rag_list), len(questions))):
    if LLM_MODEL.startswith('claude'):
        time.sleep(12)
    time.sleep(1)

    question = questions[i]
    context = context_list[i]
    data = {
        "context": context,
        "question": question
    }

    content = QA_TEMPLATE.format(**data)

    retries = 0
    while retries < max_retries:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                model=LLM_MODEL,
            )

            response = chat_completion.choices[0].message.content

            break
        except Exception as e:
            retries += 1
            print(f"An error occurred: {e}")
            print(f"retry：{retries}...")
    # print(response)
    response_rag_list.append(response)
# ============================================================ openai ============================================================