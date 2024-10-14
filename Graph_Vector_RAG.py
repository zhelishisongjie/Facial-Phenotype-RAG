from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import time
import functools
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
import torch
import torch.nn as nn
import torch.optim as optim
from torchcrf import CRF
from moverscore.moverscore_v2 import get_idf_dict, word_mover_score 
from transformers import AutoTokenizer, AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained("./results/checkpoint")
tokenizer = AutoTokenizer.from_pretrained("./results/checkpoint")

load_dotenv(os.path.join(os.path.expanduser('~'), '.spoke_neo4j_config.env'))
LLM_MODEL = 'gpt-4-turbo-2024-04-09'
os.environ["OPENAI_API_BASE"] =  "https://api.openai.com/v1/"
os.environ["OPENAI_API_KEY"] = "your api key"
filename = "Cosine-" + LLM_MODEL + "-1"
API_KEY = os.environ.get('API_KEY')
API_VERSION = os.environ.get('API_VERSION')
API_BASE = os.environ.get('OPENAI_API_BASE')
openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = API_BASE
openai.api_version = API_VERSION

temperature = 0
chat_model = ChatOpenAI(model_name = LLM_MODEL, temperature = temperature)

questions = pd.read_csv("./questions/patient_question_withoutchoices.csv")
questions = questions['Patient_question']

# Construct Graph
with open('./KG/FPKG.json', encoding='utf-8-sig') as f:
  data = json.load(f)

context_list = []
for i in tqdm(range(len(questions))):
    question = questions[i]
    G = nx.Graph()
    for i in range(0, len(data)):
        info = data[i]['p']
        start_node = info['start']
        end_node = info['end']
        G.add_node(start_node['identity'], labels=start_node['labels'], properties=start_node['properties'])
        G.add_node(end_node['identity'], labels=end_node['labels'], properties=end_node['properties'])
        relation = info['segments'][0]['relationship']
        G.add_edge(relation['start'], relation['end'], type=relation['type'], properties=relation['properties'], key=relation['identity'])

    def graph_sort(G):
        index_map = {}
        index = 0
        for node in G.nodes():
            index_map[str(node)] = index
            index += 1
        mapping = {node: i for i, node in enumerate( G. nodes())}
        G = nx. relabel_nodes( G, mapping)
        return G , index_map
    G, index_map = graph_sort(G)



    label_map = {'B-AGE': 0,'B-DISEASE': 1,'B-FACIAL_PHENOTYPE': 2,'B-GENDER': 3,'B-GENE': 4,'B-MUTATION': 5,'B-RACE': 6,'I-DISEASE': 7,'I-FACIAL_PHENOTYPE': 8,'I-GENE': 9,'I-MUTATION': 10,'I-RACE': 11,'O': 12}
    def extract_entities(tokenized_input, predicted_labels, tokenizer, reverse_label_map):
        entities_by_type = {}
        current_entity = []
        current_type = None
        for token, label_idx in zip(tokenized_input["input_ids"][0], predicted_labels):
            label = reverse_label_map[label_idx]
            if label.startswith('B-') or (current_entity and not label.startswith('I-')):
                if current_entity:
                    decoded_entity = tokenizer.decode(torch.tensor(current_entity, dtype=torch.int32))
                    if current_type not in entities_by_type:
                        entities_by_type[current_type] = []
                    entities_by_type[current_type].append(decoded_entity)
                    current_entity = []
                    current_type = None
                if label.startswith('B-'):
                    current_entity = [token]
                    current_type = label[2:]
            elif label.startswith('I-') and current_type == label[2:]:
                current_entity.append(token)
        if current_entity:
            decoded_entity = tokenizer.decode(torch.tensor(current_entity, dtype=torch.int32))
            if current_type not in entities_by_type:
                entities_by_type[current_type] = []
            entities_by_type[current_type].append(decoded_entity)
        return entities_by_type

    def merge_subwords(word_list):
        merged_list = []
        for word in word_list:
            if word.startswith('##'):
                if merged_list:
                    merged_list[-1] += word[2:]
                else:
                    merged_list.append(word[2:])
            else:
                merged_list.append(word)
        return merged_list


    reverse_label_map = {v: k for k, v in label_map.items()}
    tokenized_input = tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(model.device)
    predictions = model(tokenized_input["input_ids"])
    predicted_labels = torch.argmax(predictions.logits, dim=-1).squeeze().tolist()
    entities = extract_entities(tokenized_input, predicted_labels, tokenizer, reverse_label_map)
    for entity_type, entity_list in entities.items():
        entities[entity_type] = merge_subwords(entities[entity_type])

    entities['AGE'] = float(''.join(entities['AGE']))
    age = entities.get('AGE')
    gender = entities.get('GENDER')
    race = entities.get('RACE')
    mutation_gene = entities.get('GENE')
    mutation_detail = entities.get('MUTATION')
    facial_phenotypes = entities.get('FACIAL_PHENOTYPE')

    new_node_id = len(G.nodes)
    G.add_node( len(G.nodes), labels=['Sample'], properties={
            "race": race,
            "gender": "F" if gender == "female" else "M" if gender == "male" else gender,
            "year": age,
            "`sample number`": 1,})

    i = 0
    for n in G.nodes():  
        if G.nodes[n]['labels'][0] == 'Variation' and G.nodes[n]['properties']['details'] == mutation_detail:
            i+=1
            G.add_edge( new_node_id, n , type = "Mention_Var" , properties = {} , key = 30000 + i)
        if G.nodes[n]['labels'][0] == 'FacePhenotype' and facial_phenotypes is not None and G.nodes[n]['properties']['phenotypeName'] in facial_phenotypes: 
            i+=1
            G.add_edge( n, new_node_id , type = "Mention_FP" , properties = {} , key = 30000 + i)


    model_GLEE = GLEE(seed=100)
    model_GLEE.fit(G)
    embeddings = model_GLEE.get_embedding()
    new_node_vector = embeddings[new_node_id]
    embeddings = embeddings[:-1, :]
    similarities = cosine_similarity(new_node_vector.reshape(1,-1), embeddings)
    most_similar_idx = similarities[0].argsort()[::-1] 
    neo4j_index = [list(index_map.keys())[idx] for idx in most_similar_idx]

    disease_nodes = []
    disease_scores = []
    for idx, n in enumerate(most_similar_idx):
        if G.nodes[n]['labels'][0] == "Disease":
            disease_nodes.append(n)
            disease_scores.append( similarities[0][n] )
    top_disease_nodes = disease_nodes[:10] if len(disease_nodes) >= 10 else disease_nodes
    top_disease_scores = disease_scores[:10] if len(disease_scores) >=10 else disease_scores
    top_similar_nodes = most_similar_idx[0:20]
    context = [f"properties：{G.nodes[top_disease_nodes[i]]['properties']}, scores: {top_disease_scores[i]}" for i in range(3)]
    context_list.append(context)

def retry_decorator(max_retries=5, delay_seconds=1, backoff_factor=2):
    def decorator_retry(func):
        @functools.wraps(func)
        def wrapper_retry(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    time.sleep(delay_seconds * (backoff_factor ** retries))
                    retries += 1
                    if retries == max_retries:
                        raise RuntimeError(f"Failed after {max_retries} attempts") from e
        return wrapper_retry
    return decorator_retry

QA_TEMPLATE = """
You are an assistant that helps to form precise and medically accurate answers.
The information section provides details on the three most likely diseases based on the patient's symptoms and genetic data. 
Select the most likely disease and explain the reasoning behind your choice using medical evidence.
Make the answer sound as a response to the question.
Do not mention that you got this result based on the information provided, but ensure the explanation is medically sound and justifiable.
If the information provided is empty, answer the question normally using medical reasoning based on typical symptoms and known genetic associations.
Information: {context}
Question: {question}
"""

class ComplexAPIHandler:
    def __init__(self, base_url, api_key, model):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    @retry_decorator()
    def fetch_response(self, context, question):
        formatted_input = QA_TEMPLATE.format(context=context, question=question)
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": formatted_input}],
            model=self.model,
        )
        return chat_completion.choices[0].message.content

# Instantiate the API handler
api_handler = ComplexAPIHandler(
    base_url=os.environ.get('OPENAI_API_BASE'),
    api_key=os.environ.get("OPENAI_API_KEY"),
    model=os.getenv("LLM_MODEL", "gpt-4-turbo")
)

response_rag_list = []
for i in tqdm(range(len(response_rag_list), len(questions))):
    delay = 12 if api_handler.model.startswith('claude') else 1
    time.sleep(delay)

    context = context_list[i]
    question = questions[i]
    
    response = api_handler.fetch_response(context, question)
    response_rag_list.append(response)


questions = pd.read_csv("./questions/patient_question_withoutchoices.csv")
answer = {'questions':questions['Patient_question'] , 'True_answer':questions['True_answer'], 'Cosine RAG': response_rag_list}
answer = pd.DataFrame(answer)
answer.to_csv("./answers/" + filename + ".csv",index=False)

