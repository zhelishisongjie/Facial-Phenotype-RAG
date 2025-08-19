from langchain.chains import GraphCypherQAChain
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.graphs import Neo4jGraph
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import os
import openai
import pandas as pd
from neo4j.exceptions import CypherSyntaxError
from langchain.prompts import PromptTemplate
from tqdm import tqdm
import time
from moverscore.moverscore_v2 import get_idf_dict, word_mover_score
from openai import OpenAI

load_dotenv(os.path.join(os.path.expanduser('~'), '.spoke_neo4j_config.env'))
LLM_MODEL = 'gpt-4-turbo-2024-04-09'
os.environ["OPENAI_API_BASE"] =  "https://api.openai.com/v1/"
os.environ["OPENAI_API_KEY"] = "your api key"
username = "your username"
password = "your password"
url = "bolt://localhost:7687"
database = "neo4j"
filename = LLM_MODEL + "-1"
API_KEY = os.environ.get('API_KEY')
API_VERSION = os.environ.get('API_VERSION')
API_BASE = os.environ.get('OPENAI_API_BASE')
openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = API_BASE
openai.api_version = API_VERSION

from langchain.graphs import Neo4jGraph
graph = Neo4jGraph(
    url=url, username=username, password=password,database = database
)
temperature = 0.1
chat_model = ChatOpenAI(model_name = LLM_MODEL,temperature = temperature)
questions = pd.read_csv("./Datasets/Publication_patient_question_withoutchoices.csv")
questions = questions['Patient_question']

import logging
response_vanilla_list = []
'''
# langchain
prompt = PromptTemplate(
    #template=
    #"""
    #You are a surfer dude, having a conversation about the surf conditions on the beach.
    #Question: {question}
    #""",
    template="{question}",
    input_variables=["question"],
)
chain_vanilla = LLMChain(llm=chat_model, prompt=prompt)


max_retries = 10
for i in tqdm(range(len(questions))):  
    time.sleep(1)  
    question = questions[i]  
    retries = 0  
    while retries < max_retries:  
        try:  
            response_vanilla = chain_vanilla.invoke({"question": question})  
            response_vanilla_list.append(response_vanilla['text'])
            #print(question)
            #print(response_vanilla['text']) 
            break
        except Exception as e:  
            retries += 1  
            print(f"retry：{retries}...") 

'''


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
api_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get('OPENAI_API_BASE')

client = OpenAI(base_url=base_url, api_key=api_key)
def fetch_response(question, max_retries=10, sleep_duration=1):
    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": question}
                ],
                model=LLM_MODEL,
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error during API call: {str(e)}")
            retries += 1
            time.sleep(sleep_duration)
            logging.info(f"Retry {retries}/{max_retries}...")
            if retries == max_retries:
                raise Exception("Max retries reached, failing...")
    return None


response_vanilla_list = []
for i in tqdm(range(len(questions))):
    question = questions[i]
    if LLM_MODEL.startswith('claude'):
        time.sleep(11)
    response = fetch_response(question)
    response_vanilla_list.append(response)

cypher_list = []
CYPHER_GENERATION_TEMPLATE = """
You are a Neo4j Cypher query expert. Please generate Cypher statements to query a graph database based on user questions. 

Instructions:
Your generated Cypher query must follow Neo4j Graph database Schema provided. 
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

Schema:{schema}
The question is: {question}
"""

def remove_triple_quotes(s):  
    if s.startswith("'''") and s.endswith("'''"):  
        return s[3:-3] 
    if s.startswith("'''cypher") and s.endswith("'''"):
        return s[9:-3]
    if s.startswith("```cypher") and s.endswith("```"):
        return s[9:-3]
    return s


"""
cypher_generation_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["schema","question"],
    graph = graph,
)
chain_cypher = LLMChain(llm=chat_model, prompt=cypher_generation_prompt)



for i in tqdm(range(len(questions))):
    time.sleep(1)
    question = questions[i]
    #print(question)

    response_cypher = chain_cypher.invoke({"schema":graph.schema, "question": question})
    #print(response_cypher['text'])
    response = remove_triple_quotes(response_cypher['text'])
    cypher_list.append(response)
"""


for i in tqdm(range(len(questions))):
    if LLM_MODEL.startswith('claude'):
        time.sleep(12)
    time.sleep(1)
    question = questions[i]
    data = {
        "schema": graph.schema,
        "question": question
    }

    content = CYPHER_GENERATION_TEMPLATE.format(**data)

    chat_completion = client.chat.completions.create(
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ],
        model = LLM_MODEL,
    )
    response = remove_triple_quotes(chat_completion.choices[0].message.content)
    cypher_list.append(response)

error_count = 0
context_list = []
for query in cypher_list:  
    try:
        results = graph.query(query)  
        context_list.append(results)
    except ValueError:
        error_count += 1
        context_list.append([])


def remove_dup(context_list):
    unique_context_list = []
    for context in context_list:
        unique_context = []  
        for d in context:  
            if not any(d == item for item in unique_context):  
                unique_context.append(d)
        unique_context_list.append(unique_context)
    return unique_context_list
context_list = remove_dup(context_list)

response_rag_list = []
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


"""
QA_prompt = PromptTemplate(
    template=QA_TEMPLATE,
    input_variables=["context","question"],
)
chain_RAG = LLMChain(llm=chat_model, prompt=QA_prompt)
for i in tqdm(range(len(questions))):
    time.sleep(1)
    question = questions[i]
    context = context_list[i]
    response_rag = chain_RAG.invoke({"context":context, "question": question})

    #print(f"question:{question}")
    #print(f"context: {context}")
    #print(f"response:{response_rag['text']}")
    response_rag_list.append(response_rag['text'])
"""


for i in tqdm(range(len(questions))):
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
    chat_completion = client.chat.completions.create(
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ],
        model = LLM_MODEL,
    )
    response_rag_list.append(chat_completion.choices[0].message.content)

questions = pd.read_csv("./Datasets/Publication_patient_question_withoutchoices.csv")
answer = {'RAG': response_rag_list, 'GPT': response_vanilla_list, 'questions':questions['Patient_question'] , 'True_answer':questions['True_answer']}
answer = pd.DataFrame(answer)
answer.to_csv("./answers/" + filename + ".csv",index=False)

