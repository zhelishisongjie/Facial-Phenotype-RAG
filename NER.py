import torch
import torch.nn as nn
import torch.optim as optim
from torchcrf import CRF
from datasets import load_dataset, load_metric, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from moverscore.moverscore_v2 import get_idf_dict, word_mover_score
import numpy as np

TRAIN_DATA_PATH = './corpus/train.txt'
VALID_DATA_PATH = './corpus/dev.txt'
TEST_DATA_PATH = './corpus/test.txt'

def read_conll_file(file_path):
    with open(file_path, "r") as f:
        content = f.read().strip()
        sentences = content.split("\n\n")
        data = []
        for sentence in sentences:
            tokens = sentence.split("\n")
            token_data = []
            for token in tokens:
                token_data.append(token.split())
            data.append(token_data)
    return data

train_data = read_conll_file(TRAIN_DATA_PATH)
validation_data = read_conll_file(VALID_DATA_PATH )
test_data = read_conll_file(TEST_DATA_PATH)

print(len(train_data))
print(len(validation_data))
print(len(test_data))

def convert_to_dataset(data, label_map):
    formatted_data = {"tokens": [], "ner_tags": []}
    for sentence in data:
        tokens = [token_data[0] for token_data in sentence]
        ner_tags = [label_map[token_data[3]] for token_data in sentence]
        formatted_data["tokens"].append(tokens)
        formatted_data["ner_tags"].append(ner_tags)
    return Dataset.from_dict(formatted_data)


label_list = sorted(list(set([token_data[3] for sentence in train_data for token_data in sentence])))
label_map = {label: i for i, label in enumerate(label_list)}


train_dataset = convert_to_dataset(train_data, label_map)
validation_dataset = convert_to_dataset(validation_data, label_map)
test_dataset = convert_to_dataset(test_data, label_map)

datasets = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
    "test": test_dataset,
})

model_name = "dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list), ignore_mismatched_sizes=True)

def compute_metrics(eval_prediction):
    predictions, labels = eval_prediction
    predictions = np.argmax(predictions, axis=2)
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "classification_report": classification_report(true_labels, true_predictions),
    }


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, padding=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# train
tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

import os
output_dir = "./results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=500,
    num_train_epochs=100,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_steps=100,
    learning_rate=5e-5,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

def data_collator(data):
    input_ids = [torch.tensor(item["input_ids"]) for item in data]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in data]
    labels = [torch.tensor(item["labels"]) for item in data]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# testset
test_dataset = tokenized_datasets["test"]
evaluation_results = trainer.evaluate(test_dataset)
print(evaluation_results)

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



# test
sentence = "There is a 14.0 years old female Chinese patient with a mutation in KMT2D, namely, c. 16343G > C; p.R5448P and with facial phenotypes of Sparse eyebrow, Wide nasal bridge, Strabismus, Depressed nasal tip. What disease might this patient have?"
reverse_label_map = {v: k for k, v in label_map.items()}

tokenized_input = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(model.device)
predictions = model(tokenized_input["input_ids"])
predicted_labels = torch.argmax(predictions.logits, dim=-1).squeeze().tolist()

entities = extract_entities(tokenized_input, predicted_labels, tokenizer, reverse_label_map)
for entity_type, entity_list in entities.items():
    entities[entity_type] = merge_subwords(entities[entity_type])

entities['AGE'] = float(''.join(entities['AGE']))
entities

