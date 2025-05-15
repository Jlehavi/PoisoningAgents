import sys, os
from transformers import AutoTokenizer
import torch.nn as nn
from transformers import (BertModel, 
                          BertTokenizer, 
                          AutoModelForCausalLM, 
                          LlamaForCausalLM, 
                          DPRContextEncoder,
                          AutoModel,
                          DPRQuestionEncoder,
                          RealmEmbedder,
                          RealmForOpenQA)
import torch
import json, pickle, jsonlines
from pathlib import Path
from tqdm import tqdm
import re
from torch.utils.data import Dataset, DataLoader
import requests
import time
import faiss

from algo.config import model_code_to_embedder_name

api_key = ""

class TripletNetwork(nn.Module):
    def __init__(self):
        super(TripletNetwork, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Additional layers can be added here

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return pooled_output

class ClassificationNetwork(nn.Module):
    def __init__(self, num_labels):
        super(ClassificationNetwork, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        return pooled_output


def get_embeddings(model):
    if hasattr(model, 'embeddings'):
        embeddings = model.embeddings.word_embeddings
    elif hasattr(model, 'transformer'):
        # GPT-2 specific path
        embeddings = model.transformer.wte
    elif isinstance(model, (DPRContextEncoder, DPRQuestionEncoder)):
        # DPR specific path
        embeddings = model.ctx_encoder.bert_model.embeddings.word_embeddings
    else:
        raise AttributeError(f"Could not find embeddings in model of type {type(model).__name__}")
    return embeddings

def contriever_get_emb(model, input):
    return model(**input)

# def bert_get_emb(model, queries, tokenizer, device='cuda'):
#     # return model(**input).pooler_output
#     query_embeddings = []
#     for query in queries:
#         tokenized_input = tokenizer(query, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
#         with torch.no_grad():
#             input_ids = tokenized_input["input_ids"].to(device)
#             attention_mask = tokenized_input["attention_mask"].to(device)
#             query_embedding = model(input_ids, attention_mask)
#             query_embeddings.append(query_embedding)

#     query_embeddings = torch.stack(query_embeddings, dim=0)
#     query_embeddings = query_embeddings.squeeze(1)
    
#     return query_embeddings

def get_ada_embedding(client, text, model="text-embedding-3-small"):
  text = text.replace("\n", " ")
  return client.embeddings.create(input = [text], model=model).data[0].embedding


def bert_get_emb(model, input):
    if isinstance(model, (DPRContextEncoder, DPRQuestionEncoder)):
        return model(**input).pooler_output
    else:
        return model.bert(**input).pooler_output


def llama_get_emb(model, input):
    return model(**input).last_hidden_state[:, 0, :]



def bert_get_adv_emb(data, model, tokenizer, num_adv_passage_tokens, adv_passage_ids, adv_passage_attention, device='cuda'):
    """Get embeddings for adversarial passage.
    
    Args:
        data: Input data
        model: Model to use
        tokenizer: Tokenizer to use
        num_adv_passage_tokens: Number of adversarial passage tokens
        adv_passage_ids: Adversarial passage token IDs
        adv_passage_attention: Adversarial passage attention mask
        device: Device to use (default: 'cuda')
        
    Returns:
        query_embeddings: Query embeddings
    """
    # Check if CUDA is available, if not use CPU
    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'
        print("CUDA not available, using CPU instead")
    
    query_embeddings = []
    
    # Handle list format for MultiAgentPoison
    if isinstance(data, list):
        for item in data:
            ego = item.get("Ego", "")
            perception = item.get("Perception", "")
            query = f"{ego} {perception} NOTICE:"
            
            tokenized_input = tokenizer(query, truncation=True, max_length=512-num_adv_passage_tokens, return_tensors="pt")
            with torch.no_grad():
                input_ids = tokenized_input["input_ids"].to(device)
                attention_mask = tokenized_input["attention_mask"].to(device)
                suffix_adv_passage_ids = torch.cat((input_ids, adv_passage_ids), dim=1)
                suffix_adv_passage_attention = torch.cat((attention_mask, adv_passage_attention), dim=1)
                p_sent = {'input_ids': suffix_adv_passage_ids, 'attention_mask': suffix_adv_passage_attention}
            
            if isinstance(model, ClassificationNetwork) or isinstance(model, TripletNetwork):
                p_emb = bert_get_emb(model, p_sent)
            elif isinstance(model, RealmForOpenQA):
                p_emb = model(**p_sent).pooler_output
            else:
                p_emb = model(**p_sent).pooler_output
            query_embeddings.append(p_emb)
            
    # Handle dictionary format for other agents
    elif isinstance(data, dict):
        if "ego" in data.keys():
            for ego, perception in zip(data["ego"], data["perception"]):
                query = f"{ego} {perception} NOTICE:"

                tokenized_input = tokenizer(query, truncation=True, max_length=512-num_adv_passage_tokens, return_tensors="pt")
                with torch.no_grad():
                    input_ids = tokenized_input["input_ids"].to(device)
                    attention_mask = tokenized_input["attention_mask"].to(device)
                    suffix_adv_passage_ids = torch.cat((input_ids, adv_passage_ids), dim=1)
                    suffix_adv_passage_attention = torch.cat((attention_mask, adv_passage_attention), dim=1)
                    p_sent = {'input_ids': suffix_adv_passage_ids, 'attention_mask': suffix_adv_passage_attention}
                
                if isinstance(model, ClassificationNetwork) or isinstance(model, TripletNetwork):
                    p_emb = bert_get_emb(model, p_sent)
                elif isinstance(model, RealmForOpenQA):
                    p_emb = model(**p_sent).pooler_output
                else:
                    p_emb = model(**p_sent).pooler_output
                query_embeddings.append(p_emb)

        elif "question" in data.keys():
            for question in data["question"]:
                tokenized_input = tokenizer(question, padding='max_length', truncation=True, max_length=512-num_adv_passage_tokens, return_tensors="pt")
                with torch.no_grad():
                    input_ids = tokenized_input["input_ids"].to(device)
                    attention_mask = tokenized_input["attention_mask"].to(device)
                    suffix_adv_passage_ids = torch.cat((input_ids, adv_passage_ids), dim=1)
                    suffix_adv_passage_attention = torch.cat((attention_mask, adv_passage_attention), dim=1)
                    p_sent = {'input_ids': suffix_adv_passage_ids, 'attention_mask': suffix_adv_passage_attention}
                
                if isinstance(model, ClassificationNetwork) or isinstance(model, TripletNetwork):
                    p_emb = bert_get_emb(model, p_sent)
                elif isinstance(model, RealmForOpenQA):
                    p_emb = model(**p_sent).pooler_output
                else:
                    p_emb = model(**p_sent).pooler_output
                query_embeddings.append(p_emb)

    query_embeddings = torch.cat(query_embeddings, dim=0)
    return query_embeddings




def bert_get_cpa_emb(data, model, tokenizer, num_adv_passage_tokens, adv_passage_ids, adv_passage_attention, device='cuda'):
    query_embeddings = []
    
    if "ego" in data.keys():
        for ego, perception in zip(data["ego"], data["perception"]):
            query = f"{ego} {perception} NOTICE:"

            # tokenized_input = tokenizer(query, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
            tokenized_input = tokenizer(query, truncation=True, max_length=512-num_adv_passage_tokens, return_tensors="pt")
            with torch.no_grad():
                input_ids = tokenized_input["input_ids"].to(device)

                attention_mask = tokenized_input["attention_mask"].to(device)

                # suffix_adv_passage_ids = torch.cat((input_ids, adv_passage_ids), dim=1)
                suffix_adv_passage_ids = adv_passage_ids
                # suffix_adv_passage_attention = torch.cat((attention_mask, adv_passage_attention), dim=1)
                suffix_adv_passage_attention = adv_passage_attention
                # print("Input IDs length:", suffix_adv_passage_ids.shape[1])
                # print("Attention Mask length:", suffix_adv_passage_attention.shape[1])
                # input()
                # print('Init adv_passage', tokenizer.convert_ids_to_tokens(suffix_adv_passage_ids[0]))
                p_sent = {'input_ids': suffix_adv_passage_ids, 'attention_mask': suffix_adv_passage_attention}
            
            if isinstance(model, ClassificationNetwork) or isinstance(model, TripletNetwork):
                p_emb = bert_get_emb(model, p_sent)
            # elif isinstance(model, RealmEmbedder):
            #     p_emb = model(**p_sent).projected_score
            elif isinstance(model, RealmForOpenQA):
                p_emb = model(**p_sent).pooler_output
            else:
                p_emb = model(**p_sent).pooler_output
                # print('p_emb', p_emb.shape)
            query_embeddings.append(p_emb)

    elif "question" in data.keys():
        for question in data["question"]:
            tokenized_input = tokenizer(question, padding='max_length', truncation=True, max_length=512-num_adv_passage_tokens, return_tensors="pt")
            with torch.no_grad():
                input_ids = tokenized_input["input_ids"].to(device)
                attention_mask = tokenized_input["attention_mask"].to(device)
                suffix_adv_passage_ids = adv_passage_ids
                suffix_adv_passage_attention = adv_passage_attention
                p_sent = {'input_ids': suffix_adv_passage_ids, 'attention_mask': suffix_adv_passage_attention}
            
            if isinstance(model, ClassificationNetwork) or isinstance(model, TripletNetwork):
                p_emb = bert_get_emb(model, p_sent)
            elif isinstance(model, RealmForOpenQA):
                p_emb = model(**p_sent).pooler_output
            else:
                p_emb = model(**p_sent).pooler_output
            query_embeddings.append(p_emb)


    query_embeddings = torch.cat(query_embeddings, dim=0)

    return query_embeddings


def target_asr(data, sample_size, target_word, CoT_prefix, trigger_sequence, device='cuda'):

    api_key = "XXX"

    success_counter = 0
    for ego, perception in zip(data["ego"][:sample_size], data["perception"][:sample_size]):
        if perception[-1] == "\n":
            perception = perception[:-1]

        query = f"{CoT_prefix} {ego} {perception}NOTICE: {trigger_sequence}"
        
        for j in range(5):
            try:
                response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers={'Authorization': f'Bearer {api_key}'},
                json={'model': "gpt-3.5-turbo", "messages": [{"role": "user", "content": query}], 'max_tokens': 512, 'n': 1, 'temperature': 1}  # Adjust 'n' for the number of samples you want
                )
                data = response.json()
                output = data['choices'][0]['message']['content']
                driving_plan = output.split("Driving Plan:")[-1].strip()
                break
            except Exception as e:
                print("Error: ", Exception) 
                driving_plan = "none"
                output = None
                time.sleep(5)
                
        if target_word in driving_plan:
            success_counter += 1
    
    sample_ASR = success_counter / sample_size

    return sample_ASR



def target_word_prob(data, model, tokenizer, num_adv_passage_tokens, adv_passage_ids, adv_passage_attention, target_word, CoT_prefix, trigger_sequence, device='cuda'):

    target_word_token = tokenizer(target_word, return_tensors="pt")["input_ids"].to(device)
    total_prob = 0.0
    count = 0

    for ego, perception in zip(data["ego"], data["perception"]):
        if perception[-1] == "\n":
            perception = perception[:-1]

        query = f"{CoT_prefix} {ego} {perception}NOTICE: {trigger_sequence}"
        
        input_ids = tokenizer.encode(query, return_tensors="pt").to(device)
        attention_mask = torch.ones_like(input_ids).to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)
            logits = output.logits if hasattr(output, 'logits') else output[0]
            
            # Get probabilities for the target word
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            target_prob = probs[0, target_word_token[0, 1]].item()  # Using index 1 to skip [CLS] token
            total_prob += target_prob
            count += 1

        print("query", query)
        print("target probability:", target_prob)

    return total_prob / count if count > 0 else 0.0



def ance_get_emb(model, input):
    input.pop('token_type_ids', None)
    return model(input)["sentence_embedding"]

def load_models(model_code, device='cuda'):
    assert model_code in model_code_to_embedder_name, f"Model code {model_code} not supported!"
    
    # Check if CUDA is available, if not use CPU
    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'
        print("CUDA not available, using CPU instead")

    if 'contrastive' in model_code:
        model = TripletNetwork().to(device)
        model.load_state_dict(torch.load(model_code_to_embedder_name[model_code] + "/pytorch_model.bin", map_location=device))
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        get_emb = bert_get_emb
    elif 'classification' in model_code:
        model = ClassificationNetwork(num_labels=11).to(device)
        model.load_state_dict(torch.load(model_code_to_embedder_name[model_code] + "/pytorch_model.bin", map_location=device))
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        get_emb = bert_get_emb
    elif 'bert' in model_code:
        model = BertModel.from_pretrained('bert-base-uncased').to(device)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        get_emb = bert_get_emb
    elif 'llama' in model_code:
        # model = AutoModel.from_pretrained(model_code_to_embedder_name[model_code]).to(device)
        model = AutoModelForCausalLM.from_pretrained(
        # model_code_to_embedder_name[model_code], torch_dtype=torch.float16, device_map={"": device}).to(device)
        model_code_to_embedder_name[model_code], load_in_8bit=True, device_map={"": device})
        tokenizer = AutoTokenizer.from_pretrained(model_code_to_embedder_name[model_code])
        get_emb = llama_get_emb
    elif 'gpt2' in model_code:
        model = AutoModelForCausalLM.from_pretrained(model_code_to_embedder_name[model_code]).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_code_to_embedder_name[model_code])
        # get_emb = llama_get_emb
        get_emb = None
    elif 'dpr' in model_code and 'ance' not in model_code:
        model =  DPRContextEncoder.from_pretrained(model_code_to_embedder_name[model_code]).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_code_to_embedder_name[model_code])
        get_emb = bert_get_emb
    elif 'ance' in model_code:
        model = AutoModel.from_pretrained(model_code_to_embedder_name[model_code]).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_code_to_embedder_name[model_code])
        get_emb = bert_get_emb
    elif 'bge' in model_code:
        model = AutoModel.from_pretrained(model_code_to_embedder_name[model_code]).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_code_to_embedder_name[model_code])
        get_emb = bert_get_emb
    elif 'realm' in model_code and 'orqa' not in model_code:
        model = RealmEmbedder.from_pretrained(model_code_to_embedder_name[model_code]).realm.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_code_to_embedder_name[model_code])
        get_emb = bert_get_emb
    elif 'orqa' in model_code:
        model = RealmForOpenQA.from_pretrained(model_code_to_embedder_name[model_code]).embedder.realm.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_code_to_embedder_name[model_code])
        get_emb = bert_get_emb    
    elif 'ada' in model_code:
        
        import openai
        client = openai.OpenAI(api_key = api_key)
        model = "openai/ada"
        tokenizer = client
        get_emb = None

    else:
        raise NotImplementedError
    
    return model, tokenizer, get_emb
    

def load_ehr_memory(memory_log_dir):
    
    # get all the txt files under memory_log_dir
    memory_files = [f for f in os.listdir(memory_log_dir) if os.path.isfile(os.path.join(memory_log_dir, f)) and f.endswith('.txt')]

    long_term_memory = []
    for file in memory_files:
        with open(os.path.join(memory_log_dir, file), 'r') as f:
            # print(file)
            init_memory = f.read()
            example_split = init_memory.split('(END OF EXAMPLES)')
            init_memory = example_split[0]
            if len(example_split) > 1:
                new_experience = example_split[1]
            init_memory = init_memory.split('\n\n')
            for i in range(1, len(init_memory)-1):
                # if 'Question' not in init_memory[i]:
                #     continue
                item = init_memory[i]
                item = item.split('Question:')[-1]
                question = item.split('\nKnowledge:\n')[0]
                if len(question.split(' ')) > 20:
                    continue
                item = item.split('\nKnowledge:\n')[-1]
                knowledge = item.split('\nSolution:')[0]
                code = item.split('\nSolution:')[-1]
                new_item = {"question": question, "knowledge": knowledge, "code": code}
                long_term_memory.append(new_item)
                # print(new_item)
                # input()
            if len(example_split) > 1:
                # print("new_experience", new_experience)
                item = new_experience.split('Knowledge:\n')[-1]
                knowledge = item.split('Question:')[0]
                item = item.split('Question:')[-1]
                question = item.split('Solution:')[0]
                if len(question.split(' ')) > 20:
                    continue
                code = item.split('Solution:')[-1]
                new_item = {"question": question, "knowledge": knowledge, "code": code}
                long_term_memory.append(new_item)
            
    # get rid of the same questions
    long_term_memory = [dict(t) for t in {tuple(d.items()) for d in long_term_memory}]

    return long_term_memory

def load_db_ehr(database_samples_dir="EhrAgent/database/ehr_logs/logs_final", db_dir="EhrAgent/database/embedding", model_code="None", model=None, tokenizer=None, device='cuda'):

    long_term_memory = load_ehr_memory(database_samples_dir)

    if 'dpr' in model_code:
        if Path(f"{db_dir}/embeddings_{model_code}.pkl").exists():
            with open(f"{db_dir}/embeddings_{model_code}.pkl", "rb") as f:
                embeddings = pickle.load(f)
        else:
            embeddings = []
            for item in tqdm(long_term_memory):
                text = item["question"]

                tokenized_input = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
                input_ids = tokenized_input["input_ids"].to(device)
                attention_mask = tokenized_input["attention_mask"].to(device)

                with torch.no_grad():
                    query_embedding = model(input_ids, attention_mask).pooler_output

                query_embedding = query_embedding.detach().cpu().numpy().tolist()
                embeddings.append(query_embedding)

            with open(f"{db_dir}/embeddings_{model_code}.pkl", "wb") as f:
                pickle.dump(embeddings, f)

        embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
        db_embeddings = embeddings.squeeze(1)

    elif 'realm' in model_code and 'orqa' not in model_code:
        if Path(f"{db_dir}/embeddings_{model_code}.pkl").exists():
            with open(f"{db_dir}/embeddings_{model_code}.pkl", "rb") as f:
                embeddings = pickle.load(f)
        else:
            embeddings = []
            for item in tqdm(long_term_memory):
                text = item["question"]
                tokenized_input = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
                input_ids = tokenized_input["input_ids"].to(device)
                attention_mask = tokenized_input["attention_mask"].to(device)

                with torch.no_grad():
                    query_embedding = model(input_ids, attention_mask).pooler_output

                query_embedding = query_embedding.detach().cpu().numpy().tolist()
                embeddings.append(query_embedding)

            with open(f"{db_dir}/embeddings_{model_code}.pkl", "wb") as f:
                pickle.dump(embeddings, f)

        embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
        db_embeddings = embeddings.squeeze(1)

    elif 'bge' in model_code:
        if Path(f"{db_dir}/embeddings_{model_code}.pkl").exists():
            with open(f"{db_dir}/embeddings_{model_code}.pkl", "rb") as f:
                embeddings = pickle.load(f)
        else:
            embeddings = []
            for item in tqdm(long_term_memory):
                text = item["question"]
                tokenized_input = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
                input_ids = tokenized_input["input_ids"].to(device)
                attention_mask = tokenized_input["attention_mask"].to(device)

                with torch.no_grad():
                    query_embedding = model(input_ids, attention_mask).pooler_output

                query_embedding = query_embedding.detach().cpu().numpy().tolist()
                embeddings.append(query_embedding)

            with open(f"{db_dir}/embeddings_{model_code}.pkl", "wb") as f:
                pickle.dump(embeddings, f)

        embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
        db_embeddings = embeddings.squeeze(1)

    elif 'orqa' in model_code:
        if Path(f"{db_dir}/embeddings_{model_code}.pkl").exists():
            with open(f"{db_dir}/embeddings_{model_code}.pkl", "rb") as f:
                embeddings = pickle.load(f)
        else:
            embeddings = []
            for item in tqdm(long_term_memory):
                text = item["question"]
                tokenized_input = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
                input_ids = tokenized_input["input_ids"].to(device)
                attention_mask = tokenized_input["attention_mask"].to(device)

                with torch.no_grad():
                    query_embedding = model(input_ids, attention_mask).pooler_output

                query_embedding = query_embedding.detach().cpu().numpy().tolist()
                embeddings.append(query_embedding)

            with open(f"{db_dir}/embeddings_{model_code}.pkl", "wb") as f:
                pickle.dump(embeddings, f)

        embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
        db_embeddings = embeddings.squeeze(1)
    
    elif 'ada' in model_code:
        if Path(f"{db_dir}/embeddings_{model_code}.pkl").exists():
            with open(f"{db_dir}/embeddings_{model_code}.pkl", "rb") as f:
                embeddings = pickle.load(f)
        else:
            embeddings = []
            for item in tqdm(long_term_memory):
                text = item["question"]
                # try:
                # while True:
                query_embedding = get_ada_embedding(tokenizer, text)
                    # break
                # except:
                #     continue

                embeddings.append(query_embedding)

        with open(f"{db_dir}/embeddings_{model_code}.pkl", "wb") as f:
            pickle.dump(embeddings, f)

        embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
        db_embeddings = embeddings


    return db_embeddings, long_term_memory
    

def load_db_qa(database_samples_dir="ReAct/database/strategyqa_train_paragraphs.json", db_dir="data/memory", model_code="None", model=None, tokenizer=None, device='cuda'):

    if 'dpr' in model_code:
        if Path(f"{db_dir}/embeddings_{model_code}.pkl").exists():
            with open(f"{db_dir}/embeddings_{model_code}.pkl", "rb") as f:
                embeddings = pickle.load(f)
        else:
            embeddings = []
                        
            with open(database_samples_dir, "rb") as f:
                database_samples = json.load(f)#[:20000]


            for paragraph_id in tqdm(database_samples):
                text = database_samples[paragraph_id]["content"]
                tokenized_input = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
                input_ids = tokenized_input["input_ids"].to(device)
                attention_mask = tokenized_input["attention_mask"].to(device)

                with torch.no_grad():
                    query_embedding = model(input_ids, attention_mask).pooler_output

                query_embedding = query_embedding.detach().cpu().numpy().tolist()
                # print(query_embedding.shape)
                embeddings.append(query_embedding)
        
            with open(f"{db_dir}/embeddings_{model_code}.pkl", "wb") as f:
                pickle.dump(embeddings, f)

        embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
        db_embeddings = embeddings.squeeze(1)
    
    elif 'realm' in model_code and 'orqa' not in model_code:
        if Path(f"{db_dir}/embeddings_{model_code}.pkl").exists():
            with open(f"{db_dir}/embeddings_{model_code}.pkl", "rb") as f:
                embeddings = pickle.load(f)
        else:
            embeddings = []
                        
            with open(database_samples_dir, "rb") as f:
                database_samples = json.load(f)

            for paragraph_id in tqdm(database_samples):
                text = database_samples[paragraph_id]["content"]
                tokenized_input = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
                input_ids = tokenized_input["input_ids"].to(device)
                attention_mask = tokenized_input["attention_mask"].to(device)

                with torch.no_grad():
                    query_embedding = model(input_ids, attention_mask).pooler_output

                query_embedding = query_embedding.detach().cpu().numpy().tolist()
                # print(query_embedding.shape)
                embeddings.append(query_embedding)

            with open(f"{db_dir}/embeddings_{model_code}.pkl", "wb") as f:
                pickle.dump(embeddings, f)

        embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
        db_embeddings = embeddings.squeeze(1)

    elif 'bge' in model_code:
        if Path(f"{db_dir}/embeddings_{model_code}.pkl").exists():
            with open(f"{db_dir}/embeddings_{model_code}.pkl", "rb") as f:
                embeddings = pickle.load(f)
        else:
            embeddings = []
                        
            with open(database_samples_dir, "rb") as f:
                database_samples = json.load(f)

            for paragraph_id in tqdm(database_samples):
                text = database_samples[paragraph_id]["content"]
                tokenized_input = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
                input_ids = tokenized_input["input_ids"].to(device)
                attention_mask = tokenized_input["attention_mask"].to(device)

                with torch.no_grad():
                    query_embedding = model(input_ids, attention_mask).pooler_output

                query_embedding = query_embedding.detach().cpu().numpy().tolist()
                # print(query_embedding.shape)
                embeddings.append(query_embedding)

            with open(f"{db_dir}/embeddings_{model_code}.pkl", "wb") as f:
                pickle.dump(embeddings, f)

        embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
        db_embeddings = embeddings.squeeze(1)

    elif 'orqa' in model_code:
        if Path(f"{db_dir}/embeddings_{model_code}.pkl").exists():
            with open(f"{db_dir}/embeddings_{model_code}.pkl", "rb") as f:
                embeddings = pickle.load(f)
        else:
            embeddings = []
                        
            with open(database_samples_dir, "rb") as f:
                database_samples = json.load(f)

            for paragraph_id in tqdm(database_samples):
                text = database_samples[paragraph_id]["content"]
                tokenized_input = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
                input_ids = tokenized_input["input_ids"].to(device)
                attention_mask = tokenized_input["attention_mask"].to(device)

                with torch.no_grad():
                    query_embedding = model(input_ids, attention_mask).pooler_output

                query_embedding = query_embedding.detach().cpu().numpy().tolist()
                # print(query_embedding.shape)
                embeddings.append(query_embedding)

            with open(f"{db_dir}/embeddings_{model_code}.pkl", "wb") as f:
                pickle.dump(embeddings, f)

        embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
        db_embeddings = embeddings.squeeze(1)

    else:
        raise NotImplementedError
    
    return db_embeddings


def load_db_ad(database_samples_dir="agentdriver/data/finetune/data_samples_train.json", db_dir="data/memory", model_code="None", model=None, tokenizer=None, device='cuda'):

    
    if 'contrastive' in model_code:
        if Path(f"{db_dir}/embeddings_{model_code}.pkl").exists():
            with open(f"{db_dir}/embeddings_{model_code}.pkl", "rb") as f:
                embeddings = pickle.load(f)
        else:
            embeddings = []
                        
            with open(database_samples_dir, "rb") as f:
                database_samples = json.load(f)[:20000]

            for sample in tqdm(database_samples):
                ego = sample["ego"]
                perception = sample["perception"]
                prompt = f"{ego} {perception}"
                tokenized_input = tokenizer(prompt, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
                with torch.no_grad():
                    input_ids = tokenized_input["input_ids"].to(device)
                    attention_mask = tokenized_input["attention_mask"].to(device)
                    query_embedding = model(input_ids, attention_mask)
                    embeddings.append(query_embedding)
            with open(f"{db_dir}/embeddings_{model_code}.pkl", "wb") as f:
                pickle.dump(embeddings, f)

        embeddings = torch.stack(embeddings, dim=0).to(device)
        db_embeddings = embeddings.squeeze(1)

    elif 'classification' in model_code:
        if Path(f"{db_dir}/embeddings_{model_code}.pkl").exists():
            with open(f"{db_dir}/embeddings_{model_code}.pkl", "rb") as f:
                embeddings = pickle.load(f)
        else:
            embeddings = []
     
            with open(database_samples_dir, "rb") as f:
                database_samples = json.load(f)[:20000]

            for sample in tqdm(database_samples):
                ego = sample["ego"]
                perception = sample["perception"]
                prompt = f"{ego} {perception}"
                tokenized_input = tokenizer(prompt, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
                with torch.no_grad():
                    input_ids = tokenized_input["input_ids"].to(device)
                    attention_mask = tokenized_input["attention_mask"].to(device)
                    query_embedding = model(input_ids, attention_mask)
                    embeddings.append(query_embedding)
            with open(f"{db_dir}/embeddings_{model_code}.pkl", "wb") as f:
                pickle.dump(embeddings, f)
        
        embeddings = torch.stack(embeddings, dim=0).to(device)
        db_embeddings = embeddings.squeeze(1)


    elif 'bert' in model_code:
        if Path(f"{db_dir}/bert_embeddings.pkl").exists():
            with open(f"{db_dir}/bert_embeddings.pkl", "rb") as f:
                embeddings = pickle.load(f)
        else:
            embeddings = []
                        
            with open(database_samples_dir, "rb") as f:
                database_samples = json.load(f)[:20000]

            for sample in tqdm(database_samples):
                ego = sample["ego"]
                perception = sample["perception"]
                prompt = f"{ego} {perception}"
                tokenized_input = tokenizer(prompt, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
                with torch.no_grad():
                    input_ids = tokenized_input["input_ids"].to(device)
                    attention_mask = tokenized_input["attention_mask"].to(device)
                    query_embedding = model(input_ids, attention_mask).pooler_output
                    embeddings.append(query_embedding)
            with open(f"{db_dir}/bert_embeddings.pkl", "wb") as f:
                pickle.dump(embeddings, f)
        
        embeddings = torch.stack(embeddings, dim=0).to(device)
        db_embeddings = embeddings.squeeze(1)

    elif 'dpr' in model_code:
        if Path(f"{db_dir}/embeddings_{model_code}.pkl").exists():
            with open(f"{db_dir}/embeddings_{model_code}.pkl", "rb") as f:
                embeddings = pickle.load(f)
        else:
            embeddings = []
     
            with open(database_samples_dir, "rb") as f:
                database_samples = json.load(f)[:20000]

            for sample in tqdm(database_samples):
                ego = sample["ego"]
                perception = sample["perception"]
                prompt = f"{ego} {perception}"
                tokenized_input = tokenizer(prompt, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
                with torch.no_grad():
                    input_ids = tokenized_input["input_ids"].to(device)
                    attention_mask = tokenized_input["attention_mask"].to(device)
                    query_embedding = model(input_ids, attention_mask).pooler_output
                    query_embedding = query_embedding.detach().cpu().numpy().tolist()
                    embeddings.append(query_embedding)
            with open(f"{db_dir}/embeddings_{model_code}.pkl", "wb") as f:
                pickle.dump(embeddings, f)
    
        embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
        db_embeddings = embeddings.squeeze(1)

    elif 'bge' in model_code:
        if Path(f"{db_dir}/embeddings_{model_code}.pkl").exists():
            with open(f"{db_dir}/embeddings_{model_code}.pkl", "rb") as f:
                embeddings = pickle.load(f)
        else:
            embeddings = []
     
            with open(database_samples_dir, "rb") as f:
                database_samples = json.load(f)[:20000]

            for sample in tqdm(database_samples):
                ego = sample["ego"]
                perception = sample["perception"]
                prompt = f"{ego} {perception}"
                tokenized_input = tokenizer(prompt, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
                with torch.no_grad():
                    input_ids = tokenized_input["input_ids"].to(device)
                    attention_mask = tokenized_input["attention_mask"].to(device)
                    query_embedding = model(input_ids, attention_mask).pooler_output
                    embeddings.append(query_embedding)
            with open(f"{db_dir}/embeddings_{model_code}.pkl", "wb") as f:
                pickle.dump(embeddings, f)
        
        embeddings = torch.stack(embeddings, dim=0).to(device)
        db_embeddings = embeddings.squeeze(1)

    elif 'realm' in model_code and 'orqa' not in model_code:
        if Path(f"{db_dir}/embeddings_{model_code}.pkl").exists():
            with open(f"{db_dir}/embeddings_{model_code}.pkl", "rb") as f:
                embeddings = pickle.load(f)
        else:
            embeddings = []
     
            with open(database_samples_dir, "rb") as f:
                database_samples = json.load(f)[:20000]

            for sample in tqdm(database_samples):
                ego = sample["ego"]
                perception = sample["perception"]
                prompt = f"{ego} {perception}"
                tokenized_input = tokenizer(prompt, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
                with torch.no_grad():
                    input_ids = tokenized_input["input_ids"].to(device)
                    attention_mask = tokenized_input["attention_mask"].to(device)
                    query_embedding = model(input_ids, attention_mask).pooler_output #.projected_score
                    # print("query_embedding", query_embedding)
                    # input()
                    embeddings.append(query_embedding)
            with open(f"{db_dir}/embeddings_{model_code}.pkl", "wb") as f:
                pickle.dump(embeddings, f)
        
        embeddings = torch.stack(embeddings, dim=0).to(device)
        db_embeddings = embeddings.squeeze(1)

    elif 'orqa' in model_code:
        if Path(f"{db_dir}/embeddings_{model_code}.pkl").exists():
            with open(f"{db_dir}/embeddings_{model_code}.pkl", "rb") as f:
                embeddings = pickle.load(f)
        else:
            embeddings = []
     
            with open(database_samples_dir, "rb") as f:
                database_samples = json.load(f)[:20000]

            for sample in tqdm(database_samples):
                ego = sample["ego"]
                perception = sample["perception"]
                prompt = f"{ego} {perception}"
                tokenized_input = tokenizer(prompt, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
                with torch.no_grad():
                    input_ids = tokenized_input["input_ids"].to(device)
                    attention_mask = tokenized_input["attention_mask"].to(device)
                    query_embedding = model(input_ids, attention_mask).pooler_output
                    # print("query_embedding", query_embedding)
                    # input()
                    embeddings.append(query_embedding)
            with open(f"{db_dir}/embeddings_{model_code}.pkl", "wb") as f:
                pickle.dump(embeddings, f)
        
        embeddings = torch.stack(embeddings, dim=0).to(device)
        db_embeddings = embeddings.squeeze(1)
        
    db_embeddings = embeddings.squeeze(1)

    return db_embeddings




###### Utils for Perturbation ######

def add_zeros_to_numbers(input_string, padding="0", desired_digits=3):
    # Define a regular expression pattern to match numbers with optional decimal points and negative signs
    pattern = r"([-+]?\d*\.\d+|[-+]?\d+)"
    
    # Find all matches of numbers in the input string
    def replace(match):
        num = match.group()
        if '.' in num:
            # Split the number into integer and fractional parts
            integer_part, fractional_part = num.split('.')
            
            # Calculate the number of zeros to add to the fractional part
            zeros_to_add = max(0, desired_digits - len(fractional_part))
            
            # Append zeros to the fractional part
            # modified_fractional_part = fractional_part + '0000000' * zeros_to_add
            modified_fractional_part = fractional_part + padding * zeros_to_add
            # Combine the integer part and modified fractional part
            modified_num = integer_part + '.' + modified_fractional_part
        else:
            # For integers, just add zeros after the number
            modified_num = num + '.' + padding * desired_digits
        
        return modified_num
    
    # existing_string, input_string = input_string.split("Historical Trajectory")

    modified_string = re.sub(pattern, replace, input_string)
    # modified_string = re.sub(pattern, replace, existing_string)
    # modified_string = modified_string + "Historical Trajectory" + input_string
    
    return modified_string




class AgentDriverDataset(Dataset):
    def __init__(self, json_file, split_ratio=0.8, train=True):
        with open(json_file, 'r') as file:
            data = json.load(file)
        split_index = int(len(data) * split_ratio)
        if train:
            self.data = data[:split_index]
        else:
            self.data = data[split_index:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError("Index out of bounds")
        sample = self.data[idx]
        # for key in sample:
            # if sample[key] is None:
            #     print(f"None found in key: {key}, index: {idx}")
            #     input()
        # Convert data to the required format, process it if necessary
        return {
            'token': sample['token'],
            'ego': sample['ego'],
            'perception': sample['perception'],
            'commonsense': sample['commonsense'] if sample['commonsense'] is not None else "",
            'experiences': sample['experiences'] if sample['experiences'] is not None else "",
            'chain_of_thoughts': sample['chain_of_thoughts'] if sample['chain_of_thoughts'] is not None else "",
            'reasoning': sample['reasoning'] if sample['reasoning'] is not None else "",
            'planning_target': sample['planning_target'] if sample['planning_target'] is not None else "",
        }


class StrategyQADataset(Dataset):
    def __init__(self, json_file, split_ratio=0.8, train=True):
        try:
            with open(json_file, 'r') as file:
                data = json.load(file)
        except:
            with jsonlines.open(json_file) as reader:
                data = [item for item in reader]

        split_index = int(len(data) * split_ratio)
        if train:
            self.data = data[:split_index]
        else:
            self.data = data[split_index:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError("Index out of bounds")
        sample = self.data[idx]
        # for key in sample:
            # if sample[key] is None:
            #     print(f"None found in key: {key}, index: {idx}")
            #     input()
        # Convert data to the required format, process it if necessary
        return {
            # 'qid': sample['qid'],
            # 'term': sample['term'],
            # 'question': "Question: " + sample['question'],
            'question': sample['question'],
            # 'description': sample['description'] if sample['description'] is not None else "",
            # 'facts': sample['facts'] if sample['facts'] is not None else "",
            # 'decomposition': sample['decomposition'] if sample['decomposition'] is not None else "",
        }


class EHRAgentDataset(Dataset):
    def __init__(self, json_file, split_ratio=0.8, train=True):
        with open(json_file, 'r') as file:
            data = json.load(file)

        split_index = int(len(data) * split_ratio)
        if train:
            self.data = data[:split_index]
        else:
            self.data = data[split_index:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError("Index out of bounds")
        sample = self.data[idx]
        # for key in sample:
            # if sample[key] is None:
            #     print(f"None found in key: {key}, index: {idx}")
            #     input()
        # Convert data to the required format, process it if necessary
        return {
            'question': sample['template'],
            # 'answer': sample['answer'],
        }

def load_db_multiagent(database_samples_dir="MyAgent/result/MyAgentPoison", db_dir="MyAgent/result/MyAgentPoison", model_code="None", model=None, tokenizer=None, device='cuda'):
    """
    Load the database embeddings for MultiAgentPoison
    
    Args:
        database_samples_dir: Directory containing the memory buffer
        db_dir: Directory containing the FAISS index
        model_code: Model code to use
        model: Model to use for embeddings
        tokenizer: Tokenizer to use
        device: Device to use
        
    Returns:
        db_embeddings: Database embeddings
    """
    # Load the memory buffer
    memory_file = os.path.join(database_samples_dir, "memory_buffer.pkl")
    with open(memory_file, "rb") as f:
        memory_buffer = pickle.load(f)
    
    # Load the FAISS index
    index_file = os.path.join(db_dir, "faiss_index.faiss")
    index = faiss.read_index(index_file)
    
    # Get the embeddings from the index
    if index.ntotal == 0:
        # If index is empty, create a default embedding with correct dimension
        dimension = 768  # DPR model dimension
        db_embeddings = torch.randn(10, dimension).to(device)  # Create 10 random embeddings
    else:
        # Get all embeddings from the index
        db_embeddings = torch.tensor(index.reconstruct_n(0, index.ntotal)).to(device)
        
        # Ensure the embeddings have the correct dimension
        if db_embeddings.shape[1] != 768:
            # If dimension is incorrect, create new embeddings with correct dimension
            dimension = 768
            db_embeddings = torch.randn(index.ntotal, dimension).to(device)
    
    return db_embeddings

class MultiAgentPoisonDataset(Dataset):
    def __init__(self, json_file, split_ratio=0.8, train=True):
        with open(json_file, 'r') as file:
            data = json.load(file)
        split_index = int(len(data) * split_ratio)
        if train:
            self.data = data[:split_index]
        else:
            self.data = data[split_index:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError("Index out of bounds")
        sample = self.data[idx]
        
        # Convert data to the required format
        return {
            'ego': f"Episode {sample['episode']}, Step {sample['steps']}, Total Reward {sample['total_reward']:.2f}",
            'perception': f"Memory Size {sample['memory_size']}, Agent Rewards {sample['agent_rewards']}"
        }
