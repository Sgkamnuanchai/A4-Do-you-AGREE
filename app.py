from flask import Flask, request, jsonify, render_template
import os
import torch
import json
from sklearn.metrics.pairwise import cosine_similarity
from Bert import BERT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("./model/word2id.json", "r") as f:
    word2id = json.load(f)

n_layers = 12    # number of Encoder of Encoder Layer
n_heads  = 12    # number of heads in Multi-Head Attention
d_model  = 768  # Embedding Size
d_ff = d_model * 4  # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 2
max_len = 1000
vocab_size = len(word2id)
model = BERT(
    n_layers, 
    n_heads, 
    d_model, 
    d_ff, 
    d_k, 
    n_segments, 
    vocab_size, 
    max_len, 
    device
).to(device)

model.load_state_dict(torch.load("./model/s_bert_model.pth", map_location=device))

model.eval()


def custom_tokenizer(text, max_seq_length=128):
    tokens = text.lower().split()
    token_ids = [word2id.get(word, word2id["[MASK]"]) for word in tokens]   
    token_ids = token_ids[:max_seq_length]
    attention_mask = [1] * len(token_ids)
    padding_length = max_seq_length - len(token_ids)
    token_ids.extend([word2id["[PAD]"]] * padding_length)
    attention_mask.extend([0] * padding_length) 

    return {
        "input_ids": token_ids,
        "attention_mask": attention_mask
    }

# Define mean pooling function
def mean_pool(token_embeds, attention_mask):
    in_mask = attention_mask.unsqueeze(-1).expand(
        token_embeds.size()
    ).float()
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
        in_mask.sum(1), min=1e-9
    )
    return pool

def calculate_similarity(model, custom_tokenizer, sentence_a, sentence_b, device):
    inputs_a = custom_tokenizer(sentence_a)  
    inputs_b = custom_tokenizer(sentence_b) 

    inputs_ids_a = torch.tensor(inputs_a['input_ids']).unsqueeze(0).to(device)
    attention_a = torch.tensor(inputs_a['attention_mask']).unsqueeze(0).to(device)

    inputs_ids_b = torch.tensor(inputs_b['input_ids']).unsqueeze(0).to(device)
    attention_b = torch.tensor(inputs_b['attention_mask']).unsqueeze(0).to(device)

    u = model.get_last_hidden_state(inputs_ids_a, segment_ids=torch.zeros_like(inputs_ids_a).to(device))
    v = model.get_last_hidden_state(inputs_ids_b, segment_ids=torch.zeros_like(inputs_ids_b).to(device))

    u_mean_pool = mean_pool(u, attention_a).detach().cpu().numpy().reshape(-1)
    v_mean_pool = mean_pool(v, attention_b).detach().cpu().numpy().reshape(-1)

    similarity_score = cosine_similarity(u_mean_pool.reshape(1, -1), v_mean_pool.reshape(1, -1))[0, 0]

    # Format similarity_score to 4 decimal places
    similarity_score = f"{similarity_score:.4f}"

    if float(similarity_score) > 0.8:
        label = "Entailment"
    elif float(similarity_score) < 0.4:
        label = "Contradiction"
    else:
        label = "Neutral"

    return similarity_score, label

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/compare", methods=["POST"])
def compare():
    sentence_a = request.json.get("sentence_a")
    sentence_b = request.json.get("sentence_b")

    similarity_score, label = calculate_similarity(model, custom_tokenizer, sentence_a, sentence_b, device)

    return jsonify({
        "similarity_score": similarity_score,
        "label": label
    })

if __name__ == '__main__':
    app.run(debug=True)
