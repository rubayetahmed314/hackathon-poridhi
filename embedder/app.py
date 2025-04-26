import os
import time
import json
import redis
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

r = redis.Redis(host='redis', port=6379, decode_responses=True)
# model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
pubsub = r.pubsub()
pubsub.subscribe('query_input')

# Path to your ONNX model and tokenizer
onnx_model_path = "./xenova/model_quantized.onnx"
tokenizer_path = "./xenova"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Create ONNX Runtime session with CPU threading control
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Control CPU cores usage
session_options.intra_op_num_threads = 4  # Number of threads used to run individual operators
# session_options.inter_op_num_threads = 1  # Number of threads used to run operators in parallel

# Create inference session with the configured options
session = ort.InferenceSession(
    onnx_model_path, 
    session_options, 
    providers=['CPUExecutionProvider']
)

# Function to encode sentences using ONNX model
def encode_sentences(sentences):
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='np', max_length=128)
    
    # Get input names and run inference
    ort_inputs = {k: v for k, v in encoded_input.items() if k in session.get_inputs()[0].name or k in ['input_ids', 'attention_mask', 'token_type_ids']}
    ort_outputs = session.run(None, ort_inputs)
    
    # For sentence transformers, typically need mean pooling of token embeddings
    attention_mask = encoded_input['attention_mask']
    token_embeddings = ort_outputs[0]
    
    # Perform mean pooling
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
    embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1) / np.sum(input_mask_expanded, axis=1)
    
    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    return embeddings.tolist()  # Convert to list for Chroma compatibility


for message in pubsub.listen():
    if message['type'] == 'message':
        data = json.loads(message['data'])
        query = data['query']
        print(query, flush=True)
        start = time.time()
        embedding = encode_sentences([query])[0]
        print(time.time() - start, flush=True)
        r.publish('embed_output', json.dumps({'query': query, 'embedding': embedding}))
