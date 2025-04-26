import os
import json
import tqdm
import time
import numpy as np
import onnxruntime as ort

from transformers import AutoTokenizer
from qdrant_client import QdrantClient
from elasticsearch import Elasticsearch
from qdrant_client.models import VectorParams
import numpy as np

# Qdrant client configuration
qdrant_host = "localhost"  # docker container
qdrant_port = 6333  # Default port for Qdrant
client = QdrantClient(host=qdrant_host, port=qdrant_port)

# Initialize the Elasticsearch client
es = Elasticsearch("http://localhost:9200",
    basic_auth=('elastic', 'your_password'),
    request_timeout=120,
    max_retries=10,
    retry_on_timeout=True
)

# Name of the collection in Qdrant
collection_name = "toothpaste_practice"
try:
    existing_collections = [item[1][0].name for item in client.get_collections()]
except:
    existing_collections = []

# Check if the collection exists and create it if not
if collection_name not in existing_collections:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance="Cosine")
    )
    print(f"Collection '{collection_name}' created.")
else:
    print(f"Collection '{collection_name}' already exists.")


onnx_model_path = "./embedder/xenova/model_quantized.onnx"
tokenizer_path = "./embedder/xenova"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Create ONNX Runtime session with CPU threading control
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Control CPU cores usage
session_options.intra_op_num_threads = 8  # Number of threads used to run individual operators
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


documents = json.load(open('products_compact.json', 'r', encoding='utf-8'))
for doc in tqdm.tqdm(documents):
    merged_content = f"{doc['title']} | {doc['description']} | {doc['price']} tk price"

    # Encode the text to get embeddings
    embedding = encode_sentences([merged_content])[0]  # Get the first (and only) embedding

    # Prepare metadata for Qdrant
    metadata = {
        "sku": doc['sku'],
        "product_url": doc['product_url'],
        "image_url": doc['image_url']
    }

    product_id = es.search(index=collection_name, query={
        "term": {
            "sku.keyword": doc['sku']
        }
    })['hits']['hits'][0]['_id']

    # Upload the embedding and metadata to Qdrant
    client.upsert(
        collection_name=collection_name,
        points=[
            {
                "id": product_id,
                "vector": embedding,
                "payload": metadata
            }
        ]
    )

print("Embeddings and metadata uploaded successfully.")
