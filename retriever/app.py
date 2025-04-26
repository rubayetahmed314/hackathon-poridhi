import redis
import time
import json
from qdrant_client import QdrantClient
from elasticsearch import Elasticsearch

# Redis setup
r = redis.Redis(host='redis', port=6379, decode_responses=True)
pubsub = r.pubsub()
pubsub.subscribe('embed_output')

es = Elasticsearch("http://elasticsearch:9200",
    basic_auth=('elastic', 'your_password'),
    request_timeout=120,
    max_retries=10,
    retry_on_timeout=True
)
qdrant_client = QdrantClient(host='qdrant', port=6333)
collection_name = "toothpaste_practice"

# Ensure ES is up
try:
    es.info()
    print("Connected to Elasticsearch")
except Exception as e:
    print("Failed to connect to Elasticsearch", e)

for message in pubsub.listen():
    if message['type'] != 'message':
        continue
    data = json.loads(message['data'])
    print("Query Text:", data['query'], flush=True)

    embedding = data['embedding']
    query_text = data['query']

    start = time.time()

    # Qdrant search
    qdrant_hits = qdrant_client.query_points(
        collection_name=collection_name,
        query=embedding,
        limit=5
    ).points
    qdrant_ids = [hit.id for hit in qdrant_hits]
    merged_results = []

    for _id in qdrant_ids:
        # product = es.search(index=collection_name, query={
        #     "term": {
        #         "sku.keyword": item['sku']
        #     }
        # })['hits']['hits'][0]
        product = es.get(index=collection_name, id=_id)
        merged_results.append(product['_source'])

    # Elasticsearch search
    es_results = es.search(index=collection_name, size=5, query={
        "multi_match": {
            "query": query_text,
            "fields": ["title", "description"]
        }
    },
    sort=[{"_score": {"order": "desc"}}]  # sort by score descending
    )["hits"]["hits"]
    merged_results.extend([hit["_source"] for hit in es_results if hit["_id"] not in qdrant_ids])
    print(time.time() - start, flush=True)

    print("Merged results:", len(merged_results), flush=True)
    r.publish('retriever_output', json.dumps({
        'query': query_text,
        'results': merged_results
    }))
