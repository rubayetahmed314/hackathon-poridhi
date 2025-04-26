import os
import time
import json
import redis
from sentence_transformers import CrossEncoder

r = redis.Redis(host='redis', port=6379, decode_responses=True)
model = CrossEncoder('./ms-marco-MiniLM-L12-v2')
pubsub = r.pubsub()
pubsub.subscribe('retriever_output')

for message in pubsub.listen():
    if message['type'] == 'message':
        data = json.loads(message['data'])
        query = data['query']
        products = data['results']
        pairs = [
            (query, f"{product['title']} | {product['description']} | {product['brand']}") for product in products
        ]
        start = time.time()
        scores = [float(score) for score in model.predict(pairs)]
        print(time.time() - start, flush=True)
        for product in products:
            product.pop('description', None)
        ranked = sorted(zip(products, scores), key=lambda x: -x[1])[:5]
        # print("Top results:", ranked, flush=True)
        r.publish('reranker_output', json.dumps({'results': ranked}))

