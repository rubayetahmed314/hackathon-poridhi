import os
import json
import redis
import asyncio
import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import Config, Server

REDIS_CHANNEL_INPUT = "query_input"
REDIS_CHANNEL_OUTPUT = "reranker_output"

# Redis setup
redis_client = redis.Redis(host="redis", port=6379, decode_responses=True)
pubsub = redis_client.pubsub()
pubsub.subscribe(REDIS_CHANNEL_OUTPUT)

# Socket.IO setup
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Create a Socket.IO ASGI app and mount it to FastAPI
socket_app = socketio.ASGIApp(socketio_server=sio, other_asgi_app=app)

@app.get("/")
async def root():
    return {"message": "Server is running"}

@sio.event
async def connect(sid, environ):
    print("User connected:", sid)

@sio.event
async def disconnect(sid):
    print("User disconnected:", sid)

@sio.event
async def query(sid, data):
    print(f"Received query from {sid}: {data}", flush=True)
    query_text = data.get('query').strip()
    if query_text:
        redis_client.publish(REDIS_CHANNEL_INPUT, json.dumps({"query": query_text}))
    else:
        await sio.emit('error', {'error': 'Empty query'}, to=sid)

async def redis_listener():
    print("Started Redis listener")
    while True:
        message = pubsub.get_message()
        if message and message['type'] == 'message':
            data = json.loads(message['data'])
            await sio.emit('result', data)
        await asyncio.sleep(0.1)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(redis_listener())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(socket_app, host="0.0.0.0", port=8000)