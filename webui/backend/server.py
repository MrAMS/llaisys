import asyncio
import websockets
import os
import json
import uuid
import llaisys
import argparse
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from test_utils import *

import os, io, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
# Dictionary to store active users and their websockets
active_users = {}
history_users = {}
id_users = {}
id_tot = 0

# --- 主要 WebSocket 处理函数 ---
async def handle(websocket):
    global id_tot, id_users
    user_id = str(uuid.uuid4())
    active_users[user_id] = websocket
    print(f"New client connected with ID: {user_id}", flush=True)
    id_users[user_id] = []

    try:
        # Inform the client of their assigned user ID
        await websocket.send(json.dumps({"type": "user_id", "id": user_id}))

        async for message in websocket:
            print(f"Received message from client {user_id}: {message}", flush=True)
            # Make sure we have a conversation history for this user
            if user_id not in history_users:
                history_users[user_id] = []
                id_users[user_id] = []

            history_users[user_id].append(
                {"role": "user", "content": message}
            )
            input_content = tokenizer.apply_chat_template(
                conversation=history_users[user_id],
                add_generation_prompt=True,
                tokenize=False,
            )
            history_users[user_id].append(
                {"role": "assistant", "content": ""}
            )
            # The input needs to be a list of tokens, not the raw text
            input_tokens = tokenizer.encode(input_content)

            model.add_request(id_tot, input_tokens, 40)
            id_users[user_id].append(id_tot)
            id_tot += 1
    
    except websockets.exceptions.ConnectionClosed:
        print(f"Client {user_id} disconnected", flush=True)
        # Clean up the user from the dictionary upon disconnection
        if user_id in active_users:
            del active_users[user_id]
        if user_id in id_users:
            del id_users[user_id]
        if user_id in history_users:
            del history_users[user_id]
    except Exception as e:
        print(f"Error for client {user_id}: {e}", flush=True)

async def step_model():
    while True:
        try:
            # Step the model and get results
            _, results = model.step()
            for result in results:
                id = result.get('id')
                user_id = next((key for key, value in id_users.items() if id in value), None)

                if user_id and user_id in active_users:
                    websocket = active_users[user_id]
                    
                    # Decode the tokens back to text
                    decoded_text = tokenizer.decode(result.get('tokens'), skip_special_tokens=True)
                    global history_users
                    history_users[user_id][-1]['content'] += decoded_text
                    print(user_id[:4], ":", history_users[user_id][-1]['content'], flush=True)
                    # Send the response to the user
                    await websocket.send(decoded_text)
        except Exception as e:
            print(f"Error during model step: {e}", flush=True)
        
        # This is crucial for not blocking the event loop
        await asyncio.sleep(0.01)

def load_tokenizer(model_path=None, device_name="cpu"):
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    if model_path and os.path.isdir(model_path):
        print(f"Loading model from local path: {model_path}")
    else:
        print(f"Loading model from Hugging Face: {model_id}")
        model_path = snapshot_download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return tokenizer, model_path

def load_llaisys_model(model_path, device_name):
    model = llaisys.models.Qwen2(model_path, llaisys_device(device_name))
    return model

async def main():
    print("WebSocket server starting", flush=True)
    # Start the model stepping loop as a background task
    asyncio.create_task(step_model())
    
    async with websockets.serve(
        handle,
        "0.0.0.0",
        int(os.environ.get('PORT', 8090))
    ) as server:
        print("WebSocket server running on port 8090", flush=True)
        await asyncio.Future()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--max_steps", default=128, type=int)
    parser.add_argument("--top_p", default=0.8, type=float)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    top_p, top_k, temperature = args.top_p, args.top_k, args.temperature
    if args.test:
        top_p, top_k, temperature = 1.0, 1, 1.0

    tokenizer, model_path = load_tokenizer(args.model, args.device)

    model = load_llaisys_model(model_path, args.device)
    
    asyncio.run(main())