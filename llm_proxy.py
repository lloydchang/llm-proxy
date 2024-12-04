# file_name: llm_proxy.py

import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from typing import List
import requests

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

# Base models for OpenAI compatibility
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    max_tokens: int
    temperature: float

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: dict

# Endpoint to handle OpenAI-like requests
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        # Log request headers and body
        logging.debug(f"Request headers: {request.headers}")
        request_body = await request.json()
        logging.debug(f"Request body: {request_body}")

        # Parse request body
        chat_request = ChatCompletionRequest(**request_body)

        # Extract relevant data
        messages = chat_request.messages
        max_tokens = chat_request.max_tokens
        temperature = chat_request.temperature

        # Simulate LLM API call (replace this with actual LLM logic)
        llm_response = mock_llm_response(messages, max_tokens, temperature)  # Replace mock_llm_response with real call

        # Construct OpenAI-compatible response
        response = ChatCompletionResponse(
            id="unique-id",
            object="chat.completion",
            created=1234567890,  # Add actual timestamp
            model="model-name",
            choices=[ChatCompletionResponseChoice(index=0, message=Message(role="assistant", content="response"), finish_reason="stop")],
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
        )

        # Log response
        logging.debug(f"Response: {response.json()}")

        return response
    except ValidationError as e:
        logging.error(f"Validation error: {e.errors()}")
        return JSONResponse(status_code=422, content={"detail": e.errors()})
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

def mock_llm_response(messages, max_tokens, temperature):
    """
    Simulate a response from LLM.
    Replace this with actual LLM integration logic.
    """
    return {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "response"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "total_tokens": 20
        }
    }

