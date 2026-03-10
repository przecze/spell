import os
import time
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification
from optimum.onnxruntime import ORTModelForTokenClassification
import numpy as np


app = FastAPI(title="Token Classification API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global model and tokenizer
model = None
tokenizer = None
model_type = None


class InferenceRequest(BaseModel):
    text: str


class TokenResult(BaseModel):
    token: str
    probability: float
    position: int


class InferenceResponse(BaseModel):
    tokens: List[TokenResult]
    inference_time_ms: float
    model_type: str  # "pytorch" or "onnx"
    text: str


def load_latest_checkpoint():
    """Load the most recent checkpoint from /checkpoints directory"""
    checkpoint_dir = Path("/checkpoints")
    
    if not checkpoint_dir.exists():
        raise RuntimeError("No checkpoints directory found")
    
    checkpoint_paths = list(checkpoint_dir.glob("*"))
    if not checkpoint_paths:
        raise RuntimeError("No checkpoints found")
    
    latest_checkpoint = max(checkpoint_paths, key=os.path.getmtime)
    print(f"Loading checkpoint: {latest_checkpoint}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(latest_checkpoint)
    
    # Check if ONNX is disabled
    disable_onnx = os.getenv("DISABLE_ONNX", "false").lower() == "true"
    
    start_time = time.time()
    if disable_onnx:
        print("Loading PyTorch model (ONNX disabled)")
        model = AutoModelForTokenClassification.from_pretrained(latest_checkpoint, num_labels=1)
        model.eval()
        print(f"PyTorch model loaded in {time.time() - start_time} seconds")
        return model, tokenizer, "pytorch"
    else:
        print("Loading ONNX model")
        model = ORTModelForTokenClassification.from_pretrained(
            latest_checkpoint, export=True
        )
        print(f"ONNX model loaded in {time.time() - start_time} seconds")
        return model, tokenizer, "onnx"


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global model, tokenizer, model_type
    model, tokenizer, model_type = load_latest_checkpoint()
    print("Both PyTorch and ONNX models loaded successfully")

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """
    Predict token-level probabilities for input text using both PyTorch and ONNX models
    """
    # Tokenize input
    inputs = tokenizer(
        request.text, 
        return_tensors="pt", 
        add_special_tokens=True,
        truncation=True,
        max_length=512
    )
    
    start_time = time.time()
    outputs = model(**inputs)
    logits = outputs.logits
    
    if model_type == "pytorch":
        with torch.no_grad():
            probs = F.sigmoid(logits[0, :, 0])
    else:  # ONNX
        probs = F.sigmoid(torch.tensor(logits[0, :, 0]))
    
    inference_time = (time.time() - start_time) * 1000
    
    # Convert tokens back to strings
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Create token results
    token_results = []
    for i, (token, prob) in enumerate(zip(tokens, probs)):
        token_results.append(TokenResult(
            token=token,
            probability=float(prob),
            position=i
        ))
    
    return InferenceResponse(
        tokens=token_results,
        inference_time_ms=inference_time,
        model_type=model_type,
        text=request.text
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
