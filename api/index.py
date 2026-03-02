import os
import asyncio
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
from PIL import Image
import io

app = FastAPI()

# CORS configuration
ALLOWED_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Studio API configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyA7KcOv-UF3likV_sfZhEcAiKHM4qrHtj4")
genai.configure(api_key=GOOGLE_API_KEY)

# Available Google models (user can choose from these)
AVAILABLE_MODELS = [
    "gemini-pro",
    "gemini-pro-vision",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-pro-vision",
    "gemini-1.5-flash-vision"
]

class ChatRequest(BaseModel):
    prompt: str
    models: List[str]  # User selects which models to use
    image: Optional[str] = None  # Base64 encoded image for vision models

async def fetch_ai_response(model: str, prompt: str, image_data: Optional[str] = None):
    try:
        # Check if it's a vision model and has image
        is_vision_model = "vision" in model.lower() or model in ["gemini-pro-vision", "gemini-1.5-pro-vision", "gemini-1.5-flash-vision"]
        
        # Initialize the model
        generation_model = genai.GenerativeModel(model)
        
        if is_vision_model and image_data:
            try:
                # Decode base64 image (handle both with and without data URL prefix)
                if ',' in image_data:
                    image_bytes = base64.b64decode(image_data.split(',')[1])
                else:
                    image_bytes = base64.b64decode(image_data)
                
                image = Image.open(io.BytesIO(image_bytes))
                
                # Generate response with image
                response = await asyncio.to_thread(
                    generation_model.generate_content,
                    [prompt, image]
                )
            except Exception as img_error:
                return model, f"Error processing image: {str(img_error)}"
        else:
            # Text-only generation
            response = await asyncio.to_thread(
                generation_model.generate_content,
                prompt
            )
        
        # Check response
        if response and hasattr(response, 'text') and response.text:
            return model, response.text
        elif response and hasattr(response, 'parts'):
            # Handle structured response
            text_parts = [part.text for part in response.parts if hasattr(part, 'text')]
            if text_parts:
                return model, ' '.join(text_parts)
            else:
                return model, "Error: No text content in response"
        else:
            return model, "Error: No response generated"
            
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg.lower():
            return model, "Error: Invalid or missing API key"
        elif "quota" in error_msg.lower():
            return model, "Error: API quota exceeded"
        elif "model not found" in error_msg.lower():
            return model, f"Error: Model '{model}' not available"
        else:
            return model, f"Error: {error_msg}"

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "available_models": AVAILABLE_MODELS
    }

@app.get("/api/test")
async def test_endpoint():
    return {
        "message": "API is working!",
        "available_models": AVAILABLE_MODELS
    }

@app.get("/api/models")
async def get_models():
    """Return list of available models"""
    return {
        "models": AVAILABLE_MODELS,
        "count": len(AVAILABLE_MODELS)
    }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Google API Key not configured")
    
    if not request.models:
        raise HTTPException(status_code=400, detail="No models specified")
    
    if len(request.models) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 models allowed per request")
    
    # Validate models
    invalid_models = [m for m in request.models if m not in AVAILABLE_MODELS]
    if invalid_models:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid models: {', '.join(invalid_models)}. Available models: {', '.join(AVAILABLE_MODELS)}"
        )
    
    try:
        # Create tasks for each selected model
        tasks = [fetch_ai_response(model, request.prompt, request.image) for model in request.models]
        results = await asyncio.gather(*tasks)
        
        # Format response exactly like index.py
        response_data = {model: text for model, text in results}
        
        # Add metadata about which models were used
        return {
            "responses": response_data,
            "metadata": {
                "models_used": request.models,
                "has_image": request.image is not None
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# For Vercel serverless function
@app.get("/")
async def root():
    return {
        "message": "STWAI Backend API",
        "version": "1.0.0",
        "endpoints": [
            "/api/health",
            "/api/test", 
            "/api/models",
            "/api/chat"
        ]
    }

# This is for Vercel serverless function
handler = app
