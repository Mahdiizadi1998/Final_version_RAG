"""
Ollama Client Functions
Wrapper functions for Ollama API interactions
"""

import os
import base64
import requests
from typing import Optional, List, Dict, Any

# Global configuration
OLLAMA_BASE = os.environ.get('OLLAMA_BASE', 'http://localhost:11434')


def ollama_generate(
    model: str,
    prompt: str,
    images: Optional[List[str]] = None,
    system: Optional[str] = None,
    temperature: float = 0.1,
    timeout: int = 180
) -> str:
    """
    Generate text using Ollama API.
    
    Args:
        model: Model name (e.g., 'llama3.1:8b', 'llava:7b')
        prompt: Input prompt text
        images: Optional list of image paths for vision models
        system: Optional system prompt
        temperature: Sampling temperature (default: 0.1 for deterministic outputs)
        timeout: Request timeout in seconds
        
    Returns:
        Generated text response or error message
    """
    try:
        # Build payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        # Add system prompt if provided
        if system:
            payload["system"] = system
        
        # Add base64-encoded images if provided
        if images:
            encoded_images = []
            for img_path in images:
                try:
                    with open(img_path, 'rb') as img_file:
                        img_data = img_file.read()
                        encoded = base64.b64encode(img_data).decode('utf-8')
                        encoded_images.append(encoded)
                except Exception as e:
                    print(f"Warning: Could not encode image {img_path}: {e}")
            
            if encoded_images:
                payload["images"] = encoded_images
        
        # Make API request
        response = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json=payload,
            timeout=timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '')
        else:
            return f"Error: API returned status {response.status_code}"
            
    except requests.exceptions.Timeout:
        return "Error: Request timed out"
    except requests.exceptions.RequestException as e:
        return f"Error: Request failed - {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


def ollama_embed(model: str, text: str) -> Optional[List[float]]:
    """
    Generate embeddings using Ollama API.
    
    Args:
        model: Model name (e.g., 'llama3.1:8b')
        text: Input text to embed
        
    Returns:
        Embedding vector as list of floats, or None on error
    """
    try:
        payload = {
            "model": model,
            "prompt": text
        }
        
        response = requests.post(
            f"{OLLAMA_BASE}/api/embeddings",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('embedding')
        else:
            print(f"Error: API returned status {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def test_connection() -> bool:
    """
    Test Ollama server connection.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        response = ollama_generate(
            model="llama3.1:8b",
            prompt="Say 'SYSTEM READY' and nothing else.",
            temperature=0.0,
            timeout=30
        )
        
        if "SYSTEM READY" in response or "System ready" in response.lower():
            print("✓ Ollama connection test successful")
            print(f"  Response: {response.strip()}")
            return True
        else:
            print(f"⚠ Unexpected response: {response}")
            return False
            
    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the connection
    print("Testing Ollama connection...")
    test_connection()
