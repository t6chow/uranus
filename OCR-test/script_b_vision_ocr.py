#!/usr/bin/env python3
# How to run: 
# Option 1 (API Key): Create .env file with GOOGLE_API_KEY, then run: python script_b_vision_ocr.py <image_path>
# Option 2 (Service Account): Set environment: export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"
# Dependencies: pip install google-cloud-vision pillow python-dotenv requests

import sys
import json
import os
import base64
import requests
from google.cloud import vision
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def ocr_vision_with_api_key(image_path, api_key):
    """Perform OCR using Google Cloud Vision API with API key via REST API."""
    # Read and encode image
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    encoded_image = base64.b64encode(content).decode('utf-8')
    
    # Prepare request
    url = f'https://vision.googleapis.com/v1/images:annotate?key={api_key}'
    
    request_data = {
        'requests': [
            {
                'image': {
                    'content': encoded_image
                },
                'features': [
                    {
                        'type': 'DOCUMENT_TEXT_DETECTION',
                        'maxResults': 50
                    }
                ]
            }
        ]
    }
    
    # Make request
    response = requests.post(url, json=request_data)
    
    if response.status_code != 200:
        raise Exception(f"Vision API error: {response.status_code} - {response.text}")
    
    result = response.json()
    
    if 'error' in result:
        raise Exception(f"Vision API error: {result['error']}")
    
    # Extract blocks from response
    blocks = []
    block_id = 0
    
    responses = result.get('responses', [])
    if responses and 'fullTextAnnotation' in responses[0]:
        full_text = responses[0]['fullTextAnnotation']
        if 'pages' in full_text:
            for page in full_text['pages']:
                for block in page.get('blocks', []):
                    # Get block text by joining paragraphs
                    block_text = ""
                    for paragraph in block.get('paragraphs', []):
                        for word in paragraph.get('words', []):
                            word_text = "".join([symbol.get('text', '') for symbol in word.get('symbols', [])])
                            block_text += word_text + " "
                    
                    block_text = block_text.strip()
                    
                    if block_text:
                        # Get bounding box
                        bounding_box = block.get('boundingBox', {})
                        vertices = bounding_box.get('vertices', [])
                        if vertices:
                            x_coords = [v.get('x', 0) for v in vertices]
                            y_coords = [v.get('y', 0) for v in vertices]
                            bbox = [
                                min(x_coords),  # x0
                                min(y_coords),  # y0
                                max(x_coords),  # x1
                                max(y_coords)   # y1
                            ]
                        else:
                            bbox = [0, 0, 0, 0]
                        
                        # Get confidence if available
                        confidence = block.get('confidence')
                        
                        block_data = {
                            'id': block_id,
                            'text': block_text,
                            'bbox': bbox,
                            'confidence': confidence
                        }
                        blocks.append(block_data)
                        block_id += 1
    
    return {
        'blocks': blocks,
        'total_blocks': len(blocks)
    }

def ocr_vision_with_service_account(image_path):
    """Perform OCR using Google Cloud Vision API with service account credentials."""
    # Initialize Vision API client
    client = vision.ImageAnnotatorClient()
    
    # Read image file
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    
    # Perform document text detection
    response = client.document_text_detection(image=image)
    
    if response.error.message:
        raise Exception(f"Vision API error: {response.error.message}")
    
    # Extract blocks from response
    blocks = []
    block_id = 0
    
    # Process full text annotation
    if response.full_text_annotation:
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                # Get block text by joining paragraphs
                block_text = ""
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        word_text = "".join([symbol.text for symbol in word.symbols])
                        block_text += word_text + " "
                
                block_text = block_text.strip()
                
                if block_text:
                    # Get bounding box
                    vertices = block.bounding_box.vertices
                    if vertices:
                        x_coords = [v.x for v in vertices]
                        y_coords = [v.y for v in vertices]
                        bbox = [
                            min(x_coords),  # x0
                            min(y_coords),  # y0
                            max(x_coords),  # x1
                            max(y_coords)   # y1
                        ]
                    else:
                        bbox = [0, 0, 0, 0]
                    
                    # Get confidence if available
                    confidence = block.confidence if hasattr(block, 'confidence') else None
                    
                    block_data = {
                        'id': block_id,
                        'text': block_text,
                        'bbox': bbox,
                        'confidence': confidence
                    }
                    blocks.append(block_data)
                    block_id += 1
    
    return {
        'blocks': blocks,
        'total_blocks': len(blocks)
    }

def ocr_vision(image_path):
    """Perform OCR using Google Cloud Vision API - auto-detect authentication method."""
    api_key = os.getenv('GOOGLE_API_KEY')
    
    if api_key:
        print("Using API key authentication...")
        return ocr_vision_with_api_key(image_path, api_key)
    else:
        print("Using service account authentication...")
        return ocr_vision_with_service_account(image_path)

def main():
    if len(sys.argv) != 2:
        print("Usage: python script_b_vision_ocr.py <image_path>")
        print("Authentication options:")
        print("  1. Create .env file with GOOGLE_API_KEY")
        print("  2. Set GOOGLE_APPLICATION_CREDENTIALS environment variable")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        # Perform OCR
        result = ocr_vision(image_path)
        
        # Print short plain text to stdout
        plain_text = ' '.join([block['text'] for block in result['blocks']])
        print("Vision API OCR Result:")
        print(plain_text[:500] + "..." if len(plain_text) > 500 else plain_text)
        
        # Write JSON to file
        with open('./ocr_output_vision.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\nJSON output written to: ./ocr_output_vision.json")
        print(f"Total blocks detected: {result['total_blocks']}")
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        print("Check your authentication:")
        print("  - Ensure .env file contains GOOGLE_API_KEY, OR")
        print("  - Ensure GOOGLE_APPLICATION_CREDENTIALS points to a valid service account key")
        sys.exit(1)

if __name__ == "__main__":
    main()
