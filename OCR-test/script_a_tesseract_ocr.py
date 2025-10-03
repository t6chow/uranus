#!/usr/bin/env python3
# How to run: python script_a_tesseract_ocr.py <image_path>
# Dependencies: pip install pytesseract pillow opencv-python numpy

import sys
import json
import cv2
import numpy as np
from PIL import Image
import pytesseract

def preprocess_image(image_path):
    """Preprocess image for better OCR results."""
    # Read image with OpenCV
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(gray)
    
    # Optional upscaling for better OCR
    height, width = contrast_enhanced.shape
    if width < 1000:  # Upscale if image is small
        scale_factor = 2
        contrast_enhanced = cv2.resize(contrast_enhanced, 
                                     (width * scale_factor, height * scale_factor), 
                                     interpolation=cv2.INTER_CUBIC)
    
    # Noise reduction
    denoised = cv2.bilateralFilter(contrast_enhanced, 9, 75, 75)
    
    return denoised

def ocr_tesseract(image_path):
    """Perform OCR using pytesseract and return structured JSON."""
    # Preprocess image
    processed_img = preprocess_image(image_path)
    
    # Convert to PIL Image for pytesseract
    pil_img = Image.fromarray(processed_img)
    
    # Get detailed OCR data
    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
    
    # Build structured blocks
    blocks = []
    block_id = 0
    
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if text:  # Only include non-empty text
            block = {
                'id': block_id,
                'text': text,
                'left': int(data['left'][i]),
                'top': int(data['top'][i]),
                'width': int(data['width'][i]),
                'height': int(data['height'][i]),
                'conf': float(data['conf'][i])
            }
            blocks.append(block)
            block_id += 1
    
    return {
        'blocks': blocks,
        'total_blocks': len(blocks)
    }

def main():
    if len(sys.argv) != 2:
        print("Usage: python script_a_tesseract_ocr.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        # Perform OCR
        result = ocr_tesseract(image_path)
        
        # Print plain text to stdout
        plain_text = ' '.join([block['text'] for block in result['blocks']])
        print("OCR Result:")
        print(plain_text)
        
        # Write JSON to file
        with open('./ocr_output.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\nJSON output written to: ./ocr_output.json")
        print(f"Total blocks detected: {result['total_blocks']}")
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
