# File: D:/AI/Gits/hebrew-tutor-data-pipeline/scripts/convert_hebrew_grammar_pdf.py
# Pipeline for Hebrew grammar PDFs using Azure Document Intelligence, compatible with Conda unstructured_env-3.11.
# Fixed text extraction: Use 'line.content' instead of 'span.content' for Azure result.

import os
import cv2
import re
import json
import unicodedata
from pathlib import Path
import numpy as np
from PIL import Image
import fitz  # PyMuPDF (conda install pymupdf)
from transformers import pipeline
import hebrew_tokenizer as ht
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Load Azure credentials
load_dotenv()
AZURE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

# File paths
BASE_DIR = Path("D:/AI/Projects/HEBREW TRAINING AI AGENT/GRAMMAR")
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = Path("D:/AI/Models")

def setup_directories():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def preprocess_hebrew_image(image):
    img = np.array(image)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(img)

def convert_pdf_to_images(pdf_path, output_dir, dpi=300):
    doc = fitz.open(pdf_path)
    processed_images = []
    output_dir.mkdir(exist_ok=True)
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        processed_img = preprocess_hebrew_image(img)
        output_path = output_dir / f"page_{page_num+1}.png"
        processed_img.save(output_path)
        processed_images.append(output_path)
    doc.close()
    return processed_images

def ocr_hebrew_page(image_path):
    if not AZURE_ENDPOINT or not AZURE_KEY:
        raise ValueError("Azure endpoint or key not set in .env")
    client = DocumentIntelligenceClient(endpoint=AZURE_ENDPOINT, credential=AzureKeyCredential(AZURE_KEY))
    with open(image_path, "rb") as img:
        poller = client.begin_analyze_document("prebuilt-read", body=img)
        result = poller.result()
    text = ' '.join([line.content for page in result.pages if page.lines for line in page.lines if line.content])
    # Placeholder for D-Nikud (integrate from Dicta-IL, e.g., https://arxiv.org/abs/2402.00075)
    # from d_nikud import Diacritizer
    # dnikud = Diacritizer(model_path="D:/AI/Models/d_nikud")
    # text = dnikud.diacritize(text)
    return text

def process_hebrew_text(text):
    try:
        ner_pipeline = pipeline('ner', model='dicta-il/dictabert-ner', aggregation_strategy='simple')
        morph_pipeline = pipeline('token-classification', model='dicta-il/dictabert-morph')
    except Exception as e:
        print(f"Error loading DictaBERT models: {e}")
        return {'entities': [], 'morphological_analysis': [], 'processed_text': text}
    tokens = list(ht.tokenize(text, with_whitespaces=False))
    hebrew_tokens = [token[1] for token in tokens if token[0] == 'HEBREW']
    clean_text = ' '.join(hebrew_tokens)
    entities = ner_pipeline(clean_text) if clean_text else []
    morphology = morph_pipeline(clean_text) if clean_text else []
    return {
        'entities': entities,
        'morphological_analysis': morphology,
        'processed_text': clean_text
    }

def structure_grammar_data(text, morphological_data):
    normalized_text = unicodedata.normalize('NFC', text)
    undiacritized = re.sub(r'[\u05B0-\u05C7]', '', normalized_text)
    nikud_map = [1 if c in '\u05B0-\u05C7' else 0 for c in normalized_text]
    dagesh_map = [1 if c == '\u05BC' else 0 for c in normalized_text]
    return {
        'text': normalized_text,
        'undiacritized': undiacritized,
        'nikud_classification': {
            'nikud': nikud_map,
            'dagesh': dagesh_map
        },
        'morphological_analysis': morphological_data['morphological_analysis']
    }

def validate_hebrew_output(text):
    unicode_checks = {
        'proper_normalization': unicodedata.is_normalized('NFC', text),
        'no_encoding_errors': not bool(re.search(r'[\uFFFD]', text)),
        'hebrew_character_presence': bool(re.search(r'[\u0590-\u05FF]', text))
    }
    return {
        'unicode_quality': unicode_checks,
        'morphological_consistency': None,  # Requires YAP
        'semantic_coherence': None  # Requires DictaBERT
    }

def main():
    setup_directories()
    pdf_files = [BASE_DIR / "Gesenius-HebrewGrammar.pdf", BASE_DIR / "hebrew_grammar_davidson.pdf"]
    for pdf_path in pdf_files:
        if not pdf_path.exists():
            print(f"PDF not found: {pdf_path}")
            continue
        book_name = pdf_path.stem
        book_output_dir = OUTPUT_DIR / book_name
        book_output_dir.mkdir(exist_ok=True)
        print(f"Processing {book_name} with Azure Document Intelligence...")
        image_paths = convert_pdf_to_images(pdf_path, book_output_dir)
        all_structured_data = []
        for image_path in image_paths:
            raw_text = ocr_hebrew_page(image_path)
            processed_data = process_hebrew_text(raw_text)
            structured_data = structure_grammar_data(raw_text, processed_data)
            validation_results = validate_hebrew_output(structured_data['text'])
            page_data = {
                'page_number': int(image_path.stem.split('_')[1]),
                'structured_data': structured_data,
                'validation_results': validation_results
            }
            all_structured_data.append(page_data)
        output_file = book_output_dir / f"{book_name}_structured.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_structured_data, f, ensure_ascii=False, indent=2)
        print(f"Completed {book_name}. Output: {output_file}")

if __name__ == "__main__":
    main()