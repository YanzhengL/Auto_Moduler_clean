import os
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from openai import OpenAI
from tqdm import tqdm
import json
import time
import llm_loader
import re
from uuid import uuid4
from nltk.tokenize import sent_tokenize
from pdfminer.high_level import extract_text


# --- CONFIG ---
INPUT_DIR = "../Knowledge/pdfss"
OUTPUT_DIR = "../Knowledge/knowledge_chunks"
os.makedirs(OUTPUT_DIR, exist_ok=True)
USE_OPENAI = False  # Set to False if using a local LLM
CHUNK_SIZE = 1000  # Character chunk size
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_API_KEY = "your-api-key-here"  # Set this from env variable or config file
model_local, tokenizer = llm_loader.load_deepseek_model(model_path="../models/model.safetensors")

# --- OpenAI client setup ---
if USE_OPENAI:
    client = OpenAI(api_key=OPENAI_API_KEY)

def extract_sections_by_titles(pdf_path):
    elements = partition_pdf(filename=pdf_path,
                             strategy="ocr_only")
    sections = []
    current_section = ""

    for el in elements:
        text = el.text.strip() if el.text else ""
        if not text:
            continue

        # Split when a new section title appears
        if el.category == "Title" or el.category == "Section Header":
            if current_section:
                sections.append(current_section.strip())
            current_section = text + "\n"
        else:
            current_section += text + " "

    # Append the last section
    if current_section.strip():
        sections.append(current_section.strip())

    return [sec for sec in sections if len(sec) > 30]

def chunk_text(text_list, max_chars=CHUNK_SIZE):
    current_chunk = ""
    chunks = []
    for paragraph in text_list:
        if len(current_chunk) + len(paragraph) <= max_chars:
            current_chunk += paragraph + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def extract_theory(chunk, model_local, tokenizer, max_new_tokens=512):
    prompt = f"""<|user|>
You are an expert in electrical engineering. Extract and summarize the key DC-DC converter design theory from the following text. Include important equations or control principles.

Text:
\"\"\"
{chunk}
\"\"\"

Summarize in technical bullet points:
<|assistant|>
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model_local.device)
    outputs = model_local.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.3,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt from the output to keep only the response
    if "<|assistant|>" in decoded_output:
        return decoded_output.split("<|assistant|>")[-1].strip()
    else:
        return decoded_output.strip()

def process_pdf(pdf_file, model_local, tokenizer):
    print(f"Processing: {pdf_file}")
    #raw_text = extract_sections_by_titles(pdf_file)
    raw_text = extract_sections_no_split_sentences(pdf_file)
    #text = raw_text.replace('\\n', '\n')
    chunks = chunk_text(raw_text)

    output_data = []
    for i, chunk in enumerate(tqdm(chunks, desc="Extracting key theories")):
        summary = extract_theory(chunk, model_local, tokenizer)
        output_data.append({
            "source_pdf": Path(pdf_file).name,
            "chunk_index": i,
            "original_text": chunk,
            "summary": summary
        })
        time.sleep(0.1)  # prevent rate limit
    return output_data

def process_pdf_summarize_sections(pdf_file, model_local, tokenizer):
    print(f"Processing: {pdf_file}")
    #raw_text = extract_sections_by_titles(pdf_file)
    raw_text = extract_sections_no_split_sentences(pdf_file)

    output_data = []
    for i, section in enumerate(tqdm(raw_text, desc="Extracting key theories")):
        summary = extract_theory(section["text"], model_local, tokenizer)
        section["summary"] = summary
        output_data.append(section)
        time.sleep(0.1)  # prevent rate limit
    return output_data

def extract_sections_no_split_sentences(pdf_path):
    elements = partition_pdf(filename=pdf_path,
                             #strategy="ocr_only",
                             strategy="hi_res",  # Better for complex layouts
                           #  model_name="yolox",           # YOLOX layout detection
                             languages=["eng"],          # English OCR
                             extract_images_in_pdf=False,   # Optional: skip images to speed up
                           #  include_metadata = True
    )

    sections = []
    current_section = {
        "id": str(uuid4()),
        "section": "unknown",
        "category": "NarrativeText",
        "text": "",
        "source_pdf": pdf_path,
    }

    for el in elements:
        text = el.text.strip() if el.text else ""
        if not text or len(text) < 10:
            continue

        # Robust category extraction
        category = getattr(el, 'category', None)
        if not category and hasattr(el, 'metadata') and hasattr(el.metadata, 'category'):
            category = el.metadata.category or "NarrativeText"
        print(category)
        # Fallback: detect section headers from text manually
        is_probable_section = (
            category == "Section Header" or
            re.match(r'^\d{1,2}[\.\)]?\s?[A-Z][^\n]{1,60}$', text)  # e.g. "1. Introduction", "2) Methods"
        )

        if is_probable_section:
            # Save previous section if it has content
            if current_section["text"].strip():
                sections.append(current_section)
            # Start new section
            current_section = {
                "id": str(uuid4()),
                "section": text.lower(),
                "category": category or "Section Header",
                "text": "",
                "source_pdf": pdf_path,
            }
            continue

        # Append sentences to current section
        sentences = sent_tokenize(text)
        current_section["text"] += " " + " ".join(sentences)

    # Save last section
    if current_section["text"].strip():
        sections.append(current_section)

    # Filter out tiny fragments
    return [s for s in sections if len(s["text"].strip()) > 30]

def extract_text_summarize(pdf_path, model_local, tokenizer):
    text = extract_text(pdf_path)
    summary = extract_theory(text,model_local, tokenizer)
    return summary

def main():
    for filename in os.listdir(INPUT_DIR):
        if not filename.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(INPUT_DIR, filename)
        #result = process_pdf_summarize_sections(pdf_path, model_local, tokenizer)
        result = extract_sections_no_split_sentences(pdf_path)
        #result = extract_sections_by_titles(pdf_path)
        #result = extract_text_summarize(pdf_path)
        out_file = os.path.join(OUTPUT_DIR, f"{Path(filename).stem}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
