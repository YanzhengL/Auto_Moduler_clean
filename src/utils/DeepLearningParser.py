import fitz  # PyMuPDF
import numpy as np
import layoutparser as lp
import pytesseract
import re
import os
import json
import cv2
from PIL import Image, ImageDraw
from pathlib import Path
from bs4 import BeautifulSoup

def classify_by_regex(pil_image, block, original_label, block_label):
    x1, y1, x2, y2 = map(int, block.coordinates)
    cropped = pil_image.crop((x1, y1, x2, y2))
    cropped.save(os.path.join(OUTPUT_DIR, f"{block_label}.png"))

    # OCR with HOCR output to detect font style (bold)
    hocr = pytesseract.image_to_pdf_or_hocr(cropped, extension='hocr')
    soup = BeautifulSoup(hocr, 'html.parser')
    text = ""
    is_bold = False

    for span in soup.find_all('span', class_='ocrx_word'):
        word = span.get_text()
        title = span.get('title', '')
        text += word + " "
        if 'bold' in title.lower():
            is_bold = True

    text = text.strip()

    # Caption keyword checks
    is_figure = re.search(r'\bFigure \d+-\d+\b', text)
    is_table = re.search(r'\bTable[\s\-]*\d+', text, re.IGNORECASE)
    is_equation = re.search(r'\(\d+[-–]\d+\)', text)
    is_part_title = re.search(r'\bPART\s+\d+', text, re.IGNORECASE)

    # Short block check
    block_height = y2 - y1
    block_width = x2 - x1
    aspect_ratio = block_width / block_height if block_height != 0 else 0
    short_block = len(text.split()) < 30 and aspect_ratio < 5
    print("is_caption:", is_figure, ", is_short:", short_block)
    # Label classification
    if is_part_title:
        print("✔ PART title detected:", text)
        return "Title"
    if is_figure and len(text.split()) < 20:
        print("✔ Real Figure caption detected:", text)
        return "Caption"
    elif is_table and is_bold and short_block:
        print("✔ Real Table caption detected:", text)
        return "Table"
    elif is_equation and short_block:
        print("✔ Equation detected:", text)
        return "Equation"
    else:
        print("✘ Not a caption, keeping label:", original_label, "| Text:", text)
        return original_label

def white_out_regions(image, layout, keep_labels=["Text", "Title", "List"]):
    draw = ImageDraw.Draw(image)
    for block in layout:
        if block.type not in keep_labels:
            x1, y1, x2, y2 = map(int, block.coordinates)
            draw.rectangle([x1, y1, x2, y2], fill="white")
    return image

def extract_text_blocks(image, layout, page_num, keep_labels=["Text", "Title", "List"], padding = 10):
    data = []
    img_width, img_height = image.size
    for block in layout:
        if block.type in keep_labels:
            x1, y1, x2, y2 = map(int, block.coordinates)
            # Expand the box within image bounds
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(img_width, x2 + padding)
            y2 = min(img_height, y2 + padding + 10)
            crop = image.crop((x1, y1, x2, y2))
            # Convert to OpenCV grayscale
            img = np.array(crop.convert("L"))
            # Apply adaptive threshold to create high-contrast image
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, blockSize=15, C=10)
            text = pytesseract.image_to_string(img).strip()
            if text:
                data.append({
                    "page": page_num,
                    "label": block.type,
                    "coordinates": [x1, y1, x2, y2],
                    "text": text
                })
    return data

def crop_to_content_fixed_margins(image: Image.Image, left: int = 75, right: int = 75, top: int = 120, bottom: int = 120) -> Image.Image:
    """
    Crops fixed margins from the scanned page image.

    Args:
        image (PIL.Image): Original page image.
        left (int): Margin to remove from left side in pixels.
        right (int): Margin to remove from right side in pixels.
        top (int): Margin to remove from top in pixels.
        bottom (int): Margin to remove from bottom in pixels.

    Returns:
        PIL.Image: Cropped image.
    """
    width, height = image.size
    x_min = max(left, 0)
    x_max = min(width - right, width)
    y_min = max(top, 0)
    y_max = min(height - bottom, height)

    return image.crop((x_min, y_min, x_max, y_max))


def merge_nearby_blocks(layout, threshold=21):
    """
    Merges vertically adjacent or overlapping text blocks that likely belong to the same paragraph.

    Args:
        layout (lp.Layout): List of layoutparser blocks.
        threshold (int): Max vertical or horizontal distance to merge blocks.

    Returns:
        lp.Layout: Merged layout.
    """
    from layoutparser.elements import Rectangle, TextBlock

    merged = []
    used = set()

    for i, block1 in enumerate(layout):
        if i in used or block1.type != 'Text':
            continue

        merged_block = block1
        for j, block2 in enumerate(layout):
            if j in used or i == j or block2.type != 'Text':
                continue

            # Check vertical overlap and similar x alignment
            b1 = merged_block.block
            b2 = block2.block
            y_dist = abs(b1.y_2 - b2.y_1)
            x_diff = abs(b1.x_1 - b2.x_1)

            # If blocks are close and aligned
            if y_dist < threshold and x_diff < threshold:
                new_box = lp.elements.Rectangle(
                    x_1=min(b1.x_1, b2.x_1),
                    y_1=min(b1.y_1, b2.y_1),
                    x_2=max(b1.x_2, b2.x_2),
                    y_2=max(b1.y_2, b2.y_2)
                )
                merged_block = lp.TextBlock(new_box, type='Text', score=1.0)
                used.add(j)

        merged.append(merged_block)
        used.add(i)

    return lp.Layout(merged)

def detect_extract(pdf_path, output_dir):
    doc = fitz.open(pdf_path)
    os.makedirs(output_dir, exist_ok=True)
    # Load layout model once
    base_model_dir = Path.home() / "Projects" / "Auto_Moduler_clean"/ "src"/ "models"
    model = lp.Detectron2LayoutModel(
        config_path=str(base_model_dir / "config.yml"),
        model_path=str(base_model_dir / "model.pth"),
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
    )

    all_blocks = []

    for page_num in range(19, 23):  # You can generalize this to range(len(doc)) if needed
        page = doc[page_num]
        pix = page.get_pixmap(dpi=600)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if img_array.shape[2] == 4:  # Convert RGBA to RGB if needed
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)

        layout = model.detect(img_array)
        print("Page: ", page_num, "  Blocks: ", len(layout))
        #layout = merge_nearby_blocks(layout)
        # Convert to PIL for further processing
        pil_image = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        #print("Page: ", page_num, "  Blocks: ", len(layout))
        # Refine labels
        for i, block in enumerate(layout):
            block_label = f" page_{page_num} - block{i}"
            type_original = block.type
            block.type = classify_by_regex(pil_image, block, block.type, block_label)
            print("Page: ", page_num, " Block_number: ", i, ", Type_o:",type_original,", Type: ", block.type)
        if len(layout) != 0:
            # Save white-out image
            cleaned_img = white_out_regions(pil_image.copy(), layout)
            cleaned_img.save(os.path.join(output_dir, f"page_{page_num}_cleaned.png"))

        # Extract text blocks
        blocks = extract_text_blocks(pil_image, layout, page_num)
        all_blocks.extend(blocks)

    # Save all as JSON
    with open(os.path.join(output_dir, "rag_chunks.json"), "w", encoding="utf-8") as f:
        json.dump(all_blocks, f, indent=2, ensure_ascii=False)

    print(f"Finished processing {pdf_path}")


def main():
    for filename in os.listdir(INPUT_DIR):
        if not filename.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(INPUT_DIR, filename)

    result = detect_extract(pdf_path, OUTPUT_DIR)
    print(result)

# --- CONFIG ---
INPUT_DIR = "../Knowledge/pdfss"
OUTPUT_DIR = "../out/text_chunks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    main()