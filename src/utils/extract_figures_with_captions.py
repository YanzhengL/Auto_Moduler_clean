import fitz  # PyMuPDF
import os
import re
from pathlib import Path
import json



def search_boundary_above(caption_rect, page, step=10, window=21, previous_caption_rect=None):
    """
    Search upward from caption_rect. Stop when:
      - A horizontal strip (window height) contains >14 words → return 1 step before that.
      - The area overlaps previous_caption_rect → return 1 step below that.

    Args:
        caption_rect (fitz.Rect): Current caption bounding box.
        page (fitz.Page): PyMuPDF page object.
        step (int): Vertical step size (how far to move up each loop).
        window (int): Height of the area to check for word density.
        previous_caption_rect (fitz.Rect or None): To stop before overlapping previous caption.

    Returns:
        fitz.Rect or None: Likely bounding box of figure above the caption.
    """
    y_top = caption_rect.y0
    y_p = previous_caption_rect.y1 if previous_caption_rect is not None else 50.0
    #print("Find Central Figure, y_top: ", y_top, " y_p: ", y_p)
    while y_top - step >= y_p:
        #print("Find Central Figure and expand up-ward")
        current_top = y_top - step
        current_rect = fitz.Rect(
            30,  # left margin
            current_top - window,
            page.rect.x1 - 50,  # right margin
            current_top
        )

        # Stop if intersects previous caption
        if previous_caption_rect and current_rect.intersects(previous_caption_rect):
            print("Find Central Figure and muti-image in the page")
            return fitz.Rect(
                previous_caption_rect.x0,
                previous_caption_rect.y1,
                previous_caption_rect.x1,
                caption_rect.y1,
            )

        words_in_area = page.get_text("words", clip=current_rect)
        """ Extract just the words (5th item in each tuple)
        word_texts = [w[4] for w in words_in_area]
        # Join them into a single string
        joined_text = " ".join(word_texts)
        print("Text in area:", joined_text)
        print("Length of words:", len(words_in_area))"""

        if len(words_in_area) > 18:
            return fitz.Rect(
                30,
                current_top,
                page.rect.x1 - 50,  # right margin
                caption_rect.y1 + 5,
            )
        y_top = current_top

    if y_top - step < y_p and previous_caption_rect is not None:
        y_top = previous_caption_rect.y0
    return fitz.Rect(30, y_top, page.rect.x1 - 50,  caption_rect.y1 + 5 )

def is_central_caption(caption_rect, page_rect, tolerance=20):
    caption_center_x = (caption_rect.x0 + caption_rect.x1) / 2
    page_center_x = (page_rect.x0 + page_rect.x1) / 2
    return abs(caption_center_x - page_center_x) <= tolerance

def get_caption_side(caption_rect: fitz.Rect, page_rect: fitz.Rect) -> str:
    """
    Determine whether the caption is on the left or right side of the page.

    Args:
        caption_rect (fitz.Rect): Bounding box of the caption.
        page_rect (fitz.Rect): Full page bounding box.

    Returns:
        str: "left" if caption is on the left side, "right" if on the right side.
    """
    caption_center_x = (caption_rect.x0 + caption_rect.x1) / 2
    page_center_x = (page_rect.x0 + page_rect.x1) / 2

    return "left" if caption_center_x < page_center_x else "right"

def expand_caption_boundary(caption_rect: fitz.Rect, page,  step: float = 10.0, window: float = 30.0, previous_fig_rect=None) -> fitz.Rect:
    """
    Expand the caption_rect based on which side it's on.

    - If on the left: expand up and down only.
    - If on the right: expand left, up, and down.

    Args:
        caption_rect (fitz.Rect): The original caption rectangle.
        page_rect (fitz.Rect): The full page rectangle.
        side (str): Either "left" or "right".
        step (float): Step size for expansion.
        window (float): Total amount to expand (centered).

    Returns:
        fitz.Rect: The expanded bounding box.
    """
    page_rect = page.rect
    side = get_caption_side(caption_rect, page_rect)
    y_pre = previous_fig_rect.y0 if previous_fig_rect is not None else page_rect.y0 + 50.0
    y_top = caption_rect.y0
    y_bottom = caption_rect.y1
    #print("Figure not in the central, side is: ", side)
    if side == "left":
        """ Expand to up and down, find the boundary of the Figure"""

        while y_top - step >= y_pre:
            current_top = y_top - step
            if previous_fig_rect is not None:
                #print("previous_fig_rect.y0 - ", previous_fig_rect.y0, ", ================= y_top - ", y_top)
                y_top = previous_fig_rect.y0
                break
            current_rect = fitz.Rect(
                30,  # left margin
                current_top - window,
                page.rect.x1 - 50,  # right margin
                current_top
            )
            # Stop if intersects previous caption

            words_in_area = page.get_text("words", clip=current_rect)
            """ Extract just the words (5th item in each tuple)
            word_texts = [w[4] for w in words_in_area]
            # Join them into a single string
            joined_text = " ".join(word_texts)
            print("Text in area:", joined_text)
            print("Length of words:", len(words_in_area))"""

            if len(words_in_area) > 18:
                break

            y_top = current_top

        while y_bottom + step <= page_rect.y1 - 60.0:
            current_bottom = y_bottom + step
            current_rect = fitz.Rect(
                30,  # left margin
                current_bottom ,
                page.rect.x1 - 50,  # right margin
                current_bottom + window
            )
            words_in_area = page.get_text("words", clip=current_rect)
            """ Extract just the words (5th item in each tuple)
            word_texts = [w[4] for w in words_in_area]
            # Join them into a single string
            joined_text = " ".join(word_texts)
            print("Text in area:", joined_text)
            print("Length of words:", len(words_in_area))"""

            if len(words_in_area) > 18:
                break
            y_bottom = current_bottom

        return fitz.Rect(page_rect.x0 + 30, y_top, page.rect.x1 - 50, y_bottom)

    elif side == "right":
        """ Expand to up, down, and left to find the boundary of the Figure"""
        while y_top - step >= y_pre:
            current_top = y_top - step
            current_rect = fitz.Rect(
                caption_rect.x0,  # left margin
                current_top - (window * 2),
                page.rect.x1 - 50,  # right margin
                current_top
            )
            # Stop if intersects previous caption
            if previous_fig_rect and current_rect.intersects(previous_fig_rect):
                break
            words_in_area = page.get_text("words", clip=current_rect)
            """ Extract just the words (5th item in each tuple)
            word_texts = [w[4] for w in words_in_area]
            # Join them into a single string
            joined_text = " ".join(word_texts)
            print("Text in area:", joined_text)
            print("Length of words:", len(words_in_area))"""

            if len(words_in_area) > 18:
                break

            y_top = current_top

        while y_bottom + step <= page_rect.y1 - 60.0:
            current_bottom = y_bottom + step
            current_rect = fitz.Rect(
                caption_rect.x0,  # left margin
                current_bottom,
                page.rect.x1 - 50,  # right margin
                current_bottom + (window*2)
            )
            words_in_area = page.get_text("words", clip=current_rect)
            """ Extract just the words (5th item in each tuple)
            word_texts = [w[4] for w in words_in_area]
            # Join them into a single string
            joined_text = " ".join(word_texts)
            print("Text in area:", joined_text)
            print("Length of words:", len(words_in_area))"""

            if len(words_in_area) > 18:
                break
            y_bottom = current_bottom

        x_left = caption_rect.x1
        while x_left - step >= page_rect.x0+30:
            current_left = x_left - step
            current_rect = fitz.Rect(
                current_left - window,  # left margin
                y_top,
                current_left,  # right margin
                y_bottom
            )
            words_in_area = page.get_text("words", clip=current_rect)
            """ Extract just the words (5th item in each tuple)
            word_texts = [w[4] for w in words_in_area]
            # Join them into a single string
            joined_text = " ".join(word_texts)
            print("Text in area:", joined_text)
            print("Length of words:", len(words_in_area))"""

            if len(words_in_area) > 18:
                break
            x_left = current_left

        return fitz.Rect(x_left, y_top, page.rect.x1 - 50, y_bottom)

    else:
        raise ValueError("Side must be 'left' or 'right'")

def extract_figures_captions_bybbx(pdf_path, output_folder="bold_figures_smart"):
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    extracted_data = []

    for page_num in range(98,101):#range(min(100, len(doc))):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        page_rect = page.rect
        prev_caption = None  # store the previous block

        for i, block in enumerate(blocks):
            block_text = ""
            block_rect = fitz.Rect(0, 0, 0, 0)  # Start with empty rect
            found_figure = False

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    span_text = span.get("text", "").strip()
                    if re.search(r'Fig\.\s*\d+', span_text, re.DOTALL):
                        print("Match found!!!   span_text: ", span_text)
                        found_figure = True
                    block_text += span_text + " "
                    span_rect = fitz.Rect(span["bbox"])
                    if block_rect.is_empty:
                        block_rect = span_rect
                    else:
                        block_rect |= span_rect  # Union with existing block rect

            if found_figure:
                caption_text = block_text.strip()
                print("**** Found figure in page ", page_num, " - ", caption_text)
                caption_rect = block_rect
                #print("caption height: ", caption_rect.y1 - caption_rect.y0)
                # Simple fixed region above caption
                is_above = is_central_caption(caption_rect, page_rect)
                if is_above:
                    #print("Find Figure in the central, Skip................")
                    above_boundary_rect = search_boundary_above(caption_rect, page,4.5, 19, prev_caption)
                    figure_rect = fitz.Rect(
                        page_rect.x0 + 30,
                        above_boundary_rect.y0,
                        page_rect.x1 - 40,
                        min(caption_rect.y1 + 5, page_rect.y1)
                    )
                else:
                    figure_rect = expand_caption_boundary(caption_rect, page, 4.5, 19, prev_caption)

                # Save the image
                pix = page.get_pixmap(clip=figure_rect, dpi=200)
                safe_caption = "".join(c if c.isalnum() or c == '_' else '_' for c in caption_text)[:50]
                image_filename = f"page{page_num + 1}_{safe_caption}.png"
                image_path = os.path.join(output_folder, image_filename)
                pix.save(image_path)

                print(f"[✓] Saved : {caption_text} (Page {page_num + 1})")

                extracted_data.append({
                    "page": page_num + 1,
                    "caption": caption_text,
                    "image_path": image_path,
                    "figure description": ""
                })

                prev_caption = figure_rect  # update for next iteration

    doc.close()
    return extracted_data

def blocks_overlap_vertically(block1, block2):
    """
    Check if two blocks overlap vertically.

    Each block's bbox is a list/tuple: [x0, y0, x1, y1]
    where y0 is top and y1 is bottom (in PDF coordinate system, y increases downward).

    Returns True if there is any vertical overlap, False otherwise.
    """
    y0_1, y1_1 = block1['bbox'][1], block1['bbox'][3]
    y0_2, y1_2 = block2['bbox'][1], block2['bbox'][3]

    # Two vertical ranges [y0_1, y1_1] and [y0_2, y1_2] overlap if:
    return not (y1_1 < y0_2 or y1_2 < y0_1)

def merge_block_rects(blocks):
    """
    Given a list of blocks (each with a 'bbox': [x0, y0, x1, y1]),
    return a merged bounding box that covers all the blocks.

    Parameters:
    - blocks: list of block dicts (each with 'bbox' key)

    Returns:
    - A list: [x0_min, y0_min, x1_max, y1_max]
    """
    if not blocks:
        return None
    x0_min = min(block['bbox'][0] for block in blocks)
    y0_min = min(block['bbox'][1] for block in blocks)
    x1_max = max(block['bbox'][2] for block in blocks)
    y1_max = max(block['bbox'][3] for block in blocks)

    return [x0_min, y0_min, x1_max, y1_max]

def re_construct_block(page):
    """
    Analyze and label text blocks on a PDF page. Merge blocks that belongs to a same equation or same Figure.

    Parameters:
    - page: fitz.Page object
    - is_figure_caption: function(text:str) -> bool
    - is_equation: function(text:str) -> bool

    Returns:
    - List of dicts, each dict contains:
      {
        "block_index": int,
        "word_count": int,
        "label": str or None,
        "text": str
      }
    """
    is_figure, is_equation = False, False
    blocks = page.get_text("dict")["blocks"]
    merged_blocks = [
        {
            "text": "",  # type: str
            "bbox": [],  # type: list
            "type": 0  # type: int
            # "blocks": []  # (optional) original block references
        },
        # ... more merged blocks
    ]

    for i, block in enumerate(blocks):
        block_type = block.get("type", "?")
        bbox = block.get("bbox", "?")
        num_lines = len(block.get("lines", []))

        block_summary = f"[Block {i}] Type: {block_type} | BBox: {bbox} | Lines: {num_lines}"
        #print(block_summary)
        block_text = ""
        if i == 0:  # ignore the first block in each page, that is just headers
            continue
        ieq = 0

        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                span_text = span.get("text", "")
                line_text += span_text + " "
                is_equation = re.fullmatch(r'\(\d+\.\d+\)', span_text.strip())
                if is_equation:
                    while blocks_overlap_vertically(block,blocks[i-ieq-1]):
                        #print("Merging block: ", i - ieq)
                        ieq += 1
                    #print("Found Eq, span_text: ", span_text, " Number of blocked for this equation:", ieq)
            #print("----------- ------------------- line content: ", line_text)
            block_text += line_text.strip()+ " "

        if ieq >0 :
            merged_text = ""
            merged_rect = merge_block_rects(merged_blocks[i-ieq: i+1])
            for subb in merged_blocks[i-ieq: i+2]:
                merged_text += subb.get("text", "")
                print("------------merging.....subb.text: ",merged_text)
                merged_blocks.pop()
            merged_text += block_text
            last_block = merged_blocks.pop()
            last_block['bbox'] = merged_rect
            last_block['text'] = merged_text
            last_block['type'] = 2 # 0-text 1-figure  2-equation
            #print(last_block.keys)
            merged_blocks.append(last_block)
            continue

        new_block = {
            "text": block_text,
            "type": 2 if is_equation else 1 if is_figure else 0,
            "bbox": block["bbox"]  # or block["rect"] if your structure uses that
        }
        merged_blocks.append(new_block)
    return merged_blocks

def extract_figures_text_with_captions(pdf_path, output_folder="extracted_figures"):
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)

    for page_num in range(35,37):  # Max 21 pages
        print('\n',"===================== new page =====================")
        page = doc[page_num]
        blocks = re_construct_block(page)
        for i, block in enumerate(blocks):
            block_summary = f"[Block {i}] Type: {block.get('type')} "
            print(block_summary)
            print(" - - - block text", block['text'])
    return ""

def extract_figures_text_with_captions_old(pdf_path, output_folder="extracted_figures"):
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    page_data = []

    for page_num in range(35,37):  # Max 21 pages
        page = doc[page_num]
        #page_text = page.get_text()
        #figures = []
        #print("Page Number: ========  ", page_num, "\n")
        #print(page_text)
        blocks = page.get_text("dict")["blocks"]
        page_rect = page.rect
        prev_caption = None  # store the previous block

        for i, block in enumerate(blocks):
            block_text = ""
            block_rect = fitz.Rect(0, 0, 0, 0)  # Start with empty rect
            found_figure = False

            block_type = block.get("type", "?")
            bbox = block.get("bbox", "?")
            num_lines = len(block.get("lines", []))

            block_summary = f"[Block {i}] Type: {block_type} | BBox: {bbox} | Lines: {num_lines}"

            # Preview first few words if it's text
            if block_type == 0 and num_lines > 0:
                preview_text = ""
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        preview_text += span.get("text", "") + " "
                preview_text = preview_text.strip().replace('\n', ' ')
                if len(preview_text) > 60:
                    preview_text = preview_text[:60] + "..."
                block_summary += f" | Text Preview: \"{preview_text}\""

            print(block_summary)

            for line in block.get("lines", []):
                line_text = ""
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    print(" - - - - - - -  -   -  - span text: ", span_text, "~~~~~~~~~")
                    """  --- flags = span.get("flags", 0)
                    flags & 1 → Italic
                    flags & 2 → Bold
                    flags & 4 → Serif font
                    flags & 8 → Monospace font
                    """
                    font = span.get("font", "")
                    if "Bold" in font or "bold" in font:
                        is_bold = True
                    else:
                        is_bold = False
                    if  re.search(r'Fig\.\W*', span_text) and is_bold:
                        #print("BOLD Match found!!!   span_text:", repr(span_text), " ~~~  flag_bold: ", is_bold)
                        found_figure = True
                    line_text += span_text + ""


                    span_rect = fitz.Rect(span["bbox"])
                    if block_rect.is_empty:
                        block_rect = span_rect
                    else:
                        block_rect |= span_rect  # Union with existing block rect
                print(" - - - - - - - line_text: ", line_text)
                block_text += line_text + " "
            print(" [-] block_text: ", block_text)
            print('\n')

            if found_figure:
                caption_text = block_text.strip()
                #print("**** Found figure in page ", page_num, " - ", caption_text)
                caption_rect = block_rect
                # print("caption height: ", caption_rect.y1 - caption_rect.y0)
                # Simple fixed region above caption
                is_above = is_central_caption(caption_rect, page_rect)
                if is_above:
                    # print("Find Figure in the central, Skip................")
                    above_boundary_rect = search_boundary_above(caption_rect, page, 4.5, 19, prev_caption)
                    figure_rect = fitz.Rect(
                        page_rect.x0 + 30,
                        above_boundary_rect.y0,
                        page_rect.x1 - 40,
                        min(caption_rect.y1 + 5, page_rect.y1)
                    )
                else:
                    figure_rect = expand_caption_boundary(caption_rect, page, 4.5, 19, prev_caption)

                # Save the image
                pix = page.get_pixmap(clip=figure_rect, dpi=200)
                safe_caption = "".join(c if c.isalnum() or c == '_' else '_' for c in caption_text)[:50]
                image_filename = f"page{page_num + 1}_{safe_caption}.png"
                image_path = os.path.join(output_folder, image_filename)
                pix.save(image_path)

                print(f"[✓] Saved : {caption_text} (Page {page_num + 1})")

                page_data.append({
                    "page": page_num + 1,
                    "caption": caption_text,
                    "image_path": image_path,
                    "figure description": ""
                })

                prev_caption = figure_rect  # update for next iteration

    return page_data

def main():
    for filename in os.listdir(INPUT_DIR):
        if not filename.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(INPUT_DIR, filename)

        #result = extract_figures_captions_bybbx(pdf_path)
        result = extract_figures_text_with_captions(pdf_path)
        out_file = os.path.join(OUTPUT_DIR, f"{Path(filename).stem}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        if result:
            print("Processed", result[-1]["page"], "pages, --", len(result), "figures extracted")
        else:
            print("No figures extracted.")


# --- CONFIG ---
INPUT_DIR = "../Knowledge/pdfss"
OUTPUT_DIR = "../Knowledge/knowledge_chunks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    main()

