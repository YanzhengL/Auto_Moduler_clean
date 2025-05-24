import fitz  # PyMuPDF
import os
import re
from pathlib import Path
import json
import pytesseract
from PIL import Image
import cv2
from pytesseract import Output
import numpy as np




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
    y_top = caption_rect[1]
    y_p = previous_caption_rect[3] if previous_caption_rect is not None else 50.0
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
                caption_rect[1] + 5,
            )
        y_top = current_top

    if y_top - step < y_p and previous_caption_rect is not None:
        y_top = previous_caption_rect.y0
    return fitz.Rect(30, y_top, page.rect.x1 - 50,  caption_rect[3] + 5 )

def is_central_caption(caption_rect, page_rect, tolerance=20):
    caption_center_x = (caption_rect[0] + caption_rect[2]) / 2
    page_center_x = (page_rect[0] + page_rect[2]) / 2
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
    caption_center_x = (caption_rect[0] + caption_rect[2]) / 2
    page_center_x = (page_rect[0] + page_rect[2]) / 2

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
    y_pre = previous_fig_rect[1] if previous_fig_rect is not None else page_rect.y0 + 50.0
    y_top = caption_rect[1]
    y_bottom = caption_rect[3]
    #print("Figure not in the central, side is: ", side)
    if side == "left":
        """ Expand to up and down, find the boundary of the Figure"""

        while y_top - step >= y_pre:
            current_top = y_top - step
            if previous_fig_rect is not None:
                #print("previous_fig_rect.y0 - ", previous_fig_rect.y0, ", ================= y_top - ", y_top)
                y_top = previous_fig_rect[1]
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
                caption_rect[0],  # left margin
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
                caption_rect[0],  # left margin
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

        x_left = caption_rect[2]
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
    y0_1, y1_1 = block1[1], block1[3]
    y0_2, y1_2 = block2[1], block2[3]

    # Two vertical ranges [y0_1, y1_1] and [y0_2, y1_2] overlap if:
    return not (y1_1 < y0_2 or y1_2 < y0_1)

def blocks_overlap_vertically_or_horizontally(block1, block2):
    """
    Check if two blocks overlap vertically or horizontally.

    Each block's bbox is a list/tuple: [x0, y0, x1, y1]
    where:
    - x0 is left, x1 is right
    - y0 is top, y1 is bottom
    (PDF coordinate system: y increases downward)

    Returns True if there is any vertical or horizontal overlap, False otherwise.
    """
    x0_1, y0_1, x1_1, y1_1 = block1
    x0_2, y0_2, x1_2, y1_2 = block2

    vertical_overlap = not (y1_1 < y0_2 or y1_2 < y0_1)
    horizontal_overlap = not (x1_1 < x0_2 or x1_2 < x0_1)

    return vertical_overlap or horizontal_overlap


def is_vertically_related(block1, block2):
    y0_1, y1_1 = block1[1], block1[3]
    y0_2, y1_2 = block2[1], block2[3]

    height_1 = y1_1 - y0_1
    height_2 = y1_2 - y0_2

    # Check direct vertical overlap
    vertically_overlap = not (y1_1 < y0_2 or y1_2 < y0_1)

    # Check for small block and large block vertically close to the other
    #small_height_threshold = 7
    #large_height_threshold = 13.5
    close_proximity_threshold = 2

    #is_one_small = height_1 < small_height_threshold or height_2 < small_height_threshold
    #is_one_large = height_1 > large_height_threshold or height_2 > large_height_threshold

    #is_one_small =  height_2 < small_height_threshold
    #is_one_large =  height_2 > large_height_threshold
    vertically_close = abs(y0_2 - y1_1) < close_proximity_threshold or abs(y0_1 - y1_2) < close_proximity_threshold

    return vertically_overlap or vertically_close


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

def merge_bbox(bboxes):
    if not bboxes:
        print("bboxs list is empty, return None to emrged_rect")
        return None
    # Combine multiple bbox rectangles into one
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return fitz.Rect(x0, y0, x1, y1)

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

def extract_figures_text_with_captions(pdf_path, output_folder="out"):
    os.makedirs(f"{output_folder}/equations", exist_ok=True)
    os.makedirs(f"{output_folder}/figures", exist_ok=True)
    doc = fitz.open(pdf_path)
    none_text = []
    prev_caption = None
    for page_num in range(31, 40):  # Max 21 pages
        none_text.clear()
        page = doc[page_num]
        found_figure = False
        blocks = page.get_text("dict")["blocks"]
        equation_rects = []
        for i, block in enumerate(blocks):
            block_text = ""
            #print block information for debugging
            block_type = block.get("type", "?")
            bbox = block.get("bbox", "?")
            num_lines = len(block.get("lines", []))
            block_summary = f"[Block {i}] Type: {block_type} | BBox: {bbox} | Lines: {num_lines}"
            print(block_summary)

            for j, line in enumerate(block.get("lines", [])):
                print(" - - - -Line BBox: ", line["bbox"])
                print("-- line width: ", line['bbox'][3] - line['bbox'][1]," - - wmode: ", line['wmode'], " - - Dir: ", line['dir'])
                line_text = ""
                for q, span in enumerate(line.get("spans", [])):
                    span_text = span.get("text", "")
                    print(" - - - - - - -  -   -  - span text: ", span_text, "~~~~~~~~~")

                    # DETECT EQUATION AND PROCESS IT
                    if re.fullmatch(r'\((?:[1-9][0-9]?|99)\.(?:[1-9][0-9]?|99)\)', span_text.strip()):
                        eq_index = span_text.strip()
                        print("------ ----- --- Find EQ", eq_index)
                        equation_rects.clear()
                        # Merge vertically overlapping spans in the same line
                        qq = q
                        while qq >= 1 and is_vertically_related(span['bbox'], line["spans"][qq - 1]['bbox']):
                            equation_rects.append(line["spans"][qq - 1]["bbox"])
                            print("--------- merging span: ", qq - 1)
                            qq -= 1

                        if equation_rects:
                            equation_rects.append(span['bbox'])  # include current span
                            print("merging spans done....")
                            merged_rect = merge_bbox(equation_rects)
                        else:
                            print("The EQ is a standalone span, force merging at least 1 block previously")
                            merged_rect = block["lines"][j - 1]['bbox']

                        # Merge vertically overlapping lines in the block
                        jj = j
                        while jj >= 1 and is_vertically_related(merged_rect, block["lines"][jj - 1]['bbox']):
                            equation_rects.append(block["lines"][jj - 1]['bbox'])
                            print("--------- merging line: ", jj - 1)
                            jj -= 1

                        if equation_rects:
                            merged_rect = merge_bbox(equation_rects)
                        else:
                            print("The EQ is a standalone block, force merging at least 1 block previously")
                            merged_rect = blocks[i - 1]['bbox']

                        # Merge vertically overlapping blocks
                        ii = i
                        if ii >= 1 and is_vertically_related(merged_rect, blocks[ii - 1]['bbox']):
                            while ii >= 1 and is_vertically_related(merged_rect, blocks[ii - 1]['bbox']):
                                print("--------- merging block: ", ii - 1)
                                for l, extra_line in enumerate(blocks[ii - 1]['lines']):
                                    if is_vertically_related(merged_rect, extra_line['bbox']):
                                        equation_rects.append(extra_line['bbox'])
                                    else:
                                        break
                                merged_rect = merge_bbox(equation_rects)
                                ii -= 1

                        print("merging done, starting to save merged equation:", equation_rects)

                        # Final merging of all rectangles
                        merged_rect = merge_bbox(equation_rects)

                        if merged_rect is not None:
                            equation_rects.clear()
                            none_text.append(merged_rect)

                            # Slightly expand bounding box
                            merged_rect[0] -= 10
                            merged_rect[1] -= 2
                            merged_rect[2] += 10

                            image = page.get_pixmap(clip=merged_rect, dpi=300)
                            image_filename = f"equation{eq_index}_page{page_num}_block{i}_line{j}.png"
                            image_path = os.path.join(f"{output_folder}/equations", image_filename)
                            image.save(image_path)

                            img = Image.frombytes("RGB", [image.width, image.height], image.samples)
                            text = pytesseract.image_to_string(img)
                            print("----- equation text: ", text)


                            print("✅ Saved equation image to", image_path)
                        else:
                            print("⚠️ Warning: merged_rect is None. Skipping adjustment.")

                    # Detect FIGURE
                    if re.search(r'Fig\.\W*', span_text) and "bold" in span.get("font", "").lower():
                        # Collect overlapping spans for the image
                        print("Found figure")
                        found_figure = True

                    line_text += span_text + ""
                print(" - - - - - - - line_text: ", line_text)
                if found_figure:
                    caption_rect = line['bbox']
                    page_rect = page.rect
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
                            min(caption_rect[1] + 5, page_rect.y1)
                        )
                    else:
                        figure_rect = expand_caption_boundary(caption_rect, page, 4.5, 19, prev_caption)
                    prev_caption = caption_rect
                    # Save the image
                    pix = page.get_pixmap(clip=figure_rect, dpi=200)
                    safe_caption = "".join(c if c.isalnum() or c == '_' else '_' for c in line_text)[:50]
                    image_filename = f"page{page_num + 1}_{safe_caption}.png"
                    image_path = os.path.join(f"{output_folder}/figures", image_filename)
                    pix.save(image_path)



                    print(f"[✓] Saved : {line_text} (Page {page_num + 1})")
                    found_figure = False

                block_text += line_text + " "
            print(" [-] block_text: ", block_text)
            print('\n')
            # save the merged
            """new_page.insert_text(
            span["origin"],
            span["text"],
            fontsize=span["size"],
            fontname=span["font"],
            color=span.get("color", 0)
        )"""

        #another for loop for extracting text information
    return ""

def extract_content_bocr(pdf_path, output_folder="out"):
    os.makedirs(f"{output_folder}/equations", exist_ok=True)
    os.makedirs(f"{output_folder}/figures", exist_ok=True)
    doc = fitz.open(pdf_path)
    none_text = []
    for page_num in range(20, 23):  # Max 21 pages
        none_text.clear()
        """pix = page.get_pixmap(clip=page.rect, dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img)
        print("----- equation text: ", text)"""
        page = doc[page_num]
        pix = page.get_pixmap(dpi=300)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if image.shape[2] == 4:  # remove alpha channel if present
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # OCR with bounding boxes
        data = pytesseract.image_to_data(image, output_type=Output.DICT)

        # Create a blank mask for low-density regions
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

        n_boxes = len(data['text'])
        boxes = []
        for i in range(n_boxes):
            if int(data['conf'][i]) > 30:
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                boxes.append((x, y, w, h))
                # Draw text bounding box for reference
                cv2.rectangle(mask, (x, y), (x + w, y + h), 0, -1)  # draw text as black

        # Invert mask to find blank regions
        mask_inv = 255 - mask

        # Find contours of blank regions (possible figures/equations)
        contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print("Length of contours: ", len(contours))
        figure_index = 1
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > 20000:  # adjust threshold based on your layout
                roi = image[y:y + h, x:x + w]

                # Try to extract caption/equation number from OCR in nearby area
                caption_text = ""
                for i in range(n_boxes):
                    tx, ty, tw, th = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    if abs(ty - (y + h)) < 50 and tx > x - 50 and tx < x + w + 50:
                        caption_text += data['text'][i] + " "

                print(f"Page {page_num + 1} - Found region with caption/equation: {caption_text.strip()}")

                # Save the region and white it out on original image
                cv2.imwrite(os.path.join(output_folder, f"page{page_num + 1}_region{figure_index}.png"), roi)
                # Ensure the image is writable
                image = image.copy()
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)  # white out
                figure_index += 1

        # Save the cleaned image
        cv2.imwrite(os.path.join(output_folder, f"page{page_num + 1}_cleaned.png"), image)

    print("Finished processing.")
    return ""

def main():
    for filename in os.listdir(INPUT_DIR):
        if not filename.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(INPUT_DIR, filename)

        #result = extract_figures_captions_bybbx(pdf_path)
        #result = extract_figures_text_with_captions(pdf_path)
        result = extract_content_bocr(pdf_path)
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

