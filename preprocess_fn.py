import re
import json
from typing import List, Dict

##################################### Pre-Processing Function #####################################

def preprocess_fn(file_path:str, num_space_fr=0.47, line_len_thresh=35):
    with open(file_path, 'r') as f:
        text = f.read()

    lines = text.split('\n')

    pre_processed_lines = []
    # for line in lines:

    #     if len(line) >= line_len_thresh:
        
    #         # Remove special characters
    #         line = re.sub(r'[^\w\s]', '', line)

    #         # Remove multiple spaces
    #         line = re.sub(r'\s+', ' ', line)

    #         # Remove leading and trailing spaces
    #         line = line.strip()

    #         if ''.join(line.split()).isdigit():
    #             continue

    #         pre_processed_lines.append(line)


    pre_processed_lines = []
    spacer_cnt = 0  # Initialize outside the loop

    for line in lines:
        if line and (line[0] == '<') and (line[-1] == '>'):  # Handle <...> lines
            spacer_cnt += 1
            if spacer_cnt % 3 == 1:
                pre_processed_lines.append(f'\n\n{line}')
            elif spacer_cnt % 3 == 2:
                pre_processed_lines.append(f'{line}')
            else:
                pre_processed_lines.append(f'{line}\n\n')
            continue

        # Skip short lines
        if len(line) <= line_len_thresh:
            continue

        # Remove special characters, multiple spaces, and trim the line
        line = re.sub(r'[^\w\s.]', '', line)  # Remove special characters
        line = re.sub(r'\s+', ' ', line)  # Remove multiple spaces
        line = line.strip()  # Trim leading/trailing spaces

        # Skip numeric lines
        if ''.join(line.split()).isdigit():
            continue

        # Append the processed line with consistent formatting
        pre_processed_lines.append(f'{line}')


    cleaned_lines = []
    for line in pre_processed_lines:

        char_count = 0
        num_count = 0
        sp_count = 0
    
        for char in line:
            if (char.isalpha()) and (char != ' '):
                char_count += 1

            elif char.isdigit():
             num_count += 1

            elif char.isspace():
                sp_count += 1

        if len(line) == 0:
            continue
    
        perc_num = (num_count / len(line))
        perc_sp = (sp_count / len(line))

        if (perc_num + perc_sp) >= num_space_fr:
            continue

        if ('Fig' in line) or ('Table' in line):
            continue

        cleaned_lines.append(line)

    

###################################### Chunking Functions #####################################

def chunk_text(text: str, chunk_size: int = 100) -> List[str]:
    """
    Splits text into chunks of specified size while maintaining word boundaries.
    """
    print(f"Chunking text with size: {chunk_size}")
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if sum(len(w) + 1 for w in current_chunk) + len(word) <= chunk_size:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    print(f"Created {len(chunks)} chunks.")
    return chunks

#===============================================================================================#

def process_file(file_path: str, chunk_size: int = 100) -> List[Dict]:
    """
    Processes the input file, chunks the text by page, and attaches metadata to each chunk.
    """
    print(f"Processing file: {file_path}")

    with open(file_path, 'r') as file:
        content = file.read()

    print(f"File content loaded. Length: {len(content)} characters")

    # Regular expression to match page-level metadata and text
    page_pattern = re.compile(
    r"<file_path\s*:\s*(.*?)>\s*<filename\s*:\s*(.*?)>\s*<page\s*:\s*(\d+)>\s*(.*?)(?=(<file_path|$))", 
    re.S
    )

    results = []

    matches = list(page_pattern.finditer(content))
    print(f"Total matches found: {len(matches)}")

    for match in page_pattern.finditer(content):
        file_path_meta = match.group(1).strip()
        filename_meta = match.group(2).strip()
        page_meta = int(match.group(3).strip())
        text = match.group(4).strip()

        print(f"Matched metadata - file_path: {file_path_meta}, filename: {filename_meta}, page: {page_meta}")
        print(f"Page text length: {len(text)}")

        # Chunk the text
        chunks = chunk_text(text, chunk_size=chunk_size)

        # Attach metadata to each chunk
        for chunk in chunks:
            results.append({
                "chunk_text": chunk,
                "file_path": file_path_meta,
                "filename": filename_meta,
                "page": page_meta
            })

    print(f"Total chunks created: {len(results)}")

    return results



