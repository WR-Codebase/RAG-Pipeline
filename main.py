import os
import logging
import re
import nltk
from transformers import AutoTokenizer

# Ensure the required nltk data is downloaded
nltk.download('punkt')

logging.basicConfig(level=logging.DEBUG)

# Define patterns for headers
header_patterns = {
    'header1': re.compile(r'^[#]+\s*(.+)$'),  # Matches headers starting with one or more #
    'header2': re.compile(r'(.+)\n(===+|---+)$', re.MULTILINE)  # Matches headers underlined with === or ---
}

logging.debug("Header patterns defined: %s", header_patterns)

# Directory containing markdown files
DOCUMENTS_DIR = './documents'
CHUNKS_DIR = './chunks'

# Create chunks directory if it doesn't exist
os.makedirs(CHUNKS_DIR, exist_ok=True)

# Load the Hugging Face tokenizer
model_name = "chentong00/propositionizer-wiki-flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to load markdown files
def load_markdown_files(directory):
    markdown_files = {}
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                markdown_files[filename] = content
                logging.debug(f"Loaded file: {filename}")
    return markdown_files

# Function to split text into sections by headers
def split_by_headers(text):
    lines = text.split('\n')
    sections = []
    current_section = []

    i = 0
    while i < len(lines):
        line = lines[i]
        next_line = lines[i + 1] if i + 1 < len(lines) else None
        if is_header(line, next_line):
            logging.debug(f"Found header: {line}")
            if current_section:
                sections.append('\n'.join(current_section))
                current_section = []
            current_section.append(line)
            if next_line and (next_line.strip() == '===' or next_line.strip() == '---'):
                current_section.append(next_line)
                i += 1  # Skip processing the next line as it's already included
        else:
            current_section.append(line)
        i += 1

    if current_section:
        sections.append('\n'.join(current_section))

    logging.debug(f"Total sections created: {len(sections)}")
    return sections

# Function to determine if a line is a header
def is_header(line, next_line):
    if header_patterns['header1'].match(line):
        return True
    if next_line and header_patterns['header2'].match(f"{line}\n{next_line}"):
        return True
    return False

# Function to log headers in a markdown file
def log_headers(content, filename):
    lines = content.split('\n')
    for i, line in enumerate(lines):
        next_line = lines[i + 1] if i + 1 < len(lines) else ''
        if is_header(line, next_line):
            logging.info(f"Header found in {filename}: {line.strip()}")

# Function to perform semantic chunking using transformers
def semantic_chunking(section):
    lines = section.split('\n')
    chunks = []
    current_chunk = []
    current_chunk_length = 0
    max_length = 256  # Max tokens for T5 model

    for i, line in enumerate(lines):
        if '|' in line:
            current_chunk.append(line)
            current_chunk_length += len(tokenizer.tokenize(line))
        elif is_header(line, None):
            if line.strip() in {'===', '---'} and current_chunk:
                current_chunk.append(line)
                current_chunk_length += len(tokenizer.tokenize(line))
            else:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_chunk_length = 0

                current_chunk.append(line)
                current_chunk_length += len(tokenizer.tokenize(line))

                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if not is_header(next_line, None):
                        current_chunk.append(next_line)
                        current_chunk_length += len(tokenizer.tokenize(next_line))
                        i += 1

            if current_chunk_length > max_length:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_chunk_length = 0
        else:
            if current_chunk_length + len(tokenizer.tokenize(line)) > max_length:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_chunk_length = len(tokenizer.tokenize(line))
            else:
                current_chunk.append(line)
                current_chunk_length += len(tokenizer.tokenize(line))

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    logging.info(f"Total chunks created: {len(chunks)}")
    return chunks

# Load markdown files
markdown_files = load_markdown_files(DOCUMENTS_DIR)

# Log headers and split into chunks for each markdown file
for filename, content in markdown_files.items():
    log_headers(content, filename)
    sections = split_by_headers(content)
    for idx, section in enumerate(sections):
        semantic_chunks = semantic_chunking(section)
        for chunk_idx, chunk in enumerate(semantic_chunks):
            chunk_filename = os.path.join(CHUNKS_DIR, f"{filename}_chunk_{idx + 1}_{chunk_idx + 1}.md")
            with open(chunk_filename, 'w', encoding='utf-8') as chunk_file:
                chunk_file.write(chunk)
            logging.info(f"Chunk created: {chunk_filename}")
