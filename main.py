import os
import logging
import re
import nltk
from transformers import AutoTokenizer

# Ensure the required nltk data is downloaded
nltk.download('punkt')

logging.basicConfig(level=logging.DEBUG)

# Define patterns for headers, code blocks, and tables
header_patterns = {
    'header1': re.compile(r'^[#]+\s*(.+)$'),  # Matches headers starting with one or more #
    'header2': re.compile(r'(.+)\n(===+|---+)$', re.MULTILINE)  # Matches headers underlined with === or ---
}
code_block_pattern = re.compile(r'```')
table_pattern = re.compile(r'\|')

logging.debug("Patterns defined: %s", header_patterns)

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

# Function to chunk by tables and code blocks
def chunk_by_code_blocks_and_tables(text):
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    in_code_block = False

    for line in lines:
        # Toggle in_code_block flag
        if code_block_pattern.match(line):
            in_code_block = not in_code_block
            current_chunk.append(line)
            continue

        if not in_code_block and table_pattern.search(line):
            in_table = True
        else:
            in_table = False

        if not in_code_block and not in_table and is_header(line, None):
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
            current_chunk.append(line)
        else:
            current_chunk.append(line)

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks

# Function to determine if a line is a header
def is_header(line, next_line):
    if header_patterns['header1'].match(line):
        return True
    if next_line and header_patterns['header2'].match(f"{line}\n{next_line}"):
        return True
    return False

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

# Function to merge header-only chunks
def merge_header_only_chunks(chunks):
    merged_chunks = []
    previous_chunk = None

    for chunk in chunks:
        lines = chunk.split('\n')
        if len(lines) == 1 and is_header(lines[0], None):
            if previous_chunk:
                previous_chunk += '\n' + chunk
            else:
                previous_chunk = chunk
        else:
            if previous_chunk:
                merged_chunks.append(previous_chunk)
                previous_chunk = None
            merged_chunks.append(chunk)

    if previous_chunk:
        merged_chunks.append(previous_chunk)

    return merged_chunks

# Function to perform semantic chunking using transformers
def semantic_chunking(section):
    lines = section.split('\n')
    chunks = []
    current_chunk = []
    current_chunk_length = 0
    max_length = 300  # Max tokens for T5 model

    for line in lines:
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

# Empty chunks directory
def empty_chunks_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            logging.error(f"Error deleting file {file_path}: {e}")

# Process each markdown file
def process_markdown_files():
    # Load markdown files
    markdown_files = load_markdown_files(DOCUMENTS_DIR)

    # Empty chunks directory before processing
    empty_chunks_directory(CHUNKS_DIR)

    for filename, content in markdown_files.items():
        logging.info(f"Processing file: {filename}")

        # Step 1: Chunk by tables and code blocks
        initial_chunks = chunk_by_code_blocks_and_tables(content)
        logging.debug(f"Initial chunks count: {len(initial_chunks)}")

        # Step 2: Chunk by headers within each non-table, non-code-block chunk
        final_chunks = []
        for chunk in initial_chunks:
            if code_block_pattern.search(chunk) or table_pattern.search(chunk):
                final_chunks.append(chunk)
            else:
                header_chunks = split_by_headers(chunk)
                final_chunks.extend(header_chunks)

        # Step 2.a: Merge header-only chunks
        final_chunks = merge_header_only_chunks(final_chunks)
        logging.debug(f"Final chunks count after merging headers: {len(final_chunks)}")

        # Step 3: Semantic chunking for non-table, non-code-block chunks
        for idx, chunk in enumerate(final_chunks):
            if not (code_block_pattern.search(chunk) or table_pattern.search(chunk)):
                semantic_chunks = semantic_chunking(chunk)
                for chunk_idx, semantic_chunk in enumerate(semantic_chunks):
                    chunk_filename = os.path.join(CHUNKS_DIR, f"{filename}_chunk_{idx + 1}_{chunk_idx + 1}.md")
                    with open(chunk_filename, 'w', encoding='utf-8') as chunk_file:
                        chunk_file.write(semantic_chunk)
                    logging.info(f"Semantic chunk created: {chunk_filename}")
            else:
                chunk_filename = os.path.join(CHUNKS_DIR, f"{filename}_chunk_{idx + 1}.md")
                with open(chunk_filename, 'w', encoding='utf-8') as chunk_file:
                    chunk_file.write(chunk)
                logging.info(f"Non-semantic chunk created: {chunk_filename}")

        logging.info(f"Total chunks created for {filename}: {len(final_chunks)}")

if __name__ == "__main__":
    process_markdown_files()
