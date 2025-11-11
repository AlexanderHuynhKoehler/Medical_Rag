import json
import os
from pathlib import Path



file_path ='Json_files/Anxiety Disorder.json'



def create_chunks_from_json(file_path):
    """
    Takes a single JSON file and creates chunks with document and section context.
    
    Args:
        file_path: Path to a single JSON file
    
    Returns:
        list: List of chunk strings with context prepended
    """
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get document name from filename
    doc_name = Path(file_path).stem
    
    chunks = []
    
    for section in data:
        section_name = section['section']
        
        # Combine all content in this section
        section_text = '\n'.join(section['content'])
        
        # Prepend context: Document name and section name
        text_with_context = f"文档: {doc_name}\n章节: {section_name}\n\n{section_text}"
        chunk = {
            'text': text_with_context,
            'document': doc_name,
            'section': section_name
        }
        chunks.append(chunk)
    
    return chunks


def create_all_chunks(json_folder='Json_files'):
    """
    Loop through all JSON files in folder and create chunks from each.
    
    Args:
        json_folder: Path to folder containing JSON files
    
    Returns:
        list: All chunks from all files
    """
    from pathlib import Path
    
    all_chunks = []
    json_folder = Path(json_folder)
    
    # Get all JSON file paths
    json_files = list(json_folder.glob('*.json'))
    
    print(f"Found {len(json_files)} files")
    
    # Loop through each file
    for file_path in json_files:
        print(f"Processing: {file_path.name}")
        chunks = create_chunks_from_json(file_path)
        all_chunks.extend(chunks)  # Add to growing list
    
    print(f"\nTotal chunks created: {len(all_chunks)}")
    return all_chunks


import pickle

# Save chunks
def save_chunks(chunks, filename='chunks.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"Saved {len(chunks)} chunks to {filename}")

# Load chunks
def load_chunks(filename='chunks.pkl'):
    with open(filename, 'rb') as f:
        chunks = pickle.load(f)
    print(f"Loaded {len(chunks)} chunks from {filename}")
    return chunks

all_chunks = create_all_chunks()
save_chunks(all_chunks)

