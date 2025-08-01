#!/usr/bin/env python3
import json
import os
import re

def clean_filename(title):
    return re.sub(r'\s+', '_', re.sub(r'[^\w\s-]', '', title)).lower()

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'wikipedia_etru_content.json')
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    output_dir = os.path.join(script_dir, 'etruscan_texts')
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(output_dir):
        if file.endswith('.txt'):
            os.remove(os.path.join(output_dir, file))

    for title, content in data.items():
        if not content or not content.strip():
            continue
        filename = clean_filename(title) + '.txt'
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# {title}\n\n{content.strip()}")

if __name__ == "__main__":
    main()