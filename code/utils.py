import json
import re
import os

def read_text(path): 
    with open(path, 'r', encoding='utf-8') as f: 
        content = f.read()
    return str(content)

def save_text(s, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(s)
        
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


def ensure_directory_exists(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def get_score(response):
    # default 999
    if not response or not isinstance(response, str):
        return 999

    #r'(?:Score|score)\s*:\s*(\d+)(?=\s*\n\s*(?:Reason|reason):)'
    patterns = [
        r'(?:Score|score)\s*:\s*(\d+)',
        r'(?:Score|score) of\s*(\d+)',
        r'Assign a Score:?\s*(?:\n\s*)?Score\s+(\d+):.*'
        # ,
        # r'Scoring Decision:\s*\n\s*Score:\s*(\d+)',
        # r'Score Determination:\s*\n\s*Score:\s*(\d+)',
        # r'Score Assignment:\s*\n\s*score:\s*(\d+)',
        # r'Score Assignment\s*\n\s*score:\s*(\d+)',
        # r'Assign a Score\s*\n\s*(?:Score|score):\s*(\d+)',
        # r'Assign a Score:\s*(?:\n\s*)?(?:Score|score):\s*(\d+)',
        # r'Assign Score:\s*\n\s*(?:Score|score):\s*(\d+)',
        # r'Scoring:\s*\n\s*Score:\s*(\d+)'
    ]
    # try patterns
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                continue

    # if not match, return 999
    return 999
