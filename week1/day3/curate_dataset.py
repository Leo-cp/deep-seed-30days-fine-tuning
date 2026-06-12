#!/usr/bin/env python3
"""
DeepSeed - Week 1 Day 3: Dataset Curation & Formatting
This script implements a complete data engineering pipeline:
1. Syntax Cleaning (stripping HTML, consolidating whitespace).
2. Alpaca Format mapping.
3. Length Outlier Filtering.
4. Jaccard Similarity Deduplication (near-duplicate detection using word shingles).
5. Category/Class Balancing.
6. Exporting to a clean JSONL dataset.
"""

import json
import os
import re

# ANSI colors for styling
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
END = "\033[0m"

# 1. Define Raw "Dirty" Mock Dataset
# This dataset represents real-world issues: HTML tags, near-duplicates, outliers, and class imbalance.
RAW_DATA = [
    # Category: Coding (Dominant class)
    {
        "category": "Coding",
        "raw_instruction": "<p>Write a python function to add two numbers.</p>",
        "raw_response": "def add(a, b):\n    return a + b"
    },
    {
        "category": "Coding",
        "raw_instruction": "Write a Python function to add two numbers!  ",
        "raw_response": "def add(x, y):\n    return x + y"
    }, # Near-duplicate (same goal, slightly different phrasing and variable names)
    {
        "category": "Coding",
        "raw_instruction": "Write a Python function to add two numbers.",
        "raw_response": "def add_nums(num1, num2):\n    return num1 + num2"
    }, # Near-duplicate
    {
        "category": "Coding",
        "raw_instruction": "Create a list comprehension in python that filters even numbers.",
        "raw_response": "[x for x in my_list if x % 2 == 0]"
    },
    {
        "category": "Coding",
        "raw_instruction": "How do you define a class in Python?",
        "raw_response": "class MyClass:\n    def __init__(self):\n        pass"
    },
    {
        "category": "Coding",
        "raw_instruction": "Write a Python loop.",
        "raw_response": "for i in range(10):\n    print(i)"
    },
    {
        "category": "Coding",
        "raw_instruction": "python", 
        "raw_response": "print('hello')"
    }, # Too short (Instruction length outlier)
    {
        "category": "Coding",
        "raw_instruction": "Write a quicksort function in python.",
        "raw_response": "ok"
    }, # Too short (Response length outlier)

    # Category: Math (Under-represented class)
    {
        "category": "Math",
        "raw_instruction": "Solve for x: 2x + 5 = 15.",
        "raw_response": "2x = 10\nx = 5"
    },

    # Category: Translation (Under-represented class)
    {
        "category": "Translation",
        "raw_instruction": "Translate the sentence 'Hello world' to Spanish.",
        "raw_response": "Hola mundo"
    }
]

def clean_syntax(text):
    """Strips HTML tags and normalizes spacing/newlines."""
    if not text:
        return ""
    # Strip HTML tags
    clean = re.sub(r'<[^>]*>', '', text)
    # Replace multiple spaces with single space
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean

def get_word_shingles(text, k=2):
    """Tokenizes text and returns overlapping k-word shingles."""
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    if len(words) < k:
        return set(words)
    shingles = set()
    for i in range(len(words) - k + 1):
        shingle = " ".join(words[i:i+k])
        shingles.add(shingle)
    return shingles

def compute_jaccard(set_a, set_b):
    """Computes Jaccard Similarity coefficient between two sets."""
    union_size = len(set_a.union(set_b))
    if union_size == 0:
        return 0
    return len(set_a.intersection(set_b)) / union_size

def run_pipeline():
    print("======================================================================")
    print("Running Dataset Curation Pipeline")
    print("======================================================================\n")
    print(f"Initial raw dataset size: {BOLD}{len(RAW_DATA)}{END} examples.\n")

    # --- STAGE 1: SYNTAX CLEANING & ALPACA FORMATTING ---
    print(f"{BLUE}[Stage 1] Cleaning syntax & formatting to Alpaca schema...{END}")
    stage1_data = []
    for idx, item in enumerate(RAW_DATA):
        cleaned_instruction = clean_syntax(item["raw_instruction"])
        cleaned_response = clean_syntax(item["raw_response"])
        
        formatted = {
            "id": idx + 1,
            "category": item["category"],
            "instruction": cleaned_instruction,
            "input": "", # Optional input context is blank for these examples
            "output": cleaned_response
        }
        stage1_data.append(formatted)
    print(f"Completed. Formatted {len(stage1_data)} items.\n")

    # --- STAGE 2: LENGTH FILTERING ---
    print(f"{BLUE}[Stage 2] Running length filters (min 5 characters for inputs/outputs)...{END}")
    stage2_data = []
    for item in stage1_data:
        inst_len = len(item["instruction"])
        out_len = len(item["output"])
        
        if inst_len < 5 or out_len < 5:
            print(f"  [Filtered] ID {item['id']} due to short text: '{item['instruction']}' -> '{item['output']}'")
        else:
            stage2_data.append(item)
    print(f"Completed. Retained {BOLD}{len(stage2_data)}{END} items.\n")

    # --- STAGE 3: NEAR-DUPLICATE DEDUPLICATION (JACCARD) ---
    print(f"{BLUE}[Stage 3] Running Jaccard Near-Duplicate Deduplication (Threshold = 0.70)...{END}")
    stage3_data = []
    removed_ids = set()

    for i in range(len(stage2_data)):
        if stage2_data[i]["id"] in removed_ids:
            continue
            
        current_item = stage2_data[i]
        shingles_a = get_word_shingles(current_item["instruction"], k=2)
        stage3_data.append(current_item)
        
        # Compare pairwise with remaining items
        for j in range(i + 1, len(stage2_data)):
            compare_item = stage2_data[j]
            if compare_item["id"] in removed_ids:
                continue
                
            shingles_b = get_word_shingles(compare_item["instruction"], k=2)
            similarity = compute_jaccard(shingles_a, shingles_b)
            
            if similarity >= 0.70:
                print(f"  [Filtered] Near-duplicate detected (Jaccard Similarity: {similarity:.2f})")
                print(f"    - Base: '{current_item['instruction']}'")
                print(f"    - Dupe: '{compare_item['instruction']}'")
                removed_ids.add(compare_item["id"])
                
    print(f"Completed. Retained {BOLD}{len(stage3_data)}{END} items.\n")

    # --- STAGE 4: CATEGORY BALANCING ---
    print(f"{BLUE}[Stage 4] Performing class balancing (Max 2 examples per category)...{END}")
    # Group by category
    grouped = {}
    for item in stage3_data:
        cat = item["category"]
        grouped[cat] = grouped.get(cat, []) + [item]
    
    stage4_data = []
    print("\n  Class distribution before balancing:")
    for cat, items in grouped.items():
        print(f"    - {cat}: {len(items)} items")
        # Keep only up to 2 items
        balanced_items = items[:2]
        stage4_data.extend(balanced_items)
        
    print("\n  Class distribution after balancing:")
    balanced_grouped = {}
    for item in stage4_data:
        cat = item["category"]
        balanced_grouped[cat] = balanced_grouped.get(cat, []) + [item]
    for cat, items in balanced_grouped.items():
        print(f"    - {cat}: {len(items)} items")
    print(f"\nCompleted. Retained {BOLD}{len(stage4_data)}{END} items.\n")

    # --- STAGE 5: EXPORT TO JSONL ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(script_dir, "cleaned_dataset.jsonl")
    print(f"{BLUE}[Stage 5] Exporting cleaned dataset to {output_filename}...{END}")
    
    with open(output_filename, "w", encoding="utf-8") as f:
        for item in stage4_data:
            # We remove internal tags like 'id' and 'category' if exporting for training,
            # but we can preserve standard instruction/input/output keys.
            export_format = {
                "instruction": item["instruction"],
                "input": item["input"],
                "output": item["output"]
            }
            f.write(json.dumps(export_format) + "\n")
            
    print(f"{GREEN}Dataset curation pipeline complete!{END}")
    print(f"Cleaned dataset written successfully. Total examples: {BOLD}{len(stage4_data)}{END} (out of {len(RAW_DATA)} raw inputs).")
    print(f"Final file size: {os.path.getsize(output_filename)} bytes.")

if __name__ == "__main__":
    run_pipeline()
