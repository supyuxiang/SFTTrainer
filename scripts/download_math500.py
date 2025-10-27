"""
Download MATH-500 dataset from HuggingFace
Dataset URL: https://huggingface.co/datasets/HuggingFaceH4/MATH-500
"""

import os
from datasets import load_dataset
import json
import pandas as pd

def download_math500_dataset():
    """Download MATH-500 dataset and save to parquet and json formats"""
    
    # Set output directory
    output_dir = '/home/yxfeng/SFTTrainer/datasets/math500'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading MATH-500 dataset from HuggingFace...")
    
    # Load dataset from HuggingFace
    dataset = load_dataset('HuggingFaceH4/MATH-500')
    
    print(f"Dataset loaded successfully!")
    print(f"Available splits: {list(dataset.keys())}")
    
    # Process test split (MATH-500 only has test split)
    split_name = 'test' if 'test' in dataset else list(dataset.keys())[0]
    
    if split_name in dataset:
        data = dataset[split_name]
        print(f"\n{split_name} split: {len(data)} samples")
        print(f"Features: {data.features}")
        
        # Convert to pandas DataFrame
        df = data.to_pandas()
        
        # Save as parquet
        parquet_path = os.path.join(output_dir, f'{split_name}.parquet')
        df.to_parquet(parquet_path, index=False)
        print(f"Saved parquet to: {parquet_path}")
        
        # Convert to SFT format (prompt + response)
        sft_data = []
        for idx, row in df.iterrows():
            # Create prompt
            prompt = f"Problem: {row['problem']}\n\nSolve this step by step and provide your final answer."
            
            # Create response (solution + answer)
            response = f"{row['solution']}\n\nFinal Answer: {row['answer']}"
            
            sft_data.append({
                'prompt': prompt,
                'response': response,
                'id': row.get('unique_id', idx),
                'answer': row.get('answer', ''),
                'subject': row.get('subject', ''),
                'level': row.get('level', '')
            })
        
        # Save as JSON (both as train.json for consistency)
        json_path = os.path.join(output_dir, 'train.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(sft_data, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON to: {json_path}")
        
        # Also save original test.json
        test_json_path = os.path.join(output_dir, f'{split_name}.json')
        with open(test_json_path, 'w', encoding='utf-8') as f:
            json.dump(sft_data, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON to: {test_json_path}")
        
        # Print first example
        print("\n" + "="*100)
        print("First example:")
        print("="*100)
        print(f"Prompt: {sft_data[0]['prompt'][:200]}...")
        print(f"\nResponse: {sft_data[0]['response'][:200]}...")
    
    print(f"\n{'='*100}")
    print(f"Dataset download completed!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*100}")

if __name__ == '__main__':
    download_math500_dataset()

