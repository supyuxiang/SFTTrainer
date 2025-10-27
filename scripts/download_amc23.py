"""
Download AMC23 dataset from HuggingFace
Dataset URL: https://huggingface.co/datasets/zwhe99/amc23
"""

import os
from datasets import load_dataset
import json
import pandas as pd

def download_amc23_dataset():
    """Download AMC23 dataset and save to parquet and json formats"""
    
    # Set output directory
    output_dir = '/home/yxfeng/SFTTrainer/datasets/amc23'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading AMC23 dataset from HuggingFace...")
    
    # Load dataset from HuggingFace
    dataset = load_dataset('zwhe99/amc23')
    
    print(f"Dataset loaded successfully!")
    print(f"Available splits: {list(dataset.keys())}")
    
    # Process test split (AMC23 only has test split)
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
            prompt = f"Problem: {row['question']}\n\nSolve this step by step and provide your final answer."
            
            # Create response (since this dataset doesn't have solutions, we'll use a placeholder)
            # The answer field contains the answer
            answer = str(row['answer'])
            response = f"Let me solve this problem step by step.\n\n[Solution steps would go here]\n\nFinal Answer: {answer}"
            
            sft_data.append({
                'prompt': prompt,
                'response': response,
                'id': row.get('id', idx),
                'answer': answer,
                'url': row.get('url', '')
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
    download_amc23_dataset()

