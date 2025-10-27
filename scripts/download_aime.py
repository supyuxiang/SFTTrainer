"""
Download AIME 2024 dataset from HuggingFace
Dataset URL: https://huggingface.co/datasets/HuggingFaceH4/aime_2024
"""

import os
from datasets import load_dataset
import json
import pandas as pd

def download_aime_dataset():
    """Download AIME 2024 dataset and save to parquet and json formats"""
    
    # Set output directory
    output_dir = '/home/yxfeng/SFTTrainer/datasets/aime_2024'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading AIME 2024 dataset from HuggingFace...")
    
    # Load dataset from HuggingFace
    dataset = load_dataset('HuggingFaceH4/aime_2024')
    
    print(f"Dataset loaded successfully!")
    print(f"Available splits: {list(dataset.keys())}")
    
    # Process train split
    if 'train' in dataset:
        train_data = dataset['train']
        print(f"\nTrain split: {len(train_data)} samples")
        print(f"Features: {train_data.features}")
        
        # Convert to pandas DataFrame
        df = train_data.to_pandas()
        
        # Save as parquet
        parquet_path = os.path.join(output_dir, 'train.parquet')
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
                'id': row.get('id', idx),
                'url': row.get('url', ''),
                'year': row.get('year', '2024')
            })
        
        # Save as JSON
        json_path = os.path.join(output_dir, 'train.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(sft_data, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON to: {json_path}")
        
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
    download_aime_dataset()

