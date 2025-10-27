"""
Dynamic Fine-Tuning (DFT) Implementation

This file implements the DFT training method based on the paper summary provided.
The key innovation is the corrected loss function that addresses implicit reward
structure issues in traditional Supervised Fine-Tuning (SFT).

Main Components:
1. DFTTrainer class: Implements the DFT training loop with corrected loss
2. compute_dft_loss: Computes the DFT loss: L_DFT = -π_θ(y*|x) · log π_θ(y*|x)
3. Training loop: Follows paper's recommendations for hyperparameters

Key Differences from Standard SFT:
- Standard SFT: L_SFT = -log π_θ(y*|x)
- DFT: L_DFT = -π_θ(y*|x) · log π_θ(y*|x)

The probability π_θ(y*|x) serves as a dynamic re-weighting factor that corrects
the implicit reward structure and stabilizes gradient updates.

Date: 2024
"""

import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
import json
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '6,7,8,9'

import sys
sys.path.insert(0, '/data1/chzhang/fyx/SFTTrainer')
from data.data_basic import DataBasic


class DFTTrainer:
    """
    Dynamic Fine-Tuning (DFT) Trainer
    
    Implementation of the DFT method from the paper "Dynamic Fine-Tuning".
    
    Key Mathematical Differences from Standard SFT:
    --------------------------------------------------------------------------------
    Standard SFT Loss:
        L_SFT = -log π_θ(y*|x)
    
    DFT Loss:
        L_DFT = -Σ_{t=1}^{T} π_θ(y_t^* | x, y_{<t}^*) · log π_θ(y_t^* | x, y_{<t}^*)
               = -π_θ(y*|x) · log π_θ(y*|x)
    
    Motivation:
    Traditional SFT has an implicit reward structure where r(x,y) ∝ 1/π_θ(y|x),
    which causes optimization instability when the model assigns low probability to
    expert actions. DFT corrects this by dynamically re-weighting the loss function,
    converting probability-dependent gradient estimators into uniformly weighted updates.
    
    Implementation Details:
    - Uses stop-gradient operation (.detach()) on the probability term π_θ(y*|x)
      to ensure gradients don't flow through the reward scaling term
    - Supports multi-GPU training via torch.nn.DataParallel
    - Follows paper's training recommendations:
      * Optimizer: AdamW
      * Learning rate: 5e-5 (2e-5 for LLaMA-3.1-8B)
      * Batch size: 256
      * Sequence length: 2048
      * LR schedule: Cosine decay with 10% warmup
      * Training epochs: 1
    
    Usage Example:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_path = '/path/to/model'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        trainer = DFTTrainer(
            model=model,
            tokenizer=tokenizer,
            epochs=1,
            batch_size=1,
            lr=5e-5,
            data_path='/path/to/data.json'
        )
        
        metrics = trainer.train()
    """
    
    def __init__(self, model, tokenizer, epochs=3, batch_size=4, lr=5e-5, 
                 data_path='/data1/chzhang/fyx/SFTTrainer/datasets/math500/train.json',
                 max_length=1024):
        self.model = model
        self.tokenizer = tokenizer
        self.epochs = epochs
        self.lr = lr
        self.data_path = data_path
        self.max_length = max_length
        
        # Adjust batch_size based on number of GPUs
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if num_gpus > 1:
            # Round up to nearest multiple of num_gpus for even distribution
            self.batch_size = ((batch_size + num_gpus - 1) // num_gpus) * num_gpus
            print(f'Adjusted batch_size from {batch_size} to {self.batch_size} for {num_gpus} GPUs')
        else:
            self.batch_size = batch_size
        
        self.set_device()
        self.build_dataloader()
        self.build_optimizer()
        self.build_scheduler()
    
    def set_device(self):
        """Set the training device"""
        if torch.cuda.is_available():
            # Check number of available GPUs
            num_gpus = torch.cuda.device_count()
            print(f'Available GPUs: {num_gpus}')
            
            if num_gpus > 1:
                # Use DataParallel for multiple GPUs
                self.device = torch.device('cuda:0')
                self.model.to(self.device)
                self.model = torch.nn.DataParallel(self.model)
                print(f'Using DataParallel on {num_gpus} GPUs')
                for i in range(num_gpus):
                    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
            else:
                # Single GPU
                self.device = torch.device('cuda:0')
                self.model.to(self.device)
                print(f'Using single GPU: {torch.cuda.get_device_name(0)}')
            
            print(f'CUDA version: {torch.version.cuda}')
        else:
            self.device = torch.device('cpu')
            self.model.to(self.device)
            print(f'Using device: {self.device}')
    
    def collate_fn(self, batch):
        """
        Collate function for DataLoader
        Tokenizes prompts and responses, handles padding
        
        Args:
            batch: list of tuples [(prompt1, response1), (prompt2, response2), ...]
        
        Returns:
            dict: tokenized inputs ready for model
        """
        prompts, responses = zip(*batch)
        
        # Tokenize prompts
        prompt_inputs = self.tokenizer(
            list(prompts),
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Tokenize responses
        response_inputs = self.tokenizer(
            list(responses),
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Don't move to device here - let DataParallel handle it for multi-GPU efficiency
        return {
            'prompt_ids': prompt_inputs['input_ids'],
            'prompt_attention_mask': prompt_inputs['attention_mask'],
            'response_ids': response_inputs['input_ids'],
            'response_attention_mask': response_inputs['attention_mask']
        }
    
    def build_dataloader(self):
        """Build DataLoader from the dataset"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            datas = json.load(f)
        
        assert isinstance(datas, list), "Data must be a list"
        assert 'prompt' in datas[0] and 'response' in datas[0], "Data must have 'prompt' and 'response' keys"
        
        prompts = [data['prompt'] for data in datas]
        responses = [data['response'] for data in datas]
        dataset = list(zip(prompts, responses))
        
        # Set num_workers for multi-GPU training
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        num_workers = min(4, num_gpus * 2) if num_gpus > 1 else 0
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f'Loaded {len(dataset)} samples from {self.data_path}')
        print(f'DataLoader: batch_size={self.batch_size}, num_workers={num_workers}')
        return self.dataloader
    
    def build_optimizer(self):
        """Build AdamW optimizer with weight decay"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=0.01
        )
    
    def build_scheduler(self):
        """Build learning rate scheduler with cosine decay and warmup"""
        from transformers import get_cosine_schedule_with_warmup
        
        total_steps = len(self.dataloader) * self.epochs
        warmup_steps = int(total_steps * 0.1)  # 10% warmup as per paper
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    
    def compute_dft_loss(self, logits, labels):
        """
        Compute DFT loss: L_DFT = -Σ_{t=1}^{T} π_θ(y_t^* | x, y_{<t}^*) · log π_θ(y_t^* | x, y_{<t}^*)
        
        This is the key innovation of DFT compared to standard SFT.
        By multiplying by the probability π_θ(y*|x), we correct the implicit
        reward structure and stabilize gradient updates.
        
        Args:
            logits: (batch_size, seq_len, vocab_size) - model predictions
            labels: (batch_size, seq_len) - ground truth token IDs (-100 for ignored positions)
        
        Returns:
            loss: scalar DFT loss value
        """
        # Ensure labels are on the same device as logits
        labels = labels.to(logits.device)
        
        batch_size, seq_len, vocab_size = logits.shape
        
        # Compute probabilities and log probabilities
        probs = torch.softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
        log_probs = torch.log_softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
        
        # Create indices to extract probability and log_prob of true tokens
        batch_indices = torch.arange(batch_size, device=logits.device).unsqueeze(1).expand(-1, seq_len)
        seq_indices = torch.arange(seq_len, device=logits.device).unsqueeze(0).expand(batch_size, -1)
        
        # Extract π_θ(y_t^* | x, y_{<t}^*) for each position
        true_token_probs = probs[batch_indices, seq_indices, labels]  # (batch_size, seq_len)
        
        # Extract log π_θ(y_t^* | x, y_{<t}^*) for each position
        true_token_log_probs = log_probs[batch_indices, seq_indices, labels]  # (batch_size, seq_len)
        
        # Create mask for valid positions (not -100)
        valid_mask = (labels != -100)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Compute DFT loss: -π_θ(y*|x) · log π_θ(y*|x)
        # The paper mentions using stop-gradient for the probability term
        # to ensure gradients don't flow through the reward scaling term
        true_token_probs_detached = true_token_probs.detach()
        
        # Compute loss per token
        dft_loss_per_token = -true_token_probs_detached * true_token_log_probs
        
        # Mask out ignored positions - ensure all tensors are on the same device
        zero_tensor = torch.zeros_like(dft_loss_per_token)
        dft_loss_per_token = torch.where(valid_mask, dft_loss_per_token, zero_tensor)
        
        # Average over all valid tokens
        loss = dft_loss_per_token.sum() / valid_mask.sum()
        
        return loss
    
    def compute_loss(self, outputs, batch):
        """
        Compute DFT loss from model outputs
        
        Args:
            outputs: model forward outputs containing logits
            batch: batch dictionary with labels
        
        Returns:
            loss: DFT loss value
        """
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)
        labels = batch['labels']  # (batch_size, seq_len)
        
        loss = self.compute_dft_loss(logits, labels)
        return loss
    
    def save_model(self, output_dir='/data1/chzhang/fyx/SFTTrainer/checkpoints/dft'):
        """Save trained model and tokenizer"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Handle DataParallel: extract the underlying model
        model_to_save = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training metrics
        import json
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        
        print(f"Model saved to: {output_dir}")
        print(f"Metrics saved to: {metrics_path}")
    
    def train(self):
        """
        Main training loop for DFT
        
        Implements the training procedure described in the DFT paper:
        - Optimizer: AdamW
        - Learning rate: 5e-5 (2e-5 for LLaMA-3.1-8B)
        - Batch size: 256
        - Sequence length: 2048
        - LR schedule: Cosine decay with warmup ratio 0.1
        - Training epochs: 1
        """
        self.model.train()
        
        from tqdm import trange, tqdm
        import time
        
        self.metrics = {
            'train/loss': [],
            'train/lr': [],
            'train/time': [],
        }
        
        step_count = 0
        
        for epoch in trange(self.epochs, desc='Training epoch'):
            for batch in tqdm(self.dataloader, desc='Training batch', leave=False):
                step_count += 1
                current_time = time.time()
                
                prompt_ids = batch['prompt_ids']
                prompt_attention_mask = batch['prompt_attention_mask']
                response_ids = batch['response_ids']
                response_attention_mask = batch['response_attention_mask']
                
                # Move to device if not using DataParallel (DataParallel handles this automatically)
                if not isinstance(self.model, torch.nn.DataParallel):
                    prompt_ids = prompt_ids.to(self.device)
                    prompt_attention_mask = prompt_attention_mask.to(self.device)
                    response_ids = response_ids.to(self.device)
                    response_attention_mask = response_attention_mask.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Concatenate prompt and response
                input_ids = torch.cat([prompt_ids, response_ids], dim=1)
                attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
                
                # Create labels: only compute loss on response tokens
                # Ensure labels are on the same device as input_ids
                labels = torch.full_like(input_ids, -100)
                labels[:, prompt_ids.shape[1]:] = response_ids
                
                # Forward pass - don't pass labels to model, we'll compute DFT loss manually
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    output_attentions=False
                )
                
                # Extract logits and compute DFT loss
                logits = outputs.logits
                
                # Ensure labels are on the same device as logits
                labels = labels.to(logits.device)
                
                dft_loss = self.compute_dft_loss(logits, labels)
                
                # Backward pass
                dft_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                
                # Clear GPU cache periodically
                if step_count % 50 == 0:
                    torch.cuda.empty_cache()
                
                # Log metrics
                metrics_batch = {
                    'train/loss': dft_loss.item(),
                    'train/lr': self.scheduler.get_last_lr()[0],
                    'train/epoch': epoch,
                    'train/time': time.time() - current_time,
                }
                
                self.metrics['train/loss'].append(metrics_batch['train/loss'])
                self.metrics['train/lr'].append(metrics_batch['train/lr'])
                self.metrics['train/time'].append(metrics_batch['train/time'])
                
                # Print progress every 10 steps
                if step_count % 10 == 0:
                    print(f"Step {step_count}, Loss: {dft_loss.item():.4f}, LR: {self.scheduler.get_last_lr()[0]:.2e}")
        
        self.save_model()
        return self.metrics


if __name__ == '__main__':
    # Initialize model and tokenizer
    model_path = '/data1/models/qwen/Qwen2.5-0.5B-Instruct'
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation='eager'
    )
    
    # Create DFT trainer
    trainer = DFTTrainer(
        model=model,
        tokenizer=tokenizer,
        epochs=100,
        batch_size=1,
        lr=5e-5,  # Paper default: 5e-5 (use 2e-5 for LLaMA-3.1-8B)
        data_path='/data1/chzhang/fyx/SFTTrainer/datasets/math500/train.json',
        max_length=2048
    )
    
    # Train the model
    print("Starting DFT training...")
    metrics = trainer.train()
    
    print('-' * 100)
    print("Training completed!")
    print(f"Final loss: {metrics['train/loss'][-1]:.4f}")
    print('-' * 100)
