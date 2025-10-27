import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer,AutoModelForCausalLM
from typing import List,Tuple,Dict,Optional,Any
import os
import sys
sys.path.insert(0,'/data1/chzhang/fyx/SFTTrainer')

from data.data_ours import DataOurs


class SFTAETrainer:
    def __init__(self,model_path:str,logger,epochs:int=3,batch_size:int=32,lr:float=1e-4,is_val:bool=True):
        self.model_path = model_path
        self.logger = logger
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_val = is_val
        self.build_model()
        self.build_dataloader()
        self.total_steps = self.epochs * len(self.dataloader)
        self.build_optimizer()
        self.build_scheduler()
    
    
    def build_model(self):
        assert os.path.exists(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path,attn_implementation='eager')
        self.model.to(self.device)
        self.logger.info(f'Model loaded from {self.model_path}')
    
    def build_dataloader(self):

        self.logger.info(f'Dataloader built')
        pass

    def build_optimizer(self):
        self.optimizer = optim.AdamW(self.model.parameters(),lr=self.lr)
        self.logger.info(f'Optimizer built')
    
    def build_scheduler(self):
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,step_size=1,gamma=0.9)
        self.logger.info(f'Scheduler built')
    
    def train(self):
        self.logger.info(f'Training started')
        self.model.train()
        step = 0
        self.metrics = {
            'train/loss':[],
            'train/step':[],
            'train/lr':[],
            'train/representation':[],
            'val/loss':[],
            'val/accuracy':[]
        }
        
        pass





















