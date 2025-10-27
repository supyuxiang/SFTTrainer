import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.data import DataLoader
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '8,9'

import sys
sys.path.insert(0,'/home/yxfeng/SFTTrainer')

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
print('Using device:',device)

import json

model_path = '/data1/yxfeng/models/deepseek/DeepSeek-R1-Distill-Qwen-7B'

tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.to(device)

# model.eval() 返回的是模型本身，支持链式调用
# 调用后模型进入eval模式（关闭dropout等）
model.eval()

prompt = 'what is the reinforcement learning for LLM reasoning?'
inputs = tokenizer(prompt,return_tensors='pt').to(device)
print('inputs:',inputs)
print('-'*100)

# 使用torch.no_grad()上下文管理器来禁用梯度计算
with torch.no_grad():
    outputs = model.generate(**inputs,max_new_tokens=1000)
    print('outputs:',outputs)
    print('-'*100)
    o = tokenizer.decode(outputs[0],skip_special_tokens=True)
    print('o:',o)