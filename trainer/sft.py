import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.data import DataLoader
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '8,9'

import sys
sys.path.insert(0,'/data1/chzhang/fyx/SFTTrainer')
from data.data_basic import DataBasic

class SFTTrainer:
    def __init__(self,model,tokenizer,epochs=100,batch_size=1,lr=1e-6,data_path='/data1/chzhang/fyx/SFTTrainer/data/sft_data.json'):
        self.model = model
        self.tokenizer = tokenizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.data_path = data_path
        self.set_device()
        self.build_dataloader()
        self.build_optimizer()
        self.build_scheduler()
    
    def set_device(self):
        self.device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
    
    def collate_fn(self, batch):
        """
        collate_fn: 将多个样本合并成一个batch
        
        【什么是collate_fn？】
        collate_fn是DataLoader的一个参数，用于自定义如何将多个样本合并成一个batch。
        如果不指定，DataLoader会使用默认的collate_fn（简单拼接列表）。
        
        【为什么需要自定义collate_fn？】
        对于NLP任务，我们需要：
        1. Tokenization: 将文本转换为token IDs
        2. Padding: 将不同长度的序列填充到相同长度
        3. 转换为tensor: 方便输入模型
        
        【例子】
        Input batch: [
            ("What is AI?", "AI is..."),
            ("What is ML?", "ML is...")
        ]
        ↓ collate_fn处理
        Output: {
            'prompt_ids': tensor([[101, 2023, ...], [101, 2023, ...]]),
            'response_ids': tensor([[3456, 7890, ...], [1234, 5678, ...]])
        }
        
        Args:
            batch: list of tuples [(prompt1, response1), (prompt2, response2), ...]
        
        Returns:
            dict: tokenized inputs ready for model
        """
        # 解包batch
        prompts, responses = zip(*batch)
        
        # Tokenize prompts
        # padding=True: 将不同长度的序列填充到batch中最长序列的长度
        # truncation=True: 超过max_length的序列会被截断
        # max_length=512: 最大长度限制
        prompt_inputs = self.tokenizer(
            list(prompts),
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Tokenize responses（同样处理）
        response_inputs = self.tokenizer(
            list(responses),
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        return {
            'prompt_ids': prompt_inputs['input_ids'].to(self.device),
            'prompt_attention_mask': prompt_inputs['attention_mask'].to(self.device),
            'response_ids': response_inputs['input_ids'].to(self.device),
            'response_attention_mask': response_inputs['attention_mask'].to(self.device)
        }
    
    def build_dataloader(self):
        """
        构建DataLoader
        
        【DataLoader的作用】
        DataLoader是PyTorch中用于批量加载数据的工具，它会：
        1. 从dataset中采样batch_size个样本
        2. 调用collate_fn处理这些样本
        3. 返回处理后的batch给训练循环
        
        【为什么需要collate_fn=self.collate_fn？】
        默认的collate_fn只能处理简单的数据类型（如数字）。
        对于NLP任务，我们需要：
        - Tokenization（文本→token IDs）
        - Padding（统一长度）
        - 转换为tensor
        
        所以必须使用自定义的collate_fn！
        """
        with open(self.data_path,'r',encoding='utf-8') as f:
            datas = json.load(f)
        
        # 修复：datas不是data
        assert isinstance(datas, list), "Data must be a list"
        
        # 修复：字典应该用in检查键，不是hasattr
        assert 'prompt' in datas[0] and 'response' in datas[0], "Data must have 'prompt' and 'response' keys"
        
        prompts = [data['prompt'] for data in datas]
        responses = [data['response'] for data in datas]
        dataset = list(zip(prompts, responses))
        
        # 使用自定义的collate_fn
        # collate_fn=self.collate_fn
        # 这个参数告诉DataLoader使用我们的自定义函数来处理batch
        # 每次从dataset取batch_size个样本时，都会调用self.collate_fn
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn  # 自定义的批处理函数
        )
        return self.dataloader
    
    def build_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(),lr=self.lr,weight_decay=0.01)
    
    def build_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=1,gamma=0.9,last_epoch=-1)
    
    def compute_loss(self, outputs, batch):
        """
        计算SFT的损失
        
        SFT Loss = CrossEntropy(predicted_tokens, target_tokens)
        
        详细讲解：
        1. 模型输出logits: (batch_size, seq_len, vocab_size)
        2. 目标tokens: (batch_size, seq_len)
        3. 计算交叉熵损失: -log P(target_token | previous_tokens)
        
        Args:
            outputs: 模型forward的outputs对象（包含logits和loss）
            batch: 包含input_ids和labels的batch
        
        Returns:
            loss: 标量loss值
        """
        # 情况1: outputs包含loss（最简单）
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            return outputs.loss
        
        # 情况2: 手动计算损失
        logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
        labels = batch.response_ids  # Shape: (batch_size, seq_len)
        
        # 计算交叉熵损失
        # shift_logits = logits[..., :-1, :].contiguous()  # 预测的logits
        # shift_labels = labels[..., 1:].contiguous()      # 目标（向右移一位）
        
        # 或者不shift（如果labels已经是正确对齐的）
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Reshape logits和labels
        # logits: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
        # labels: (batch_size, seq_len) -> (batch_size * seq_len,)
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        
        # 计算损失
        loss = loss_fn(logits_flat, labels_flat)
        
        return loss

    def save_model(self, output_dir='/data1/chzhang/fyx/SFTTrainer/checkpoints'):
        """
        保存训练好的模型和tokenizer
        
        Args:
            output_dir: 保存路径
        """
        import os
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存模型和tokenizer（使用Transformers的标准方法）
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # 保存训练指标
        import json
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        
        print(f"模型已保存到: {output_dir}")
        print(f"训练指标已保存到: {metrics_path}")

    def train(self):
        """
        训练循环
        
        【self.model.train() vs self.model.eval()】
        - model.train(): 设置训练模式，启用dropout、batch_norm等训练特性
        - model.eval(): 设置评估模式，关闭dropout，固定batch_norm的参数
        
        【为什么训练时要用torch.no_grad()？】
        本代码没有用torch.no_grad()，因为在训练时需要计算梯度（loss.backward()）
        torch.no_grad()只在推理（生成文本）时使用，可以节省内存和加速
        
        【Context Manager Protocol】
        - model.eval()返回模型本身，不是context manager，不能用with语句
        - torch.no_grad()返回context manager，可以用with语句
        """
        self.model.train()  # 设置训练模式（启用dropout等）
        from tqdm import trange
        from tqdm import tqdm
        import time
        from omegaconf import OmegaConf,DictConfig

        self.metrics = {
            'train/loss':[],
            'train/lr':[],
            'train/time':[],
            'val/loss':[],
        }

        hidden_states_list = []
        attentions_list = []
        step_count = 0

        for epoch in trange(self.epochs,desc='training_epoch'):            
            for batch in tqdm(self.dataloader,desc='training_batch',leave=False):
                step_count += 1
                current_time = time.time()
                
                prompt_ids = batch['prompt_ids']
                prompt_attention_mask = batch['prompt_attention_mask']
                response_ids = batch['response_ids']
                response_attention_mask = batch['response_attention_mask']
                
                self.optimizer.zero_grad()

                # ============================================================================
                # SFT训练核心逻辑：拼接prompt和response，只对response部分计算loss
                # ============================================================================
                
                # 【步骤1】拼接prompt和response
                # dim=1表示沿序列长度维度拼接（不是batch维度）
                # 示例: prompt_ids=[101,2023], response_ids=[3456,7890]
                #       ↓ torch.cat(..., dim=1)
                #       input_ids=[101,2023,3456,7890]
                input_ids = torch.cat([prompt_ids, response_ids], dim=1)
                attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
                
                # 【步骤2】创建labels，实现"只对response部分计算loss"
                # 
                # labels的作用：告诉模型哪些位置需要预测，哪些位置忽略
                # labels中-100的位置：不会计算loss（Prompt部分）
                # labels中真实token ID的位置：会计算loss（Response部分）
                #
                # 【关键】为什么用-100？
                # -100是PyTorch nn.CrossEntropyLoss的默认ignore_index值
                # 这是一个约定俗成的值，因为真实的token ID范围通常是0到vocab_size-1
                # -100不在这个范围内，不会与真实token ID冲突
                # 你可以改成其他值（如-999），但需要指定nn.CrossEntropyLoss(ignore_index=-999)
                #
                # 【创建过程】
                # 1. torch.full_like(input_ids, -100): 创建与input_ids同样形状的全-100张量
                # 2. labels[:, prompt_ids.shape[1]:]: 切片选择prompt之后的位置
                # 3. = response_ids: 用真实的response token ID覆盖
                #
                # 示例:
                # input_ids = [101, 2023, 3456, 7890]  (prompt=2个token, response=2个token)
                # labels = [-100, -100, 3456, 7890]   (前2个忽略，后2个计算loss)
                labels = torch.full_like(input_ids, -100)
                labels[:, prompt_ids.shape[1]:] = response_ids
                
                # ============================================================================
                # 【步骤3】模型前向传播 - 详细讲解model()参数
                # ============================================================================
                #
                # outputs = self.model(...) 返回的是ModelOutput对象，包含：
                #   - logits: (batch_size, seq_len, vocab_size) - 每个位置对每个词的预测分数
                #   - loss: 标量 - 如果传入labels参数，模型会自动计算loss
                #   - hidden_states: 每层的隐藏状态（可选）
                #   - attentions: 注意力权重（可选）
                #
                # 【参数详解】
                # 1. input_ids: (batch_size, seq_len)
                #    作用：输入序列的token IDs
                #    内容：prompt和response拼接后的完整序列
                #
                # 2. attention_mask: (batch_size, seq_len)
                #    作用：告诉模型哪些位置是真实token，哪些是padding
                #    值：1表示真实token，0表示padding
                #    为什么需要：batch中样本长度不同，需要padding到相同长度
                #
                # 3. labels: (batch_size, seq_len)
                #    作用：告诉模型计算loss时：
                #          - labels[i,j] = -100: 位置j不计算loss（Prompt部分）
                #          - labels[i,j] = token_id: 位置j预测token_id，计算loss（Response部分）
                #    
                #    【模型内部如何处理labels？】
                #    当传入labels参数时，模型的forward方法会：
                #    1. 计算logits: 对每个位置预测所有可能词的分数
                #    2. 自动计算loss:
                #       - 提取logits: (batch*seq_len, vocab_size)
                #       - Reshape labels: (batch*seq_len,)
                #       - 调用nn.CrossEntropyLoss(logits, labels, ignore_index=-100)
                #       - 只对labels != -100的位置计算交叉熵
                #    3. 返回ModelOutput(loss=loss, logits=logits)
                #
                # 【这就是为什么传入labels会自动计算loss】
                # 它内部做了：
                #   loss = CrossEntropyLoss(
                #       logits.view(-1, vocab_size), 
                #       labels.view(-1), 
                #       ignore_index=-100
                #   )
                #
                outputs = self.model(
                    input_ids=input_ids,      # 完整输入序列 [prompt][response]
                    attention_mask=attention_mask,  # 标记真实token位置
                    labels=labels,            # 标记哪些位置计算loss，哪些忽略(-100)
                    output_hidden_states=True,  # 返回hidden_states
                    output_attentions=True     # 返回attentions
                )

                # 【步骤4】提取loss
                # outputs.loss 是模型自动计算的交叉熵损失
                # 它只计算labels中不为-100的位置的平均loss
                loss_batch = outputs.loss
                hidden_states = outputs.hidden_states
                attentions = outputs.attentions
                hidden_states_list.append(hidden_states)
                attentions_list.append(attentions)
                
                
                # Backward pass
                loss_batch.backward()
                
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                
                # 定期清理GPU缓存
                if step_count % 50 == 0:
                    torch.cuda.empty_cache()
                
                metrics_batch = {
                    'train/loss':loss_batch.item(),
                    'train/lr':self.scheduler.get_last_lr()[0],
                    'train/epoch':epoch,
                    'train/time':time.time() - current_time,
                }
                self.metrics['train/loss'].append(metrics_batch['train/loss'])
                self.metrics['train/lr'].append(metrics_batch['train/lr'])
                self.metrics['train/time'].append(metrics_batch['train/time'])
                # self.metrics['val/loss'].append(metrics_batch['val/loss'])  # 暂时没有val
        self.save_model()
        return self.metrics

if __name__ == '__main__':
    model_path = '/data1/models/qwen/Qwen2.5-0.5B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    
    # 强制使用 eager attention 以支持 output_attentions=True
    # SDPA (Scaled Dot Product Attention) 优化版本不支持返回注意力权重
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation='eager'  # 使用 eager attention 而不是 SDPA
    )
    
    trainer = SFTTrainer(model,tokenizer)
    metrics = trainer.train()
    print('hidden_states_list:',trainer.hidden_states_list)
    print('-'*100)
    print('attentions_list:',trainer.attentions_list)
    print('-'*100)
    print(metrics)