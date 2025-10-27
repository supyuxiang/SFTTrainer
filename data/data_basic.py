import os
import pandas as pd
from torch.utils.data import DataLoader

class DataBasic:
    def __init__(self,data_path:str,batch_size:int=1,shuffle:bool=True,max_length:int=1024,data_filter:bool=True):
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_length = max_length
        self.data_filter = data_filter
        self.build_dataset()
        self.build_dataloader()
    
    def build_dataset(self):
        assert os.path.exists(self.data_path)
        if self.data_path.endswith('.json'):
            import json
            with open(self.data_path,'r',encoding='utf-8') as f:
                self.data = json.load(f)
        elif self.data_path.endswith('.yaml'):
            import yaml
            with open(self.data_path,'r',encoding='utf-8') as f:
                self.data = yaml.safe_load(f)
        else:
            raise ValueError(f'Unsupported file extension: {self.data_path}')
        
        assert isinstance(self.data,list)
        assert 'prompt' in self.data[0].keys() and 'response' in self.data[0].keys()

        prompt_list = [item['prompt'] for item in self.data]
        response_list = [item['response'] for item in self.data]

        self.dataset = list(zip(prompt_list,response_list))
        
        if self.data_filter:
            print('filtering data...')
            self._dataset_filter()
        
    
    def build_dataloader(self):
        '''
        DataLoader 主要支持三种类型的 dataset 输入：

        1. Map-style 数据集 (最常见)
        实现了 __getitem__()和 __len__()方法，通过索引访问数据样本。示例：torchvision.datasets.ImageFolder

        2. Iterable-style 数据集
        实现了 __iter__()方法，适用于流式数据或无法随机访问的数据。示例：处理大型数据库或实时生成的数据。
           
        3. 自定义数据集
        用户自定义的数据集类，需要实现上述方法之一。示例：自定义数据集类。
        '''
        self.dataloader = DataLoader(self.dataset,batch_size=self.batch_size,shuffle=self.shuffle,collate_fn=self.collate_fn)
    
    def _dataset_filter(self):
        for i in range(len(self.dataset)):
            if len(self.dataset[i][0]) + len(self.dataset[i][1]) > self.max_length:
                self.dataset.pop(i)
            print(f'filtered {len(self.data) - len(self.dataset)} data')


    def collate_fn(self,batch):
        prompt_list = [item[0] for item in batch]
        response_list = [item[1] for item in batch]
        prompt_ids = self.tokenizer(prompt_list,return_tensors='pt',padding=True,truncation=True,max_length=self.max_length)['input_ids']
        response_ids = self.tokenizer(response_list,return_tensors='pt',padding=True,truncation=True,max_length=self.max_length)['input_ids']
        return prompt_ids,response_ids

if __name__ == '__main__':
    data_path = '/home/yxfeng/SFTTrainer/datasets/math500/train_simple.json'
    data_basic = DataBasic(data_path,batch_size=2,shuffle=True,max_length=1024)
    print(data_basic.dataset)
    print(data_basic.dataloader)


        















