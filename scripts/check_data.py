import json
import yaml
import os
import sys


class CheckData:
    def __init__(self,data_path,dump_path:str=None,is_check:bool=False):
        self.dump_path = dump_path
        self.data_path = data_path
        self.load_data()
        if is_check:
            self.check_data()
        self.rebuild_data()
        self.dump_data()
    
    def load_data(self):
        assert os.path.exists(self.data_path)
        if self.data_path.endswith('.parquet'):
            import pandas as pd
            self.data = pd.read_parquet(self.data_path)
        elif self.data_path.endswith('.json'):
            with open(self.data_path,'r',encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            raise ValueError(f"Unsupported data format: {self.data_path}")
    
    def check_data(self):
        print(f"Loaded {len(self.data)} data points")
        print(f"First 5 data points: {self.data.head()}")
        print(f"Last 5 data points: {self.data.tail()}")
        print(f"Data types: {self.data.dtypes}")
        print(f"Missing values: {self.data.isnull().sum()}")
        print(f"Data shape: {self.data.shape}")
        print(f"Data columns: {self.data.columns}")
        print(f"Data index: {self.data.index}")
        print(f"Data values: {self.data.values}")
        print('='*1000)
        example = self.data.iloc[0,:]
        print(f"Example data: {example}")
        print(f'keys: {example.keys()}')
        for key,value in example.items():
            print(f"{key}: {value}")
            print('='*100)
    
    def rebuild_data(self):

        self.rebuild_data = {
            'prompt':[],
            'response':[]
        }
        is_json = self.data_path.endswith('.json')
        if is_json:
            for example in self.data:
                prompt = example.get('prompt')
                response = example.get('response')
                self.rebuild_data['prompt'].append(prompt)
                self.rebuild_data['response'].append(response)
            return
            

        if self.data_path.endswith('aime_2024/train.parquet'):
            for i in range(len(self.data)):
                example = self.data.iloc[i,:]
                prompt = example.get('problem')
                response = example.get('solution')
                self.rebuild_data['prompt'].append(prompt)
                self.rebuild_data['response'].append(response)
        else:
            for i in range(len(self.data)):
                example = self.data.iloc[i,:]
                prompt = example.get('prompt')[0].get('content')
                response = example.get('extra_info').get('answer')
                self.rebuild_data['prompt'].append(prompt)
                self.rebuild_data['response'].append(response)
    
    def dump_data(self):
        if self.dump_path:
            assert self.dump_path.endswith('.json'), "Dump path must be a json file"
            # Create directory if it doesn't exist
            dir_path = os.path.dirname(self.dump_path)
            if dir_path:  # Only create if there's a directory path
                os.makedirs(dir_path, exist_ok=True)
            # Save the file
            with open(self.dump_path,'w',encoding='utf-8') as f:
                json.dump(self.rebuild_data,f,ensure_ascii=False,indent=4)
            print(f"Data dumped to {self.dump_path}")
        
            

'''if __name__ == '__main__':
    data_path = '/data1/chzhang/fyx/SFTTrainer/datasets/gsm8k/test.parquet'
    dump_path = '/data1/chzhang/fyx/SFTTrainer/datasets/gsm8k/test.json'
    check_data = CheckData(data_path,dump_path)'''

'''if __name__ == '__main__':
    data_path = '/data1/chzhang/fyx/SFTTrainer/datasets/aime_2024/train.parquet'
    dump_path = '/data1/chzhang/fyx/SFTTrainer/datasets/aime_2024/train_simple.json'
    check_data = CheckData(data_path,dump_path)'''


'''if __name__ == '__main__':
    data_path = '/data1/chzhang/fyx/SFTTrainer/datasets/amc23/train.json'
    dump_path = '/data1/chzhang/fyx/SFTTrainer/datasets/amc23/train_simple.json'
    check_data = CheckData(data_path,dump_path)'''

'''if __name__ == '__main__':
    data_path = '/data1/chzhang/fyx/SFTTrainer/datasets/amc23/test.json'
    dump_path = '/data1/chzhang/fyx/SFTTrainer/datasets/amc23/test_simple.json'
    check_data = CheckData(data_path,dump_path)'''

'''if __name__ == '__main__':
    data_path = '/data1/chzhang/fyx/SFTTrainer/datasets/math500/train.json'
    dump_path = '/data1/chzhang/fyx/SFTTrainer/datasets/math500/train_simple.json'
    check_data = CheckData(data_path,dump_path)'''

if __name__ == '__main__':
    data_path = '/data1/chzhang/fyx/SFTTrainer/datasets/math500/test.json'
    dump_path = '/data1/chzhang/fyx/SFTTrainer/datasets/math500/test_simple.json'
    check_data = CheckData(data_path,dump_path)




























