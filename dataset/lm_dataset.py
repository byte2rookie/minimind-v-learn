import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
class PretrainDataset(Dataset):
    def __init__(self,data_path,tokenizer,max_seqlen=1024):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seqlen = max_seqlen
        self.samples = self.load_data(data_path)
    
    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        text = str(sample['text'])
        encoding = self.tokenizer(
            text,
            max_length=self.max_seqlen,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze()   
        loss_mask = (input_ids != self.tokenizer.pad_token_id) # 确认input_ids中非填充部分，loss_mask形状类似[True, True, ..., False]
        X=torch.tensor(input_ids[:-1],dtype=torch.long)  # 去掉最后一个token
        Y=torch.tensor(input_ids[1:],dtype=torch.long)  # 去掉第一个
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask