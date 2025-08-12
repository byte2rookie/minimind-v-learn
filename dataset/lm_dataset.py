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
    

class SFTDataset(Dataset):
    def __init__(self,data_path,tokenizer,max_seqlen=1024):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seqlen = max_seqlen
        self.samples = self.load_data(data_path)
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids
        print(f"bos_id={self.bos_id}, eos_id={self.eos_id}")

    def __len__(self):
        return len(self.samples)    

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples
    
    def _create_chat_prompt(self, conversations):
            """构建符合ChatML格式的对话"""
            messages = []
            for i, turn in enumerate(conversations):
                role = 'user' if i % 2 == 0 else 'assistant'
                messages.append({"role": role, "content": turn['content']})
            res=self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            print(f"res={res}")
            return res
    
    def _generate_loss_mask(self, input_ids):
            input_ids_list = input_ids.tolist()
            loss_mask = [0] * len(input_ids_list)
            i = 0
            while i < len(input_ids_list):
                if input_ids_list[i:i + len(self.bos_id)] == self.bos_id:
                    start = i + len(self.bos_id)
                    end = start
                    while end < len(input_ids_list):
                        if input_ids_list[end:end + len(self.eos_id)] == self.eos_id:
                            break
                        end += 1
                    for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_seqlen)):
                        loss_mask[j] = 1
                    i = end + len(self.eos_id) if end < len(input_ids_list) else len(input_ids_list)
                else:
                    i += 1
            return loss_mask
    
    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = sample['conversations']
        input_ids = self._create_chat_prompt(conversations)
        input_ids = self.tokenizer(input_ids, max_length=self.max_seqlen, padding='max_length', truncation=True,return_tensors='pt').input_ids.squeeze()

        
        loss_mask = self._generate_loss_mask(input_ids)
        
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask