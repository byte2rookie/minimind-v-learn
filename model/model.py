import torch 
from torch import nn
from config import MinimindConfig
class Embed(nn.Module):
    def __init__(self, config:MinimindConfig):
        super(Embed, self).__init__()
        vocab_size = config.vocab_size
        embed_dim = config.embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, input_ids):
        return self.embedding(input_ids)

class RMSNorm(nn.Module):
    def __init__(self,config:MinimindConfig):
        super(RMSNorm,self).__init__()
        self.embed_dim = config.embed_dim
        self.eps = config.norm_eps
        self.gamma = nn.Parameter(torch.ones(self.embed_dim))
    
    def forward(self,x):
        # print(f"X after RMSNorm: {x}")
        return x*self.gamma*torch.rsqrt(x.pow(2).mean(dim=-1,keepdim=True)+self.eps)


def precompute_pos_cis(embed_dim=512,max_seqlen=512,theta=1e5):
    freqs= 1/theta**torch.arange(0,embed_dim,2)[:embed_dim//2].float()
    m=torch.arange(max_seqlen,device=freqs.device)
    freqs= torch.outer(m,freqs).float() #获取了mtheta
    pos_cis = torch.polar(torch.ones_like(freqs),freqs) #将mtheta化为极坐标模式
    return pos_cis

def apply_rotary(xq,xk,pos_cis):
    xq_=torch.view_as_complex(xq.float().reshape(*xq.shape[:-1],-1,2))
    xk_=torch.view_as_complex(xk.float().reshape(*xk.shape[:-1],-1,2))
    #输入的pos_cis一般比xq,xk都要大，需要把pos_cis的形状和xq对齐
    #xq一般都是(bs,seqlen,head,head_dim)
    def unite_shape(pos_cis,  x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1],  x.shape[-1]), f"pos_cis shape {pos_cis.shape} does not match x shape {x.shape}"
        shape = [d if i == 1 or i == ndim - 1 else 1 for i,  d in enumerate(x.shape)]
        return pos_cis.view(*shape)
    pos_cis = unite_shape(pos_cis, xq_)
    xq_ = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_ = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_, xk_

# repeat_kv是必须要用到的,对齐GQA里的KV与Q的形状
def repeat_kv(x,rep_num):
    if rep_num == 1:
        return x
    bs,seqlen,head,head_dim=x.shape
    return x[:,:,:,None,:].expand(bs,seqlen,head,rep_num,head_dim).reshape(bs,seqlen,head*rep_num,head_dim)


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class GroupQueryAttention(nn.Module):
    def __init__(self,config:MinimindConfig):
        super(GroupQueryAttention,self).__init__()
        #基本属性
        self.embed_dim = config.embed_dim
        self.head_num = config.head_num
        self.kv_head_num = config.kv_head_num
        self.head_dim = self.embed_dim // self.head_num
        assert self.embed_dim % self.head_num == 0, "embed_dim must be divisible by head_num"
        self.rep_num = self.head_num // self.kv_head_num
        assert self.head_num % self.kv_head_num == 0, "kv_head_num must be divisible by head_num"
        self.Flash = hasattr(torch.nn.functional,'scaled_dot_product_attention') and config.Flash
        #网络层
        self.q_proj = nn.Linear(self.embed_dim,self.head_num * self.head_dim)
        self.k_proj = nn.Linear(self.embed_dim,self.kv_head_num * self.head_dim)
        self.v_proj = nn.Linear(self.embed_dim,self.kv_head_num * self.head_dim)
        self.o_proj = nn.Linear(self.head_num*self.head_dim,self.embed_dim)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.res_dropout = nn.Dropout(config.attn_res_dropout)
        self.max_seqlen = config.max_seqlen
        #临时
        mask = torch.full((1,1, self.max_seqlen, self.max_seqlen), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer('mask', mask)
    def forward(self,x,
                pos_cis=None,
                past_key_value=None,
                use_cache=False):
        bs,seqlen,embed_dim = x.shape
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        xq = xq.view(bs, seqlen, self.head_num, self.head_dim)
        xk = xk.view(bs, seqlen, self.kv_head_num, self.head_dim)
        xv = xv.view(bs, seqlen, self.kv_head_num, self.head_dim)
        if pos_cis is None:
            pos_cis = precompute_pos_cis(embed_dim=self.head_dim, max_seqlen=seqlen)
        xq, xk = apply_rotary(xq, xk, pos_cis)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq = xq.transpose(1,2)
        xk = repeat_kv(xk, self.rep_num).transpose(1,2)
        xv = repeat_kv(xv, self.rep_num).transpose(1,2)
        if self.Flash:
            attn_output = F.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=None, dropout_p=self.attn_dropout,is_causal=True)
        else:
            scores = torch.matmul(xq, xk.transpose(-2, -1)) / (math.sqrt(self.head_dim))
            scores+= self.mask[:, :, :seqlen, :seqlen]
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, xv)
        attn_output = attn_output.transpose(1, 2).reshape(bs, seqlen, -1)
        attn_output = self.o_proj(attn_output)
        attn_output = self.res_dropout(attn_output)
        # print(f"attn_output = {attn_output}")
        return attn_output, past_kv
    

from transformers.activations import ACT2FN
from torch import nn
class FeedForward(nn.Module):
    def __init__(self,config:MinimindConfig):
        super(FeedForward,self).__init__()
        self.embed_dim = config.embed_dim
        self.ffn_dim = config.ffn_dim
        self.ffn_dropout = config.ffn_dropout
        self.act_fn = config.act_fn
        self.gate = nn.Linear(self.embed_dim, self.ffn_dim)
        self.up_proj = nn.Linear(self.embed_dim,self.ffn_dim)
        self.down_proj = nn.Linear(self.ffn_dim, self.embed_dim)
        self.dropout = nn.Dropout(self.ffn_dropout)
        self.act_fn = ACT2FN[self.act_fn]
    def forward(self,x):
        return self.down_proj(self.dropout(self.act_fn(self.gate(x)) * self.up_proj(x)))
    

class Minimind_Block(nn.Module):
    def __init__(self,layer_id,config:MinimindConfig):
        super(Minimind_Block,self).__init__()
        self.layer_id = layer_id
        self.embed_dim = config.embed_dim
        self.head_num = config.head_num
        self.kv_head_num = config.kv_head_num
        self.ffn_dim = config.ffn_dim
        self.attn_dropout = config.attn_dropout
        self.res_attn_dropout = config.attn_res_dropout
        self.ffn_dropout = config.ffn_dropout
        self.Flash = config.Flash
        self.max_seqlen = config.max_seqlen
        self.act_fn = config.act_fn
        
        self.attention = GroupQueryAttention(config)
        self.rmsnorm1 = RMSNorm(config)
        self.ffn = FeedForward(config)
        self.rmsnorm2 = RMSNorm(config)
    def forward(self, x, pos_cis=None, past_key_value=None, use_cache=False):
        # norm1
        x = self.rmsnorm1(x)
        # attention
        attn_output, past_kv = self.attention(x, pos_cis=pos_cis, past_key_value=past_key_value, use_cache=use_cache)
        # residual connection
        x = x + attn_output
        # norm2
        x = self.rmsnorm2(x)
        # feed forward
        ffn_output = self.ffn(x)
        # residual connection
        x = x + ffn_output
        # print(f"Block {self.layer_id} output : {x}")
        return x, past_kv

class Minimind_Dense(nn.Module):
    def __init__(self,config:MinimindConfig):
        super(Minimind_Dense,self).__init__()
        self.blocks = nn.ModuleList([
            Minimind_Block(layer_id, config)
            for layer_id in range(config.block_num)
        ])
    def forward(self, x, pos_cis=None, past_key_values=None, use_cache=False):
        if past_key_values is None:
            past_key_values = [None] * self.block_num
        for i, block in enumerate(self.blocks):
            x, past_kv = block(x, pos_cis=pos_cis, past_key_value=past_key_values[i], use_cache=use_cache)
            if use_cache:
                past_key_values[i] = past_kv
        return x, past_key_values


from transformers import PreTrainedModel,GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

class MinimindForCausalLM(PreTrainedModel,GenerationMixin):
    config_class = MinimindConfig
    base_model_prefix = "minimind"
    def __init__(self,params:MinimindConfig = None):
        self.params = params if params is not None else MinimindConfig()
        super(MinimindForCausalLM,self).__init__(self.params)
        self.embed = Embed(config=self.params)
        self.rmsnorm = RMSNorm(config=self.params)
        self.minimind_dense = Minimind_Dense(config=self.params)
        self.lm_head = nn.Linear(params.embed_dim, params.vocab_size)
        ## 临时属性
        pos_cis = precompute_pos_cis(embed_dim=params.head_dim, max_seqlen=params.max_seqlen)
        self.register_buffer('pos_cis', pos_cis,persistent=False)
        self.OUT = CausalLMOutputWithPast()
    
    def forward(self,input_ids=None,
                past_key_values=None,
                use_cache = False,
                **args):
        past_key_values = past_key_values if past_key_values is not None else [None] * self.params.block_num
        start_pos = args.get('start_pos', 0)
        hidden_states = self.embed(input_ids)
        pos_cis = self.pos_cis[start_pos:start_pos+hidden_states.size(1)]
        past_kvs = []
        ### 
        # print("######FORWARD GENERATION######")
        # print(f"input_ids shape: {input_ids.shape if input_ids is not None else 'None'}")
        for i, block in enumerate(self.minimind_dense.blocks):
            hidden_states, past_kv = block(hidden_states, pos_cis=pos_cis, past_key_value=past_key_values[i], use_cache=use_cache)
            if use_cache:
                past_kvs.append(past_kv)
            else:
                past_kvs.append(None)
            # print(f"Block {i+1}/{self.params.block_num} processed, hidden_states shape: {hidden_states.shape}")
        hidden_states = self.rmsnorm(hidden_states)
        logits = self.lm_head(hidden_states)
        # print("######FORWARD GENERATION END######")
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('past_key_values', past_kvs)
        self.OUT.__setitem__('hidden_states', hidden_states)
        return self.OUT
    
    # 推理函数（包含top-p或者top-k等策略）
    @torch.inference_mode()
    def generate(self,  input_ids,  eos_token_id=1,  max_new_tokens=512,  temperature=0.75,  top_p=0.90, 
                stream=False,  rp=1,  use_cache=True,  pad_token_id=3,  **args):
        # 流式生成 （返回生成器, 自由控制输出）
        if stream:
            return self._stream(input_ids,  eos_token_id,  max_new_tokens,  temperature,  top_p,  rp,  use_cache,  **args)
        # 直接生成 （一步到位）
        generated = []
        for i in range(input_ids.size(0)):
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
            out = self._stream(non_pad,  eos_token_id,  max_new_tokens,  temperature,  top_p,  rp,  use_cache,  **args)
            tokens_list = [tokens[:,  -1:] for tokens in out]
            print(f'new tokens list :{tokens_list}\n')
            gen = torch.cat(tokens_list,  dim=-1) if tokens_list else non_pad
            full_sequence = torch.cat([non_pad,  gen],  dim=-1)
            generated.append(full_sequence)
        max_length = max(seq.size(1) for seq in generated)
        generated = [
            torch.cat([seq,  torch.full((1,  max_length - seq.size(1)),  pad_token_id,  dtype=seq.dtype,  device=seq.device)], dim=-1) 
            for seq in generated
        ]
        return torch.cat(generated,  dim=0)

    # 流式输出
    def _stream(self,  input_ids,  eos_token_id,  max_new_tokens,  temperature,  top_p,  rp,  use_cache,  **args):
        start,  first_seq,  past_kvs = input_ids.shape[1],  True,  None
        new_token_idx = 0 #  new token 计数器
        while input_ids.shape[1] < max_new_tokens - 1:
            print(f'gernerating new token: idx = {start + new_token_idx}')
            if first_seq or not use_cache: # 若第一次生成序列 or 无 KV Cache,  每次生成传入整个 token id 序列
                out,  first_seq = self(input_ids,  past_key_values=past_kvs,  use_cache=use_cache,  **args),  False
            else: # 若非第一次生成 and 有 KV Cache, 每次传入最后一个 token id 与 KV Cache 进行推理加速
                out = self(input_ids[:,  -1:],  past_key_values=past_kvs,  use_cache=use_cache, 
                           start_pos=input_ids.shape[1] - 1,  **args)
            logits,  past_kvs = out.logits[:,  -1,  :],  out.past_key_values # logits.shape: (batch_size,  seq_len,  vocab_size), 获取最后一位 logits
            logits[:,  list(set(input_ids.tolist()[0]))] /= rp # 对生成 token 进行惩罚, 降低后续重复生成几率
            logits /= (temperature + 1e-9) # 调整温度, 控制生成多样性
            if top_p is not None and top_p < 1.0: # top-p 采样
                sorted_logits,  sorted_indices = torch.sort(logits,  descending=True,  dim=-1)
                sorted_probs = F.softmax(sorted_logits,  dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs,  dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:,  1:] = sorted_indices_to_remove[:,  :-1].clone()
                sorted_indices_to_remove[:,  0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1,  sorted_indices,  sorted_indices_to_remove)
                logits[indices_to_remove] = -float('1e9')
            input_ids_next = torch.multinomial(F.softmax(logits,  dim=-1),  num_samples=1) # 从保留的 token 中采样
            input_ids = torch.cat((input_ids,  input_ids_next),  dim=1)
            new_token_idx += 1
            yield input_ids[:,  start:]
            if input_ids_next.item() == eos_token_id:
                break


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer_path= "./"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(tokenizer)
    # 写一个虚拟的小的数据集，只有两条数据的集
    data=[
        {'text':'<|im_start|>鉴别一组中文文章的风格和特点<|im_end|> '
    },
        {'text':'<|im_start|>根据输入的内容，编写一个类别标签。<|im_end|> <|im_start|>'
    }
    ]
    config = MinimindConfig()
    model2 = MinimindForCausalLM(config)
    for i in range(2):
        print(data[i]['text'])

    # 接下来将该data的内容利用tokenizer编码
    input_texts = [item['text'] for item in data]
    input_test3 = tokenizer(input_texts, padding='max_length', truncation=True, max_length=512,return_tensors='pt')
    print(input_test3)
    print(input_test3['input_ids'].shape)  # 输出形状应为 (batch_size, sequence_length)
    outputs4 = model2.generate(input_ids=input_test3['input_ids'], max_new_tokens=300, use_cache=True,)
    outputs4_1 = model2.generate(input_ids=input_test3['input_ids'], max_new_tokens=512, use_cache=True,)
    print(outputs4.shape)  # 输出形状应为 (batch_size, sequence_length +
    word1=tokenizer.decode(outputs4[0], skip_special_tokens=True)  # 解码第一个batch的输出
    word2=tokenizer.decode(outputs4_1[0], skip_special_tokens=True)  # 解码第二个batch的输出
    print(word1)
    print(word2)