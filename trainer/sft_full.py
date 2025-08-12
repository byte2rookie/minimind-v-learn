# 这里只展示完整的ddp改造后的代码全貌
import math
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from contextlib import nullcontext
import time
import sys
import os
from transformers import AutoTokenizer

# 获取当前 notebook 所在目录（trainer/）
current_dir = os.path.dirname(os.path.abspath("__file__"))  # 注意 Jupyter 中可能需要调整
# 或者直接写死路径
current_dir = "/data/zyp/jinbu/ZZY/minimind-v-learn/trainer"
# 上一级目录就是项目根目录，拼接 model 路径
model_dir = os.path.join(os.path.dirname(current_dir), "model")
sys.path.append(model_dir)
# 现在可以用绝对导入
from model import MinimindForCausalLM, MinimindConfig

from pathlib import Path
# 项目根目录：/data/zyp/jinbu/ZZY/minimind-v-learn
root_dir = Path("/data/zyp/jinbu/ZZY/minimind-v-learn")
# 将根目录添加到 Python 可搜索路径
sys.path.append(str(root_dir))
from dataset.lm_dataset import PretrainDataset, SFTDataset



class sft_args:
    out_dir = "../out"
    epochs = 1
    batch_size = 32
    learning_rate = 5e-4
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = "bfloat16"
    use_wandb = False
    wandb_project = "MiniMind-SFT"
    num_workers = 1
    ddp = False
    accumulation_steps = 8
    grad_clip = 1.0
    warmup_iters = 0
    log_interval = 100
    save_interval = 100
    local_rank = -1
    embed_dim = 512
    block_num = 8
    max_seqlen = 1024
    use_moe = False
    # data_path = "../data/sft_data.jsonl"  # toy_dataset  
    data_path = "../data/sft_512.jsonl" #full_dataset


def Logger(content):
    # ddp改造后如果是ddp模式就不进行logger打印
    if train_args.ddp and train_args.local_rank != 0:
        return
    print(content)

def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    model = MinimindForCausalLM(lm_config).to(train_args.device)
    moe_path = '_moe' if train_args.use_moe else ''
    ckp = f'{train_args.save_dir}/pretrain_{lm_config.embed_dim}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=train_args.device)
    model.load_state_dict(state_dict, strict=False)
    Logger(f'加载模型参数 {ckp}')
    Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer

# ddp的核心库
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
# 初始化分布式环境
def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


def get_lr(current_step, total_steps, lr):
    # 余弦退火学习率调度
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

def train_epoch(epoch):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(train_args.device)
        Y = Y.to(train_args.device)
        loss_mask = loss_mask.to(train_args.device)

        lr = get_lr(epoch * iter_per_epoch + step, train_args.epochs * iter_per_epoch, train_args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            # print(f"X = {X}")
            # print(f"res={res}")
            # if torch.isnan(res.logits).any() or torch.isinf(res.logits).any():
            #     Logger(f"Warning: logits contains NaN/Inf at step {step}")
            #     # 打印logits的范围，辅助排查
            #     Logger(f"logits range: {res.logits.min().item()} ~ {res.logits.max().item()}")
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            # print(f"loss_mask.sum(): {loss_mask.sum()}")
            loss = (loss * loss_mask).sum() / loss_mask.sum() # 这里的loss 是有效非pad的token的平均loss
            # loss += res.aux_loss
            loss = loss / train_args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % train_args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)  # 清空梯度，为下一个iter做准备

        if step % train_args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    train_args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * train_args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))


        if (step + 1) % train_args.save_interval == 0 and (not ddp or dist.get_rank() == 0):  # 仅在非ddp或主进程中保存模型，防止重复保存
            model.eval()
            moe_path = '_moe' if train_args.use_moe else ''
            ckp = f'{train_args.save_dir}/sft_full_{config.embed_dim}{moe_path}.pth'
            Logger(f'保存模型到 {ckp}')
            # 增加一个ddp的判断，因为ddp模式下模型被封装在 model.module 中
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            model.train()

if __name__ == "__main__":
    # 首先是训练参数设定
    train_args = sft_args()
    train_args.save_dir = os.path.join(train_args.out_dir)
    # 确保输出目录存在
    os.makedirs(train_args.save_dir, exist_ok=True)
    # 初始化模型配置
    config = MinimindConfig(
        embed_dim=train_args.embed_dim,
        block_num=train_args.block_num,
        max_seqlen=train_args.max_seqlen,
    )
    print(f'查看工作设备 {train_args.device}')

    # runtime初始化
    device_type = "cuda" if "cuda" in train_args.device else "cpu"
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast() # 在 cuda 上启动混精度训练，否则空白上下文

    tokens_per_iter = train_args.batch_size * train_args.max_seqlen # 每次迭代处理的 token 数
    
    # ddp初始化
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"
    # 如果是ddp模式，进行seed设定
    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)
    # 对每张卡的seed进行偏移
    if ddp:
        init_distributed_mode()
        train_args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        # 同时设置 CUDA 的随机种子
        torch.cuda.manual_seed(base_seed + rank)
    # 加载ds
    model, tokenizer = init_model(config)
    print(model)
    print(tokenizer)
    train_ds = SFTDataset(
        data_path=train_args.data_path,
        tokenizer=tokenizer,
        max_seqlen=train_args.max_seqlen,
    )
    train_sampler = DistributedSampler(train_ds) if ddp else None   #ddp模式下使用分布式采样器，确保采样均匀   
    train_loader = DataLoader(
        train_ds,
        batch_size=train_args.batch_size,
        shuffle=False,
        num_workers=train_args.num_workers,
        pin_memory=True,
        drop_last=False,
        sampler=train_sampler,  # 如果是ddp模式，使用分布式采样器
    )
    iter_per_epoch = len(train_loader) # 计算每个 epoch 的迭代次数
    Logger(f'使用分布式采样器: {train_sampler is not None}')

    # 设置精度和优化器
    scaler = torch.cuda.amp.GradScaler(enabled=(train_args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=train_args.learning_rate)

    # 临时变量忽略，封装ddp模型
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])
        Logger(f"ddp模式，local_rank: {ddp_local_rank}, global_rank: {dist.get_rank()}, world_size: {dist.get_world_size()}")
    
    # 开始训练
    for epoch in range(train_args.epochs):
        train_epoch(epoch)

# 运行命令为 torchrun --nproc_per_node=8 sft_full.py
