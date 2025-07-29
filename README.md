# minimind-v-learn
从0构建Minimind-v

# 配置环境(与minimind相同)
## 第0步
`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`


# 项目目的
1.学习从0构建自己的LLM项目
2.学习最新的MOE原理
3.学习transformers库
4.学习DPO和GRPO等业界post-train的原理
5.学习优秀的项目构成


# 项目目录说明
- data 存放数据集
- dataset 存放数据集的定义文件
- model 存放模型文件
- trainer 存放trainer定义文件
- inference 存放推理文件
- pretrained 存放训练得到的ckpt文件
- scripts 存放一些杂项脚本
- utils 存放一些通用类
- images 存放宣发图片和讲解图片

在搭起项目骨架后，使用gitignore文件忽略data的同步

# 学习路径
LLM主要分为tokenizer,model 两个部分的组成和训练<br>
训练主要有pretrain,finetune,post-train 三个训练阶段<br>
我主要借鉴[minimind构建记录](https://github.com/Nijikadesu/breakdown-minimind)<br>
来分阶段实现tokenizer的训练、model的训练，以及最终的部署

# 开始之前需要下载数据集
我们在训练tokenizer和pretrain阶段，共用同一个处理好的数据集放在data文件夹下
[modelscope下载](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)

# step1-tokenizer
由于tokenizer并非Model的一部分，所以训练tokenizer的过程我们放在scripts下
