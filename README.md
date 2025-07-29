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
