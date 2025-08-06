from transformers import PretrainedConfig

class MinimindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self,
                # tokenizer相关 
                 vocab_size = 6400,
                 model_max_length=32768,
                 eos_token_id = 1,
                 bos_token_id = 2,
                # attn 参数
                max_seqlen=1024,
                embed_dim=512,
                head_num=8,
                kv_head_num=4,
                attn_dropout=0.1,
                attn_res_dropout=0.1,
                Flash=False,
                # ffn 参数,
                ffn_dim=2048,
                act_fn = "silu",
                ffn_dropout=0.1,
                # norm 参数
                rmsnorm_eps=1e-6,
                # block参数,
                block_num=8,
                ## MOE 参数
                # 其他参数
                **kwargs
                ):
        super(MinimindConfig,self).__init__(**kwargs)
        # tokenizer相关
        self.vocab_size = vocab_size
        self.model_max_length = model_max_length
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id

        # 结构参数
        self.block_num = block_num
        self.embed_dim = embed_dim
        self.head_num = head_num
        self.head_dim = embed_dim // head_num
        self.kv_head_num = kv_head_num
        self.ffn_dim = ffn_dim
        self.attn_res_dropout = attn_res_dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.Flash = Flash
        self.max_seqlen = max_seqlen
        self.norm_eps = rmsnorm_eps
        self.act_fn = act_fn
        ## MOE 参数
