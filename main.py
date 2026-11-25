import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 函数作用：缩放点积注意力数学公式
def scaled_dot_product_attention(q,k,v,mask=None):
#读取向量q的最后一个维度
    d_k = q.size(-1)
#先做k矩阵最后两维的转置，再与q向量做矩阵乘法，然后除以d_k的平方根做缩放，防止方差过大，softmax函数极值过大影响学习效率
    score = torch.matmul(q,k.transpose(-2,-1))/torch.sqrt(torch.tensor(d_k,dtype=torch.float32))
# 判断mask是否为None
    if mask is not None:
        # 把mask等于0的地方替换为负无穷，使softmax函数为0，屏蔽信息
        score = score.masked_fill(mask==0,-1e9)
    # 在score的最后一个维度执行softmax归一化
    attention_weight = F.softmax(score,dim=-1)
    #用归一化后的矩阵与V做乘积得到加权求和的输出特征
    output = torch.matmul(attention_weight,v)
    return output,attention_weight
# 定义多头注意力模块
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_head):
        super().__init__()
        assert d_model % num_head ==0 ,"d_model 必须被num_head整除"
        #给类实例绑定三个整数属性
        self.d_model = d_model
        self.num_head = num_head
        self.d_k = d_model//num_head
        #实例化四个d_model维度输入输出的线性层
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_o = nn.Linear(d_model,d_model)
    #函数作用：把单头大维度特征拆分成多头并行计算的4维布局
    def split_head(self,x):
        #读取x张量第0维的长度，也就是批量大小batch_size
        batch_size = x.size(0)
        #先调用view做维度重塑，在调用transpose调换下标1，2的维度
        return x.view(batch_size,-1,self.num_head,self.d_k).transpose(1,2)
    #函数作用：多头注意力向前传播主逻辑
    def forward(self,q,k,v,mask=None):
        #读取输入q的第0维，获取当前批量大小
        batch_size = q.size(0)
#         先把qkv通过线形层映射，在调用split_head拆分为多头格式
        q = self.split_head(self.w_q(q))
        k = self.split_head(self.w_k(k))
        v = self.split_head(self.w_v(v))
#         调用纯函数scaled_dot_product_attention，传入拆分后的qkv
        attn_output,attention_weight = scaled_dot_product_attention(q,k,v,mask)
#         先交换维度转置回（BSHD_K），调用contiguous生成内存连续副本，再用view合并两个维度变回d_model
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)
#         把拼接好的特征传入全连接层，完成跨头特征融合
        output = self.w_o(attn_output)
        return output,attention_weight
#类作用：生成固定的正余弦位置编码，不需要参与训练
class PositionEncoding(nn.Module):
    def __init__(self,d_model,max_len=50000):
        super().__init__()
#         初始化形状为（maxlen,dmodel）的全零向量pe
        pe = torch.zeros(max_len,d_model)
#         生成0到maxlen-1的下标向量，新增维度变成列向量，形状（maxlen,1）
        position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
#         用对数指数等价变换计算论文定义的分母项div_term
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-torch.log(torch.tensor(10000.0))/d_model))
#         所有奇偶数下标余弦正弦
        pe[:,0::2] = torch.sin(position *div_term)
        pe[:,1::2] = torch.cos(position *div_term)
#         在pe向量前面新增一个batch维度
        pe = pe.unsqueeze(0)
#         把pe张量注册为pytorch的buffer缓存，不会被优化器更新
        self.register_buffer('pe',pe)
# 函数作用：把预生成的位置编码加到输入特征的后面
    def forward(self,x):
#         截取pe的前seq_len个位置与x做广位相加
        x = x+self.pe[:,:x.size(1),:]
        return x
# 类作用：定义transformer层里的独立前馈网络
class FeeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
#         定义第一层全连接，把d_model映射到高维d_ff,第二层把d_ff映射回d_model,第三层定义dropout层，训练时随机丢弃部分神经元输出
        self.linear1 = nn.Linear(d_model,d_ff)
        self.linear2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)
#   函数作用：完整向前流程，线性层1->relu->dropout->线性层2
    def forward(self,x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
#  类作用：Transformer单个编码器层，由自注意力和FFN两个子层组成
class EncoderLayer(nn.Module):
    def __init__(self,d_model,num_head,d_ff,dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model,num_head)
        self.ffn = FeeedForward(d_model,d_ff,dropout=0.1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,x,src_mask=None):
        # 执行自注意力向前计算，获取输出结果,三个x分别是qkv
        attn_out,_ = self.self_attn(x,x,x,src_mask)
#         第一组残差加归一化
        x = self.norm1(x+self.dropout1(attn_out))
#         执行前馈网络向前计算
        ffn_out = self.ffn(x)
#       二次残差相加+DROPOUT+层归一化
        x = self.norm2(x+self.dropout2(ffn_out))
        return x
# 类作用：Transformer单个解码器，有三个子层构成
class DecoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout=0.1):
        super().__init__()
        self.masked_self_atten = MultiHeadAttention(d_model,num_heads)
        self.cross_atten = MultiHeadAttention(d_model,num_heads)
        self.ffn = FeeedForward(d_model,d_ff,dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    #     src_mask只在编码器起作用，唯一作用是遮挡输入里的padding填充占位符，不遮挡任何一个有效token。tgt_mask是解码器掩码，做两件事第一遮挡padding填充占位符，第二强制遮挡未来位置
    def forward(self,x,enc_output,src_mask=None,tgt_mask=None):
#       执行带因果掩码的自注意力向前计算
        attn_out1,_ = self.masked_self_atten(x,x,x,tgt_mask)
#         第一组残差加归一化
        x = self.norm1(x+self.dropout1(attn_out1))
# 执行交叉注意力向前计算，Q来自解码器，kv来自编码器输出
        attn_out2,_ = self.cross_atten(x,enc_output,enc_output,src_mask)
        x = self.norm2(x+self.dropout2(attn_out2))
        ffn_out = self.ffn(x)
        x = self.norm3(x+self.dropout3(ffn_out))
        return x
# 类作用：完整端到端简易Transformer模型，整合所有编码器解码器组件
class SimpleTransformer(nn.Module):
    def __init__(self,src_vocab_size, tgt_vocab_size, d_model=64, num_heads=2, d_ff=256, num_layers=2, max_len=500,dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size,d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size,d_model)
        self.pos_encoding = PositionEncoding(d_model,max_len)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model,num_heads,d_ff,dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model,num_heads,d_ff,dropout) for _ in range(num_layers)])
#         执行动作：实例化最终输出投影层，把d_model特征映射回目标此表为度
        self.linear = nn.Linear(d_model,tgt_vocab_size)
    def forward(self,src,tgt,src_mask=None,tgt_mask=None):
#         执行操作;原输入经过词嵌入+乘sqrt(d_model)做幅值缩放
        x = self.src_embedding(src)*torch.sqrt(torch.tensor(self.d_model,dtype=torch.float32))
#           执行操作：给嵌入特征加上位置编码
        x = self.pos_encoding(x)
#       执行操作：把特征值依次传入每一个编码器层，逐层提取上下文特征
        for layer in self.encoder_layers:
            x = layer(x,src_mask)
        enc_output = x
        x = self.tgt_embedding(tgt)*torch.sqrt(torch.tensor(self.d_model,dtype=torch.float32))
        x = self.pos_encoding(x)
        for layer in self.decoder_layers:
            x = layer(x,enc_output,src_mask,tgt_mask)
        output = self.linear(x)
        return output
# 【执行动作】定义测试用的超参数，d_model_test=64，max_len_test=100
d_model_test = 64
max_len_test = 100
# 【执行动作】实例化位置编码模块pe_layer
pe_layer = PositionEncoding(d_model_test, max_len_test)
# 【执行动作】把pe张量从计算图分离，删掉多余batch维度，转成numpy数组
# 【实际输出】pe_matrix形状为(100, 64)的二维numpy数组，所有元素都是正弦余弦计算的合法数值
pe_matrix = pe_layer.pe.squeeze(0).detach().numpy()

# 【执行动作】创建尺寸为10*6英寸的画布对象
plt.figure(figsize=(10, 6))
# 【执行动作】在画布上绘制seaborn热力图，配色用viridis渐变配色
sns.heatmap(pe_matrix, cmap='viridis')
# 【执行动作】设置X轴标签文字为Dimension (d_model)
plt.xlabel("Dimension (d_model)")
# 【执行动作】设置Y轴标签文字为Position
plt.ylabel("Position")
# 【执行动作】设置图表标题文字
plt.title("Sinusoidal Positional Encoding Heatmap")
# 【执行动作】弹出matplotlib窗口，显示绘制好的位置编码热力图
plt.show()
