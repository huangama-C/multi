# =============================================================
# Multi30k 德→英 小型机器翻译 训练脚本
# 环境: PyTorch 2.13.0+cu132, RTX 5070 Laptop (sm_120)
# =============================================================
import os
import math
import time
import random
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# 模型代码从你自己的 main.py 里导入（确保 main.py 和本文件同目录）
from main import SimpleTransformer

# -------- 解决 Windows 平台 OpenMP 库冲突导致的程序崩溃，Windows 专属兼容配置 --------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ================= 超参数配置 【所有可调节的训练旋钮集中管理，改一处全生效】 =================
CFG = {
    "data_dir": r"E:\Multi\data\multi",  # 输入路径：存放 train.de/train.en 等6个数据文件的文件夹
    "save_dir": r"E:\code\runs",  # 输出路径：训练完的模型权重、曲线图、历史数据都存这里
    "train_size": 29000,  # 训练集句对数量：取前8000句，把总训练时间控制在几分钟内
    "val_size": 1000,  # 验证集句对数量：取前1000句，评估模型泛化能力
    "min_freq": 2,  # 最小词频：出现次数小于2的词不进正式词表，统一归为<unk>，避免词表过于稀疏
    "max_len": 50,  # 最大句长：德语/英语单句超过50个词直接丢弃，防止超长句撑爆GPU显存
    "d_model": 256,  # Transformer 隐层维度：所有注意力层、前馈层的特征维度都是256
    "num_heads": 8,  # 注意力头数：8头并行注意力，256能被8整除，每个头分到32维特征
    "d_ff": 1024,  # 前馈网络中间层维度：Transformer 论文中默认是 d_model*4，256*4=1024
    "num_layers": 3,  # 编码器/解码器层数：各3层，小任务不需要6层，减少参数量防止过拟合
    "dropout": 0.1,  # Dropout 丢弃概率：训练时随机丢掉10%神经元，防止模型过拟合
    "batch_size": 64,  # 批次大小：一次同时喂给模型64个句对，平衡显存占用和训练效率
    "epochs": 12,  # 训练轮数：把8000句完整过12遍
    "warmup": 2000,  # Noam 学习率预热步数：前2000步学习率从0线性上升到峰值
    "label_smooth": 0.1,  # 标签平滑系数：不强制模型100%押注正确词，给所有其他词留0.1/N的概率空间，防过拟合
    "clip": 1.0,  # 梯度裁剪阈值：把梯度的最大L2范数限制在1.0以内，防止反向传播时梯度爆炸
    "seed": 42,  # 全局随机种子：固定为42，让每次运行的shuffle、参数初始化结果完全一致
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # 自动判断设备：有RTX 5070就用GPU跑，否则退回CPU
}

# 如果 save_dir 文件夹不存在就自动创建，exist_ok=True 表示已存在也不报错，避免重复运行抛异常
os.makedirs(CFG["save_dir"], exist_ok=True)
# 固定Python层面随机种子，让 random.sample、random.shuffle 结果可复现
random.seed(CFG["seed"])
# 固定PyTorch层面随机种子，让模型参数初始化、Dropout随机丢弃结果完全固定
torch.manual_seed(CFG["seed"])

# 定义翻译任务的4个专用特殊符号，纯文本词表里不会出现
PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIALS = [PAD, UNK, BOS, EOS]


# ================= 1. 读取平行语料数据 =================
def read_pairs(de_path, en_path, limit=None):
    """
    读取德英平行语料，返回分词后的句对列表
    参数 de_path: 德语语料文件路径；en_path: 英语语料文件路径；limit: 最多读取多少对，控制数据集规模
    返回: [(德语分词列表, 英语分词列表), ...]
    """
    # 同时打开两个语料文件，指定utf-8编码避免德文特殊字符乱码
    with open(de_path, encoding="utf-8") as f_de, open(en_path, encoding="utf-8") as f_en:
        # 逐行读德语文件，strip()去掉每行首尾的换行符和多余空格，得到德文句子列表
        de_lines = [l.strip() for l in f_de]
        # 逐行读英语文件，同理得到英文句子列表
        en_lines = [l.strip() for l in f_en]

    pairs = []
    # zip把两个句子列表按索引一一配对，每次取出一句德语和一句对应的英语
    for de, en in zip(de_lines, en_lines):
        # 转全小写，按空格切分得到德语词列表，比如"Ein Hund" → ["ein", "hund"]
        de_tok = de.lower().split()
        # 同理得到英语词列表
        en_tok = en.lower().split()

        # 过滤掉全空的无效句
        if len(de_tok) < 1 or len(en_tok) < 1:
            continue
        # 过滤掉超过最大句长50的超长句
        if len(de_tok) > CFG["max_len"] or len(en_tok) > CFG["max_len"]:
            continue

        # 把合法的分词句对存入列表
        pairs.append((de_tok, en_tok))
        # 达到设定的limit数量就提前break，不用继续读后面的文件内容
        if limit and len(pairs) >= limit:
            break
    return pairs


print("📥 正在读取数据 ...")
# 读取训练集，限制最多8000对
train_pairs = read_pairs(
    os.path.join(CFG["data_dir"], "train.de"),
    os.path.join(CFG["data_dir"], "train.en"),
    CFG["train_size"],
)
# 读取验证集，限制最多1000对
val_pairs = read_pairs(
    os.path.join(CFG["data_dir"], "val.de"),
    os.path.join(CFG["data_dir"], "val.en"),
    CFG["val_size"],
)
# 打印实际读取到的有效句对数量
print(f"   训练句对: {len(train_pairs)}  验证句对: {len(val_pairs)}")


# ================= 2. 构建词表：实现 词 ↔ 数字编号 的映射 =================
class Vocab:
    """
    自定义词表类：把文本词映射成整数编号，反向也能从编号转回词
    """

    def __init__(self, tokens_list, min_freq=1):
        # 初始化计数器，用来统计所有词的出现次数
        counter = Counter()
        for tokens in tokens_list:
            counter.update(tokens)  # 把当前句子的所有词倒进计数器自动累加计数

        # 初始化"编号→词"的列表（index to string）：前4个位置固定放4个特殊符号
        self.itos = list(SPECIALS)
        # 遍历按词频从高到低排序的真实词
        for w, c in counter.most_common():
            # 仅保留出现频次≥min_freq的词，且不要重复加入已有的特殊符号
            if c >= min_freq and w not in self.itos:
                self.itos.append(w)

        # 生成反向字典"词→编号"（string to index），enumerate自动把列表下标作为编号
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, tokens):
        """
        把一个词列表转成编号列表，遇到不在词表里的词自动映射成<unk>的编号
        实例：tokens = ["ein", "hund", "xyz"] → 返回 [45, 88, 1] （1是<unk>的编号）
        """
        return [self.stoi.get(w, self.stoi[UNK]) for w in tokens]

    def __len__(self):
        # 返回词表总大小，决定Transformer中Embedding层的行数
        return len(self.itos)


# 基于训练集的德语分词列表，构建源语言（德语）词表
src_vocab = Vocab([p[0] for p in train_pairs], CFG["min_freq"])
# 基于训练集的英语分词列表，构建目标语言（英语）词表
tgt_vocab = Vocab([p[1] for p in train_pairs], CFG["min_freq"])
# 打印两个词表的大小
print(f"   源词表(德): {len(src_vocab)}  目标词表(英): {len(tgt_vocab)}")


# ================= 3. 自定义Dataset和collate函数，生成可批量送入模型的张量 =================
class MTDataset(Dataset):
    """
    自定义数据集类，继承PyTorch的Dataset规范接口
    """

    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.data = []
        for de, en in pairs:
            # 德语句首尾添加<bos>和<eos>，转成编号序列：[bos_id] + 句子编号 + [eos_id]
            # 实例："ein hund" → [2, 45, 88, 3] （2是bos，3是eos）
            src_ids = [src_vocab.stoi[BOS]] + src_vocab.encode(de) + [src_vocab.stoi[EOS]]
            # 英语句同理处理成目标侧编号序列
            tgt_ids = [tgt_vocab.stoi[BOS]] + tgt_vocab.encode(en) + [tgt_vocab.stoi[EOS]]
            self.data.append((src_ids, tgt_ids))

    def __len__(self):
        # 规定Dataset的接口，返回总数据条数，给DataLoader做迭代次数参考
        return len(self.data)

    def __getitem__(self, idx):
        # 规定Dataset的接口，支持用 dataset[5] 直接取第5条数据
        return self.data[idx]


def collate_batch(batch):
    """
    自定义组批函数：把一批长度不同的编号序列，补齐成统一长度的张量矩阵
    实例：batch里有3条句长分别为3、5、4 → 补齐到最长5，全部填充<pad>到等长
    """
    # zip(*batch) 解包：把64个(源,目标)句对，拆成64个源序列、64个目标序列两组
    src_batch, tgt_batch = zip(*batch)

    # 求出当前批次里最长的德语句长度
    max_src = max(len(s) for s in src_batch)
    # 求出当前批次里最长的英语句长度
    max_tgt = max(len(t) for t in tgt_batch)

    # 先初始化一个形状为 (batch_size, max_src) 的全<pad>张量
    pad_src = torch.full((len(batch), max_src), src_vocab.stoi[PAD], dtype=torch.long)
    # 同理初始化目标侧全<pad>张量
    pad_tgt = torch.full((len(batch), max_tgt), tgt_vocab.stoi[PAD], dtype=torch.long)

    # 遍历每一条句对，把真实的编号序列填进对应的行，剩下的位置保留<pad>
    for i, (s, t) in enumerate(zip(src_batch, tgt_batch)):
        pad_src[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        pad_tgt[i, :len(t)] = torch.tensor(t, dtype=torch.long)

    # 返回补齐后的两个批量张量，形状都是 (B, S)
    return pad_src, pad_tgt


# 实例化训练集Dataset
train_ds = MTDataset(train_pairs, src_vocab, tgt_vocab)
# 实例化验证集Dataset
val_ds = MTDataset(val_pairs, src_vocab, tgt_vocab)
# 训练集DataLoader：batch_size=64，shuffle=True每轮打乱数据顺序，避免模型记住固定顺序
train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True, collate_fn=collate_batch)
# 验证集DataLoader：不需要打乱顺序，结果更稳定
val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False, collate_fn=collate_batch)


# ================= 4. 构造两种掩码：控制注意力能看到哪些位置 =================
def make_src_mask(src, pad_id):
    """
    生成源侧掩码：屏蔽源句中的padding位置，让编码器注意力不去看无意义的填充
    输入 src 形状 (B, S) → 输出 mask 形状 (B, 1, 1, S)
    实例：src = [2,45,88,3,0,0] → mask在非0位置为True，最后两个padding位置为False
    """
    # 逐元素判断：不是padding的位置返回True，是padding返回False，得到(B,S)的bool矩阵
    mask = (src != pad_id)
    # 连续升两个维度，从(B,S)变成(B,1,1,S)，后续注意力计算时靠广播自动适配所有头和查询位置
    return mask.unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt, pad_id):
    """
    生成目标侧掩码：同时实现两个功能——屏蔽padding + 因果掩码（解码器不能偷看未来的词）
    输入 tgt 形状 (B, T) → 输出 mask 形状 (B,1,T,T)
    """
    B, T = tgt.shape
    # 第一步：生成padding掩码，形状 (B,1,1,T)，非padding位置为True
    pad_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2)
    # 第二步：生成下三角因果掩码，形状(T,T)，第t行只有前t个位置为True
    # 实例：T=3 → [[1,0,0],[1,1,0],[1,1,1]]，表示生成第0个词看0号位置，生成第1个词看0和1号...
    causal = torch.tril(torch.ones(T, T, device=tgt.device)).bool()
    # 升维到(1,1,T,T)，支持批量广播
    causal = causal.unsqueeze(0).unsqueeze(0)
    # 两个bool矩阵逐元素与运算：同时满足"不是padding"和"是过去可看位置"，才返回True
    return pad_mask & causal


# ================= 5. 初始化模型、优化器、损失函数 =================
model = SimpleTransformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=CFG["d_model"],
    num_heads=CFG["num_heads"],
    d_ff=CFG["d_ff"],
    num_layers=CFG["num_layers"],
    max_len=CFG["max_len"] + 10,  # 留10的余量，防止位置编码边界溢出
    dropout=CFG["dropout"],
).to(CFG["device"])  # 把整个模型的参数搬到GPU上计算

# 打印模型总参数量，单位百万，帮助判断模型大小是否合理
print(f"🚀 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
# 打印实际使用的计算设备
print(f"🖥️  使用设备: {CFG['device']}")

# 初始化Adam优化器：lr先设1.0为占位值，实际每步由Noam公式动态更新
optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
# 初始化交叉熵损失函数：忽略padding位置的损失计算，带标签平滑
criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.stoi[PAD], label_smoothing=CFG["label_smooth"])


def noam_lr(step):
    """
    Transformer论文原版Noam学习率调度器：先线性升温后指数衰减
    前warmup步：lr随step线性增大 → 之后lr随step的-0.5次方缓慢下降
    设计目的：训练初期模型不稳定，用小学习率热身避免训飞，后期逐步降lr收敛
    """
    d_model = CFG["d_model"]
    warmup = CFG["warmup"]
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)


# ================= 6. 正式训练主循环 =================
# 初始化历史记录字典，存每轮的训练损失、验证损失、验证准确率，后续用于画图
history = {"train_loss": [], "val_loss": [], "val_acc": []}
global_step = 0  # 全局累计训练步数，跨epoch不重置，用于Noam学习率计算

for epoch in range(1, CFG["epochs"] + 1):
    model.train()  # 切换模型到训练模式：开启dropout、batchnorm的训练状态
    total_loss, n_batches = 0.0, 0  # 初始化本轮累计损失和批次计数器
    t0 = time.time()  # 记录本轮开始时间

    # 遍历训练集的所有批次
    for src, tgt in train_loader:
        # 把当前批次的源张量和目标张量搬到GPU上
        src = src.to(CFG["device"])
        tgt = tgt.to(CFG["device"])

        # Teacher Forcing错位技巧：输入去掉最后一个词，标签去掉第一个词
        # 实例：原始tgt = [bos, a, dog, eos]
        # tgt_in = [bos, a, dog] → 喂给解码器的输入
        # tgt_out = [a, dog, eos] → 模型要预测的标准答案
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        # 生成当前批次的两种掩码
        src_mask = make_src_mask(src, src_vocab.stoi[PAD])
        tgt_mask = make_tgt_mask(tgt_in, tgt_vocab.stoi[PAD])

        # 全局步数+1
        global_step += 1
        # 用Noam公式算出当前步的学习率
        lr = noam_lr(global_step)
        # 把新的lr手动写入优化器的所有参数组
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # 第一步：清空上一步残留的梯度，PyTorch默认梯度是累加的不清零会出错
        optimizer.zero_grad()
        # 第二步：模型前向传播，得到logits形状 (B, T-1, 目标词表大小)
        logits = model(src, tgt_in, src_mask, tgt_mask)
        # 第三步：计算损失，把三维的logits和二维的tgt_out拉平成CrossEntropy要求的形状
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        # 第四步：反向传播，自动微分计算所有参数的梯度
        loss.backward()
        # 第五步：梯度裁剪，把所有梯度的L2范数限制在1.0以内，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
        # 第六步：优化器执行参数更新，用梯度减去lr*梯度
        optimizer.step()

        # 累加当前步的损失，item()把tensor转成Python普通数字，防止在GPU上占内存
        total_loss += loss.item()
        n_batches += 1

    # 计算本轮训练的平均损失
    train_loss = total_loss / n_batches

    # ---- 验证阶段：不更新参数，只评估泛化能力 ----
    model.eval()  # 切换模型到评估模式：关闭dropout、固定batchnorm统计量
    val_loss, correct, total = 0.0, 0, 0  # 初始化验证损失、正确预测数、总有效token数
    with torch.no_grad():  # 上下文管理器：关闭梯度计算，省显存同时提速
        for src, tgt in val_loader:
            src = src.to(CFG["device"])
            tgt = tgt.to(CFG["device"])
            # 同样执行Teacher Forcing错位拆分
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask = make_src_mask(src, src_vocab.stoi[PAD])
            tgt_mask = make_tgt_mask(tgt_in, tgt_vocab.stoi[PAD])

            # 前向传播得到预测打分
            logits = model(src, tgt_in, src_mask, tgt_mask)
            # 累加验证损失
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            val_loss += loss.item()

            # argmax(-1)在词表维度取分数最大的索引，得到预测的编号序列
            preds = logits.argmax(-1)
            # 生成掩码：标记出哪些位置不是padding，只在真实词上算准确率
            mask = tgt_out != tgt_vocab.stoi[PAD]
            # 统计mask下预测等于答案的总个数
            correct += (preds[mask] == tgt_out[mask]).sum().item()
            # 统计mask下总有效token数
            total += mask.sum().item()

    # 计算本轮平均验证损失
    val_loss /= len(val_loader)
    # 计算本轮token级准确率：正确数/总数
    val_acc = correct / total

    # 把本轮结果写入历史记录
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    # 格式化打印本轮所有关键指标
    print(f"Epoch {epoch:2d}/{CFG['epochs']} | "
          f"train_loss={train_loss:.3f} | val_loss={val_loss:.3f} | "
          f"token_acc={val_acc:.3f} | lr={lr:.2e} | {time.time() - t0:.1f}s")

# 训练完成后，把所有必要信息打包成字典保存到磁盘
# 不单独只存model.state_dict，而是连词表和超参数一起存，避免推理时加载不匹配
torch.save({
    "model_state": model.state_dict(),  # 模型权重参数
    "src_vocab": src_vocab.itos,  # 源语言词表
    "tgt_vocab": tgt_vocab.itos,  # 目标语言词表
    "cfg": CFG,  # 训练时的超参数
}, os.path.join(CFG["save_dir"], "mt_model.pt"))
print("💾 模型已保存")

# ================= 7. 绘制Loss曲线和准确率曲线 =================
sns.set_style("whitegrid")  # 设置seaborn的白色网格风格，图更清晰
# 创建1行2列的子图画布，总大小12x4.5英寸
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
epochs_x = range(1, CFG["epochs"] + 1)  # x轴是1到12的轮数
# 左图：绘制训练损失曲线，带圆圈标记，标注图例
axes[0].plot(epochs_x, history["train_loss"], "o-", label="Train Loss")
# 左图：绘制验证损失曲线，带方块标记
axes[0].plot(epochs_x, history["val_loss"], "s-", label="Val Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss Curve")
axes[0].legend()  # 显示图例
# 右图：绘制验证token准确率曲线，带三角标记，绿色
axes[1].plot(epochs_x, history["val_acc"], "^-", color="green", label="Token Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Validation Token Accuracy")
axes[1].legend()

plt.tight_layout()  # 自动调整子图间距，防止标题/标签重叠
plt.savefig(os.path.join(CFG["save_dir"], "loss_curve.png"), dpi=150)  # 高分辨率保存图片
plt.show()  # 弹窗显示图像

# 把训练历史存成JSON文本文件，后续可以直接读取不用跑第二遍统计数据
import json

with open(os.path.join(CFG["save_dir"], "history.json"), "w") as f:
    json.dump(history, f, indent=2)
print("📈 曲线与训练历史已保存")
