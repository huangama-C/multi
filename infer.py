# =============================================================
# 加载训练好的模型，贪心解码，输出翻译示例
# =============================================================
import os
import torch
from main import SimpleTransformer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 再次配置兼容环境避免崩溃

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"

# 加载训练保存的checkpoint，map_location="cpu"先读进CPU，避免没有GPU时报错
ckpt = torch.load(r"E:\code\runs\mt_model.pt", map_location="cpu", weights_only=False)
cfg = ckpt["cfg"]  # 取出训练时的超参数
src_itos = ckpt["src_vocab"]  # 取出源语言编号→词列表
tgt_itos = ckpt["tgt_vocab"]  # 取出目标语言编号→词列表
# 手动反向生成目标语言的词→编号字典
src_stoi = {w: i for i, w in enumerate(src_itos)}
tgt_stoi = {w: i for i, w in enumerate(tgt_itos)}
device = "cuda" if torch.cuda.is_available() else "cpu"

# 重建和训练时完全同结构的模型
model = SimpleTransformer(
    src_vocab_size=len(src_itos),
    tgt_vocab_size=len(tgt_itos),
    d_model=cfg["d_model"], num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
    num_layers=cfg["num_layers"], max_len=cfg["max_len"] + 10, dropout=cfg["dropout"],
).to(device)
# 把训练好的权重灌进模型
model.load_state_dict(ckpt["model_state"])
model.eval()  # 推理必须切评估模式


@torch.no_grad()  # 整个翻译函数关闭梯度，省显存提速
def translate(sentence, max_len=50):
    """
    贪心解码函数：输入德语句子，逐词输出英语译文
    每次生成一个概率最大的词，接到当前序列后面，下次把完整序列重新喂进模型
    """
    tokens = sentence.lower().split()  # 德语输入转小写分词
    # 转成带bos和eos的编号序列
    src_ids = [src_stoi[BOS]] + [src_stoi.get(w, src_stoi[UNK]) for w in tokens] + [src_stoi[EOS]]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)  # 外面加一层[]，batch维度变成1
    src_mask = (src != src_stoi[PAD]).unsqueeze(1).unsqueeze(2)  # 生成源掩码

    ys = [tgt_stoi[BOS]]  # 已生成序列从<bos>开始
    for _ in range(max_len):  # 最多生成50个词，防止无限循环
        tgt = torch.tensor([ys], dtype=torch.long, device=device)  # 把当前已生成序列转成张量
        T = tgt.size(1)  # 当前已生成的序列长度
        # 构造当前步的因果掩码
        tgt_mask = torch.tril(torch.ones(T, T, device=device)).bool().unsqueeze(0).unsqueeze(0)
        tgt_mask = tgt_mask & (tgt != tgt_stoi[PAD]).unsqueeze(1).unsqueeze(2)

        # 模型前向，得到当前步所有位置的打分
        logits = model(src, tgt, src_mask, tgt_mask)
        # 只取最后一个位置的全词表打分，argmax得到概率最大的下一个词编号
        next_id = logits[0, -1, :].argmax().item()
        # 如果生成的是<eos>，说明翻译结束，跳出循环
        if next_id == tgt_stoi[EOS]:
            break
        # 把新生成的词编号加到序列尾部
        ys.append(next_id)

    # 把编号转回词，去掉开头的<bos>，拼成最终的译文字符串
    words = [tgt_itos[i] for i in ys[1:]]
    return " ".join(words)


# -------- 手动定义的几个测试德语句子，直接看翻译效果 --------
examples = [
    "ein mann in einem blauen hemd steht auf einer leiter und putzt ein fenster",
    "zwei junge männer spielen fußball auf einem feld",
    "ein hund läuft durch den schnee",
    "eine frau mit einem roten kleid tanzt",
    "kinder spielen im park mit einem ball",
]
print("=" * 70)
for de in examples:
    en = translate(de)
    print(f"DE : {de}")
    print(f"EN : {en}")
    print("-" * 70)

# -------- 从验证集随机抽5个样本，和官方参考译文对照 --------
print("\n【与参考译文对照】")
import random

# 读取验证集原始的德语文本和英语参考文本
val_de = open(r"E:\Multi\data\multi\val.de", encoding="utf-8").read().strip().split("\n")
val_en = open(r"E:\Multi\data\multi\val.en", encoding="utf-8").read().strip().split("\n")
# 随机采样5个索引
idx = random.sample(range(len(val_de)), 5)
for i in idx:
    print(f"DE      : {val_de[i]}")
    print(f"模型输出: {translate(val_de[i])}")
    print(f"参考译文: {val_en[i]}")
    print("-" * 70)
