with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
# Define a simple character-level tokenizer.
#encoder and decoder
string_to_int={}
for i,ch in enumerate(chars):
    string_to_int[ch]=i
    
int_to_string={}
for i,ch in enumerate(chars):
    int_to_string[i]=ch

def encode(s):
    ids=[]
    for c in s:
        ids.append(string_to_int[c])
    return ids
def decode(l):
    chars = []
    for i in l:
        chars.append(int_to_string[i])
    return ''.join(chars)
#round-trip check
roundtrip_text = decode(encode("my name is YujingDong"))
print(roundtrip_text)


import torch
data = torch.tensor(encode(text), dtype=torch.long)
#split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337)
batch_size=4
block_size=8
def get_batch(data):
     # randomly choose starting positions in the text
     start_positions = torch.randint(
        low=0,
        high=len(data) - block_size,
        size=(batch_size,)
    )
     inputs = []
     targets = []
     for start in start_positions:
         inputs.append(data[start:start+block_size])
         targets.append(data[start+1:start+block_size+1])
    
     x=torch.stack(inputs)
     y=torch.stack(targets)
     return x,y
 
import torch.nn as nn
import torch.nn.functional as F
class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):   
        if targets is None:
            loss=None
        else:
            logits=self.token_embedding_table(idx) 
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            targets=idx.view(B*T)        
            loss=F.cross_entropy(logits, targets)
        return logits,loss
    def generate(self, idx, max_new_tokens):
        """
    Generate new tokens autoregressively.

    Starting from the existing token sequence `idx`,
    the model repeatedly:
    - predicts the probability distribution of the next token
    - samples one token from that distribution
    - appends the sampled token to the sequence

    This process is repeated `max_new_tokens` times.

    Args:
        idx: Tensor of shape (B, T) containing the current token sequence.
        max_new_tokens: Number of new tokens to generate.

    Returns:
        Tensor containing the original tokens followed by the generated tokens.
    """
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
        
#Train the model
xb, yb = get_batch(train_data)
m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
optimizer=torch.optim.AdamW(m.parameters(), lr=1e-3)
batch_size=32
for steps in range(10000):
    logits,loss=m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


"""
这段代码是一个“单头自注意力（single-head self-attention）”的完整演示版本，
用于理解 attention 的核心计算流程，而不是最终用于搭建模型的代码。
整体流程如下：
1. 构造一个假的输入张量 x，形状为 (B, T, C)，
   表示 B 条序列、每条序列 T 个 token、每个 token 是 C 维向量。
2. 通过三个可学习的线性变换，将输入映射为 Query / Key / Value。
3. 使用 Q @ K^T 计算注意力分数（attention scores），得到每个位置对其他位置的相关性。
4. 使用下三角矩阵（causal mask）屏蔽未来位置，确保自回归（不能看未来）。
5. 对注意力分数做 softmax，得到注意力权重（每一行是概率分布）。
6. 使用注意力权重对 Value 做加权平均，得到最终的 attention 输出。

这段代码的目的是：
- 直观展示 self-attention 的数学本质
- 帮助理解 Q / K / V、mask、softmax、加权求和之间的关系

注：
- 这里没有封装成 nn.Module
- 没有 scale（1/sqrt(d_k)）
- 没有 dropout
- 只演示一个 attention head
"""
torch.manual_seed(1337)

B, T, C = 4, 8, 32
x = torch.randn(B, T, C)

head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)
q = query(x)
wei = q @ k.transpose(-2, -1)
# Create a causal mask to prevent attending to future positions
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v
