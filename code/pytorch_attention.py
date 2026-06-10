import torch
import torch.nn.functional as F
import numpy as np

def pytorch_attention(query, key, value, mask, num_heads):
    # 1. 拆分查询、键和值以适应多头注意力
    batch_size, seq_len, num_heads,head_dim = query.size()
    
    # 转秩（batch_size, seq_len, num_heads, head_dim） -> (batch_size, num_heads, seq_len, head_dim)
    Q = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    V = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # 2. 计算注意力分数
    scores = torch.matmul(Q, k.transpose(-2, -1)) / np.sqrt(head_dim)
    
    # 3. 应用掩码
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # 4. Softmax归一化
    attn_weights = F.softmax(scores, dim=-1)
    
    # 5. 加权求和
    output = torch.matmul(attn_weights, V)
    
    # 6. 合并多头
    output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)
    return output

# 参数设置
batch = 2
seq_len = 5
num_heads = 4  
head_dim = 8  

np.random.seed(42)
q_np = np.random.rand(batch, seq_len, num_heads, head_dim)  
k_np = np.random.rand(batch, seq_len, num_heads, head_dim)
v_np = np.random.rand(batch, seq_len, num_heads, head_dim)
q = torch.tensor(q_np)
k = torch.tensor(k_np)
v = torch.tensor(v_np)
mask = torch.ones((seq_len, seq_len), dtype=torch.bool)  # 示例掩码，可以根据需要调整

output = pytorch_attention(q, k, v, mask, num_heads)
print(output.shape)  # 应该输出torch.Size([batch, seq_len, d_model])
