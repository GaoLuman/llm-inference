## test_transformer_attention.py说明
这个文件采用了python和numpy，对transformer的Self-attention的过程进行了一个原理复现，验证了张量结构在这个过程中的传递。  
自注意力计算公式为：<br>
$Attention(Q, K, V) = softmax(\frac{Q·K^T} {\sqrt{d_k}}) · V$ <br>
Q,K,V的张量类型为[batch,seq_len,num_heads,head_dim]<br>
经过验证，最后输出的结果张量类型正确
