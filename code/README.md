## test_transformer_attention.py说明
这个文件采用了python和numpy，对transformer的Self-attention的过程进行了一个原理复现，验证了张量结构在这个过程中的传递。  
自注意力计算公式为：<br>
$Attention(Q, K, V) = softmax(\frac{Q·K^T} {\sqrt{d_k}}) · V$ <br>
Q,K,V的张量类型为[batch,seq_len,num_heads,head_dim]<br>
经过验证，最后输出的结果张量类型正确，没有实现合并多头

## pytorch_attention.py说明
这个文件采用pytorch实现了一个attention的复现，对比numpy可以看出代码更加精炼，有一些已经封装的函数可以直接使用  

## fastapi_vllm_generate.py说明
调用fastapi实现了一个vllm的接口，实验结果在docs文档中

## fastapi_stream.py说明
新增了一个流式输出接口，实验结果在docs文档中
