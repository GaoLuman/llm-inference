#!/bin/bash/env python3

import numpy as np

#SoftMax函数
def softmax(x,axis = -1):
    max = x - np.max(x,axis=axis,keepdims=True)
    e_x = np.exp(max)
    return e_x/np.sum(e_x,axis=axis, keepdims=True)

def compute_rope_angles(seq_len,head_dim,base=10000.0):
    #计算每个位置的cos和sin
    inv_freq = 1.0 / (base ** (np.arange(0,head_dim,2)/head_dim))
    pos = np.arange(seq_len)
    angles = np.outer(pos,inv_freq)
    cos = np.cos(angles)
    sin = np.sin(angles) # max_seq_len,head_dim/2)
    return cos,sin

#RoPE
def RoPE(x,cos,sin):
    batch, seq_len, nums_head, head_dim = x.shape
    x_even = x[..., 0::2]  # (..., head_dim//2) 最后一维取双数
    x_odd  = x[..., 1::2]  # (..., head_dim//2) 最后一维取单数
   
    cos_ = cos[:seq_len,None,:]  #shape:(seq_len,1,head_dim/2)
    sin_ = sin[:seq_len,None,:]
    rotated_even = x_even * cos_ - x_odd * sin_
    rotated_odd = x_even * sin_ + x_odd * cos_
    # 重新交错合并，先stack后reshape才能将双数偶数嵌合
    # stack + reshape
    x_rot = np.stack([rotated_even, rotated_odd], axis=-1)
    x_rot = x_rot.reshape(x.shape) #(batch,seq_len,nums_head_head_dim)
    return x_rot

def attention_compute(q, k, v, cos, sin):
    '''
    输入：
        q,k,v:shape(batch,seq_len,nums_head,head_dim)
        cos,sin :所有RoPE的角度表 shape(max_seq_len,head_dim/2)
    输出：
        output:shape(batch,seq_len,nums_head,head_dim)
    '''
    batch, seq_len, nums_head, head_dim = q.shape
    
    #1.对q和k进行RoPE操作
    q = RoPE(q,cos,sin)
    k = RoPE(k,cos,sin)

    #2.计算注意力分数
    scores = np.einsum('bqhd,bkhd->bhqk',q,k) #shape:(batch,head_dim,q_len,k_len)
    scores = scores/np.sqrt(head_dim)

    #3.创建掩码，下三角矩阵
    mask = np.tril(np.ones((seq_len,seq_len),dtype=bool))
    scores = np.where(mask,scores,-1e9)

    #4.Softmax
    atten = softmax(scores, axis=-1)

    #5.加权求和
    out = np.einsum('bhqk,bkhd->bqhd',atten,v)
    return out

if __name__ == '__main__':
    np.random.seed(42)
    batch = 2
    seq_len = 5
    num_heads = 4
    head_dim = 8
    max_len_seq = 10

    cos,sin = compute_rope_angles(max_len_seq,head_dim)

    # 随机生成Q,K,V
    q = np.random.randn(batch, seq_len,num_heads,head_dim)
    k = np.random.randn(batch, seq_len,num_heads,head_dim)
    v = np.random.randn(batch, seq_len,num_heads,head_dim)

    #计算注意力
    out = attention_compute(q,k,v,cos,sin)
    print("Out put shape:",out.shape)
