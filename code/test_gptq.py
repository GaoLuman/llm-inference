import numpy as np
import torch


def quantize_gptq_simulation(weight,calib_x,bits = 4,blocksize =128):
    '''
    GPTQ量化模拟
    weight权重矩阵,(output_dim,input_dim)
    calib_x校准矩阵(num_samples,input_dim)
    '''
    #1.计算Hessian矩阵
    n_sample,n_feat = calib_x.shape
    H = calib_x.T @ calib_x / n_sample #近似获取Hessian矩阵
    damping = 0.01 * np.mean(np.diag(H)) #自适应阻尼，防止出现H奇异
    H += damping * np.eye(n_feat)
    H_inv = np.linalg.inv(H) #求逆
    print(f"shape:{H_inv.shape}")
    
    #2.计算缩放因子
    max_val = np.abs(weight).max()
    bits_range = 2 **(bits - 1) -1
    scale = max_val / bits_range

    #3.逐列进行GPTQ量化
    W = weight.copy()
    n_col = W.shape[1]

    for i in range(0,n_col,blocksize):
        end = min(i+blocksize,n_col)

        for j in range(i,end):
            q_col = np.round(W[:,j:j+1]/ scale)
            err = W[:,j:j+1] - q_col*scale
            W[:,j:j+1] = q_col*scale #反量化后的值

            H_j = H_inv[j,j]
            if j+1 < end:
                W[:,j+1:] -= (err * H_j)#弥补后面列误差
    
    return W

if __name__ == "__main__":
    np.random.seed(42)
    weight = np.random.randn(128,256)
    calib_x = np.random.randn(64,256)

    quantized_weight = quantize_gptq_simulation(weight,calib_x)

    error = np.mean(np.abs(weight-quantized_weight))
    print(f"量化误差{error}")
