import numpy as np

data = np.load("teacher_nv10_nh30_s5000.npy")

print(data)
print(data.shape)
print(data.dtype)

# 平均
mean = np.mean(data)

# 分散
variance = np.var(data)

print("平均:", mean)
print("分散:", variance)