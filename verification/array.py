import numpy as np

A = np.array([[3,4],[7,8]])
A_inv = np.linalg.inv(A); print(A_inv)
B = np.array([[1,4],[3,3]])
C = np.array([[2,6],[5,1]])
x = 2; y = 3

# H は行列計算×スカラー　Iは要素計算×スカラー　もちろん結果は違う
H = x * A @ B
print("H = \n", H)
H = A @ B * x
print("H = \n", H)

I = A * x * B
print("I = \n", I)

# (スカラー*行列)@(スカラー*行列) を一緒に計算するか、別々に計算するかで結果は同じなのか
J = x * A; print("J = \n", J)
K = y * B; print("K = \n", K)
L = J @ K
print("L = \n", L)
L = (x * A) @ (y * B)
print("L = \n", L)

# 行列計算中のスカラーの位置はどこでもいいのか
M = (x * A) @ B @ C
print("M = \n", M)
M = A @ (x * B) @ C
print("M = \n", M)
M = A @ B @ (x * C)
print("M = \n", M)