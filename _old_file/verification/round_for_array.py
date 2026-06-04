import numpy as np

#round()の引数に式をいれても大丈夫そう
a = round(1 + 1e-5, 2)
print(a)

#変数でもOK？
i = 3e-6; j = 6
a = i + j; print(a)
print("after round", round(i + j, 3))

A = np.array([[1e-20,3e-26],[5e-20,1e-15]])
print(A)
A = np.round(A,22)
print(A)

a = 1e-5;  b= 1e-6;  c= 1e-7
A = np.array([a,b,c])
A = np.round(A,6)
print("A = \n", A)
print("a =", a, "b =", b, "c =", c)
print("a =", A[0], "b =", A[1], "c =", A[2])