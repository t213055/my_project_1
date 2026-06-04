#逆行列の計算
#逆行列が存在しない場合もあるので注意
import numpy as np

a = np.array([[1,2],[3,4]])
print(a)

#ライブラリを使った計算
b = np.linalg.inv(a)
print(b)

c = a @ b
print(c)

#解析的に求めた逆行列
d = (1/(-2))*np.array([[4,-2],[-3,1]])
print(d)

e = a @ d
print(e)