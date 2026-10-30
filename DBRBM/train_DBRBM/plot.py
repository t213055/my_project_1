import matplotlib.pyplot as plt

# データを用意
x = [1,2,3]
y = [5,10,15]

# 描画オブジェクト（fig）とサブプロット（ax）を生成する
fig, ax = plt.subplots()

# 折れ線グラフを表示
ax.plot(x, y)
# グラフにタイトルをつける
ax.set_title('テストタイトル', fontname="Meiryo")
# x軸のラベルを設定
plt.xlabel("x軸ラベル")
# y軸のラベルを設定
plt.ylabel("y軸ラベル")

# グラフを表示
plt.show()