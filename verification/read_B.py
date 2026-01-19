filename = "output_B_try.txt"

# 1. ファイルを一度読んで、「行」と「chi_vhの値」をセットで保持する
lines_with_values = []

with open(filename, "r", encoding="utf-8") as f:
    for line in f:
        line_strip = line.rstrip("\n")

        # chi_vh の値を取り出す
        # 形式: "chi_vh = 1.2e-09"
        if "chi_vh" in line:
            # "=" の右側を取り、そのまま float に変換
            try:
                value_str = line_strip.split("chi_vh =")[1].strip()
                value = float(value_str)
            except Exception:
                # 解析に失敗した場合は value を None にする
                value = None
        else:
            value = None

        lines_with_values.append((line_strip, value))

# 2. chi_vh の値がある行だけ抽出して、上位3つを選ぶ
valid_values = [(i, v) for i, (_, v) in enumerate(lines_with_values) if v is not None]
# value の大きい順にソート
valid_values.sort(key=lambda x: x[1], reverse=True)
top3_indices = set(i for i, _ in valid_values[:3])

# 3. もとの行に「〇」を付けて出力
for idx, (text, v) in enumerate(lines_with_values):
    if idx in top3_indices:
        print(text + " 〇")
    else:
        print(text)
