import numpy as np

def npy_to_csv(npy_file_path, csv_file_path):
    # .npyファイルの読み込み
    data = np.load(npy_file_path)
    
    # 配列の次元数を確認（np.savetxtは1次元か2次元のみ対応）
    if data.ndim > 2:
        print(f"エラー: {data.ndim}次元の配列です。CSVとして保存するには2次元以下に変換（reshape）してください。")
        return

    # CSVファイルとして保存
    np.savetxt(csv_file_path, data, delimiter=',', fmt='%s')
    print(f"保存が完了しました: {csv_file_path}")

# 実行例
if __name__ == "__main__":
    npy_file = 'll_snh10_max_ratio8.000_20260703_1536.npy'   # 読み込む.npyファイル名
    csv_file = '8.00.csv'  # 出力する.csvファイル名
    
    # もしテスト用の.npyファイルが手元になければ、以下の2行でダミーデータを作成できます
    # np.save(npy_file, np.array([[1, 2, 3], [4, 5, 6]]))
    
    npy_to_csv(npy_file, csv_file)