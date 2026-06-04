import os
import pickle
import tarfile
import urllib.request

def prepare_cifar10():
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    folder = "cifar-10-batches-py"

    # ダウンロード
    if not os.path.exists(filename):
        print("Downloading CIFAR-10...")
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")

    # 解凍
    if not os.path.exists(folder):
        print("Extracting files...")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall()
        print("Extraction complete.")

    return folder

# 実行
data_folder = prepare_cifar10()