from pathlib import Path
import subprocess
import time

max_retry = 3
retry = 0

python_exe = Path("venv/Scripts/python.exe")

while True:
    result = subprocess.run([str(python_exe), "evaluate_lc.py"])

    if result.returncode == 0:
        print("正常終了。再実行せず終了します。")
        break

    retry += 1
    if retry >= max_retry:
        break

    else:
        print("異常終了。再実行します。")
        time.sleep(2)
