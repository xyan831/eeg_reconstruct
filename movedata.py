import os
import shutil
import re

src_dir = "result/data_train"
dst_dir = "result/data_gen"

os.makedirs(dst_dir, exist_ok=True)

for fname in os.listdir(src_dir):
    if not fname.endswith(".mat"):
        continue

    if "both" in fname:
        print(f"Skipping (both): {fname}")
        continue

    src_path = os.path.join(src_dir, fname)
    dst_path = os.path.join(dst_dir, fname)

    print(f"Copying: {fname}")
    shutil.copyfile(src_path, dst_path)

print("copy ok")

