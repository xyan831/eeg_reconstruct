import os
import shutil
import re

src_dir = "result/data_train"
dst_dir = "result/data_gen"

os.makedirs(dst_dir, exist_ok=True)

def safe_replace(filename):
    # Skip "both"
    if "both" in filename:
        return None

    newname = filename

    # Replace nseiz ? non_seizure_data
    newname = re.sub(r"nseiz(?=\.|_|$)", "non_seizure_data", newname)

    # Replace seiz ? seizure_data (not part of nseiz)
    newname = re.sub(r"(?<!n)seiz(?=\.|_|$)", "seizure_data", newname)

    return newname

for fname in os.listdir(src_dir):
    if not fname.endswith(".mat"):
        continue

    newname = safe_replace(fname)
    if newname is None:
        print(f"Skipping (both): {fname}")
        continue

    src_path = os.path.join(src_dir, fname)
    dst_path = os.path.join(dst_dir, newname)

    print(f"Copying: {fname} -> {newname}")
    shutil.copyfile(src_path, dst_path)

print("Done! Clean copy + rename complete.")

