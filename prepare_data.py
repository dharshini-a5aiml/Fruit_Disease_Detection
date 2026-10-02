import os
import random
import shutil
import re

RAW_DIR = os.path.join("dataset", "raw")
TRAIN_DIR = os.path.join("dataset", "train")
TEST_DIR = os.path.join("dataset", "test")

MAX_PER_CLASS = 400   # use at most this many images per class
TEST_SPLIT = 0.2      # 20% of images go to test
EXTENSIONS = (".jpg", ".jpeg", ".png")

random.seed(42)

def clean_name(name):
    # remove spaces and brackets so folder names are simple
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    return name.strip("_")

# start fresh each time so you never get mixed-up old files
for folder in (TRAIN_DIR, TEST_DIR):
    if os.path.exists(folder):
        shutil.rmtree(folder)

total_train = 0
total_test = 0

for fruit in sorted(os.listdir(RAW_DIR)):
    fruit_path = os.path.join(RAW_DIR, fruit)
    if not os.path.isdir(fruit_path):
        continue
    for cls in sorted(os.listdir(fruit_path)):
        cls_path = os.path.join(fruit_path, cls)
        if not os.path.isdir(cls_path):
            continue

        images = [f for f in os.listdir(cls_path)
                  if f.lower().endswith(EXTENSIONS)]
        random.shuffle(images)
        images = images[:MAX_PER_CLASS]

        n_test = int(len(images) * TEST_SPLIT)
        test_imgs = images[:n_test]
        train_imgs = images[n_test:]

        new_name = clean_name(cls)
        for split_dir, split_imgs in ((TRAIN_DIR, train_imgs),
                                      (TEST_DIR, test_imgs)):
            out = os.path.join(split_dir, new_name)
            os.makedirs(out, exist_ok=True)
            for img in split_imgs:
                shutil.copy(os.path.join(cls_path, img),
                            os.path.join(out, img))

        total_train += len(train_imgs)
        total_test += len(test_imgs)
        print(f"{new_name}: train={len(train_imgs)}  test={len(test_imgs)}")

print("-" * 40)
print("TOTAL train:", total_train, " TOTAL test:", total_test)