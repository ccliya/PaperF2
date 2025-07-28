import os
import shutil

train_raw = "data/UIEB/raw"
train_ref = 'data/UIEB/reference'
test_raw = "data/UIEB/raw_test"
test_ref = 'data/UIEB/reference_test'

for img in os.listdir(test_raw):
    basename = os.path.basename(img)
    if basename in os.listdir(train_raw):
        os.remove(os.path.join(train_raw, img))
        print(f"Removed {img} from test set as it exists in train set.")
    if basename in os.listdir(train_ref):
        shutil.copy(os.path.join(train_ref, basename), os.path.join(test_ref, basename))
        os.remove(os.path.join(train_ref, basename))