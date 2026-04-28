from fastai.vision.all import *
from pathlib import Path

learn = load_learner('models/model.pkl')

test_path = Path('data/test')

if not test_path.exists():
    raise FileNotFoundError(f"Test folder not found: {test_path}")

correct = 0
total = 0
failed = 0  # images the model couldn't process at all

# Folder structure expected: data/test/<class_name>/<image_files>
for cls in test_path.iterdir():
    if not cls.is_dir():
        continue

    for img in cls.iterdir():
        if img.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue

        try:
            pred, _, _ = learn.predict(img)
        except Exception as e:
            print(f"Skipped {img.name}: {e}")
            failed += 1
            continue

        if pred == cls.name:
            correct += 1
        total += 1

if total == 0:
    print("No images found. Check that your test folder has subfolders named after each class.")
else:
    print(f"Tested : {total} images")
    print(f"Skipped: {failed} images (load/predict errors)")
    print(f"Accuracy: {(correct / total) * 100:.2f}%")