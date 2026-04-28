from fastai.vision.all import *
from pathlib import Path
import torch


def main():
    path = Path('../data')

    if not path.exists():
        raise FileNotFoundError(f"Data folder not found: {path.resolve()}")

    dls = ImageDataLoaders.from_folder(
        path,
        train='train',
        valid='valid',
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(),  # random flips, zoom, lighting — helps the model generalise
        bs=16,
        num_workers=0  # set to 0 on Windows to avoid dataloader crashes
    )

    print(f"Classes: {dls.vocab}")
    print(f"Train: {len(dls.train_ds)} images, Valid: {len(dls.valid_ds)} images")

    learn = vision_learner(dls, resnet50, metrics=accuracy)

    # lr_find helps pick a good learning rate instead of guessing
    suggested_lr = learn.lr_find(suggest_funcs=minimum).valley
    print(f"Suggested learning rate: {suggested_lr:.2e}")

    learn.fine_tune(8, base_lr=suggested_lr)

    Path('../models').mkdir(exist_ok=True)
    learn.export('../models/model.pkl')

    print("Training complete. Model saved to ../models/model.pkl")


if __name__ == '__main__':
    main()