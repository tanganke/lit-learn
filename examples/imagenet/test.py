import argparse

import lightning as L
from lightning_utilities.core.rank_zero import rank_zero_only
from torchmetrics.classification import MulticlassAccuracy
from train import MODEL_FACTORY, get_dataloaders

from lit_learn.lit_modules import ERM_LitModule


def parse_args():
    parser = argparse.ArgumentParser(description="ImageNet Testing with lit-learn")
    parser.add_argument(
        "--data_root", type=str, required=True, help="Path to ImageNet dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["resnet34", "resnet50", "resnet101", "resnet152"],
        default="resnet50",
        help="Model architecture",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for dataloaders"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for dataloaders"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to model checkpoint"
    )
    args = parser.parse_args()
    return args


def main(args):
    # initialize model
    model = MODEL_FACTORY[args.model]()

    # initialize dataloaders
    train_loader, val_loader = get_dataloaders(args)

    lit_module = ERM_LitModule.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        model=model,
        metrics={
            "acc@1": MulticlassAccuracy(num_classes=1000),
            "acc@5": MulticlassAccuracy(num_classes=1000, top_k=5),
        },
    )

    trainer = L.Trainer(devices=1, num_nodes=1, logger=False)
    trainer.test(lit_module, dataloaders=val_loader)


if __name__ == "__main__":
    args = parse_args()
    main(args)
