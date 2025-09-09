import argparse

import lightning as L
import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from torchvision import transforms
from torchvision.datasets import ImageNet
from torchvision.models import resnet34, resnet50, resnet101, resnet152

from lit_learn.lit_modules import ERM_LitModule


def parse_args():
    parser = argparse.ArgumentParser(description="ImageNet Training with lit-learn")
    parser.add_argument(
        "--data_root", type=str, required=True, help="Path to ImageNet dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet34", "resnet50", "resnet101", "resnet152"],
        help="Model architecture",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size per GPU (baseline: 32)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=90,
        help="Number of epochs to train (baseline: 90)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="Initial learning rate (baseline: 0.1)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="SGD momentum (baseline: 0.9)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay (baseline: 1e-4)"
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.0, help="Label smoothing"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam"],
        help="Optimizer type (baseline: sgd)",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="step",
        choices=["step", "cosine"],
        help="Learning rate scheduler type (baseline: step)",
    )
    parser.add_argument(
        "--lr_step_size",
        type=int,
        default=30,
        help="Step size for StepLR scheduler (baseline: 30)",
    )
    parser.add_argument(
        "--lr_gamma",
        type=float,
        default=0.1,
        help="Gamma for StepLR scheduler (baseline: 0.1)",
    )
    args = parser.parse_args()
    return args


def get_dataloaders(args):
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = ImageNet(
        root=args.data_root,
        split="train",
        transform=transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                to_tensor,
                normalize,
            ]
        ),
    )
    val_dataset = ImageNet(
        root=args.data_root,
        split="val",
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                to_tensor,
                normalize,
            ]
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    return train_loader, val_loader


def main(args):
    model_factory = {
        "resnet34": lambda: resnet34(weights=None, num_classes=1000),
        "resnet50": lambda: resnet50(weights=None, num_classes=1000),
        "resnet101": lambda: resnet101(weights=None, num_classes=1000),
        "resnet152": lambda: resnet152(weights=None, num_classes=1000),
    }

    # initialize model
    model = model_factory[args.model]()

    # initialize optimizer and scheduler
    if args.optimizer == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_scheduler == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_epochs
        )

    lit_module = ERM_LitModule(
        model,
        optimizers={
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        },
        objective=nn.CrossEntropyLoss(label_smoothing=args.label_smoothing),
        metrics={
            "acc@1": MulticlassAccuracy(num_classes=1000),
            "acc@5": MulticlassAccuracy(num_classes=1000, top_k=5),
        },
        metrics_on_prog_bar=True,
    )

    # initialize dataloaders
    train_loader, val_loader = get_dataloaders(args)

    # setup trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
    )
    trainer.fit(lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    args = parse_args()
    main(args)
