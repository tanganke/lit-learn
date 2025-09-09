# ImageNet Training Example

Train ResNet models on ImageNet using lit-learn's ERM (Empirical Risk Minimization) module.

## Usage

Train a model:

```bash
python train.py --data_root /path/to/imagenet
```

Test the trained model:

```bash
python test.py --data_root /path/to/imagenet --checkpoint_path /path/to/checkpoint.ckpt
```

## Baseline Configuration

- **Model**: ResNet-50
- **Batch size**: 32 per GPU 
- **Epochs**: 90
- **Optimizer**: SGD (momentum=0.9, weight_decay=1e-4)
- **Learning rate**: 0.1 with StepLR (step_size=30, gamma=0.1)

## Options

- `--model`: resnet34/50/101/152
- `--batch_size`: batch size per GPU
- `--lr`: initial learning rate
- `--optimizer`: sgd/adam
- `--scheduler`: step/cosine

## Requirements

- ImageNet dataset
- PyTorch Lightning
- torchvision
