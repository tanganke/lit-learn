# lit-learn: A Toolkit for Multi-Task and Multi-Objective Optimization

lit-learn is a Python toolkit designed for multi-task and multi-objective optimization built on **PyTorch Lightning** and **Lightning Fabric**. It provides a flexible and efficient framework for researchers and practitioners to tackle complex optimization problems across various domains.

## 🏗️ Architecture

lit-learn follows a **layered architecture** with **strategy pattern** for algorithms:

```
┌─────────────────────────────────────┐
│           User Interface            │  ← Examples, Notebooks
├─────────────────────────────────────┤
│          Trainer Adapters           │  ← Lightning/Fabric Adapters  
├─────────────────────────────────────┤
│            Algorithms               │  ← Multi-task/Multi-objective
├─────────────────────────────────────┤
│         Strategy Layer              │  ← Optimization Strategies
├─────────────────────────────────────┤
│            Core Layer               │  ← Base Classes, Objectives
└─────────────────────────────────────┘
```

## 📦 Installation

```bash
git clone https://github.com/tanganke/lit-learn.git
cd lit-learn
pip install -e .
```



## 📚 Documentation

Comprehensive documentation is available:

```bash
cd docs
make html
```

## 🛠️ Extending lit-learn

### Adding Custom Algorithms

```python
from lit_learn.algorithms import BaseAlgorithm

class MyCustomAlgorithm(BaseAlgorithm):
    def _get_default_params(self):
        return {"custom_param": 1.0}
    
    def _compute_algorithm_loss(self, batch, stage):
        # Your custom loss computation
        return {"total_loss": loss}
```

### Adding Custom Objectives

```python
from lit_learn.core.objectives import BaseObjective

class MyCustomObjective(BaseObjective):
    def compute(self, predictions, targets):
        # Your custom objective computation
        return custom_loss
```

## 🎯 Roadmap

- [ ] Core architecture with Lightning/Fabric support
- [ ] Multi-task learning algorithms
- [ ] Multi-objective optimization methods
- [ ] Advanced Pareto optimization (NSGA-II, MOGA)
- [ ] Bayesian optimization integration
- [ ] AutoML for hyperparameter tuning
- [ ] More pre-built algorithms
- [ ] Integration with popular datasets

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

