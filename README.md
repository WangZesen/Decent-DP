# Decentralized Data Parallel (Decent-DP)

**Decent-DP** is a cutting-edge PyTorch extension designed to simplify and accelerate decentralized data parallel training. As the official implementation of the paper [**[ICLR'25] From Promise to Practice: Realizing High-performance Decentralized Training**](https://github.com/WangZesen/Decentralized-Training-Exp), Decent-DP empowers you to scale multi-worker training efficiently—eliminating centralized bottlenecks and streamlining your deep learning pipelines.

[![arXiv](https://img.shields.io/badge/arXiv-2401.11998-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2410.11998) 
[![OpenReview](https://img.shields.io/badge/OpenReview-Paper-blue)](https://openreview.net/forum?id=lo3nlFHOft)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyTorch Extension](https://img.shields.io/badge/PyTorch-Extension-brightgreen.svg)](https://pytorch.org/)

<p align="center">
  <img src="https://github.com/WangZesen/Decent-DP/blob/76c157d9b79c2ec83d692012da32c8f4998fc031/docs/images/logo-light.png?raw=True"/>
</p>

## ✨ Key Features

- **Decentralized Architecture**  
  Efficiently distributes training across multiple workers without relying on a central coordinator.

- **Seamless PyTorch Integration**  
  Easily plug into your existing PyTorch codebase with minimal modifications.

- **High-Performance**  
  Optimized for speed and scalability based on state-of-the-art research.

- **Flexible and Extensible**  
  Supports various algorithmic schemas to suit different training scenarios and model architectures.



## ⚙️ Installation

### Prerequisites

- Python 3.11+
- [PyTorch](https://pytorch.org/)

### Via pip

Install directly from PyPI:

```bash
pip install decent-dp
```

### From Source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/WangZesen/Decent-DP.git
cd Decent-DP
pip install -e .
```


## 🚀 Quickstart

Here is a complete example of how to use Decent-DP to train a model:

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from decent_dp.ddp import DecentralizedDataParallel as DecentDP
from decent_dp.optim import optim_fn_adamw
from decent_dp.utils import initialize_dist

# Initialize distributed environment
rank, world_size = initialize_dist()

# Create your model
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
).cuda()

# Wrap model with DecentDP
model = DecentDP(
    model,
    optim_fn=optim_fn_adamw,  # or your custom optimizer function
    topology="complete"      # or "ring", "one-peer-exp", "alternating-exp-ring"
)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = nn.functional.mse_loss(output, target)
        
        # Zero gradients, backward pass
        model.zero_grad()
        loss.backward()
        # Note: optimizer.step() is automatically called by DecentDP
        
    # Evaluation
    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            val_loss = nn.functional.mse_loss(output, target)
```

Launch the script on multiple processes/nodes using [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html):

```bash
torchrun --nproc_per_node=4 your_training_script.py
```

## 📚 Key Concepts

### Decentralized Training
Unlike traditional centralized approaches where all workers communicate with a single parameter server, decentralized training allows workers to communicate directly with their neighbors. This eliminates bottlenecks and improves scalability.

### Communication Topologies
Decent-DP supports various communication patterns:
- **Complete**: All workers communicate with each other in each iteration
- **Ring**: Workers form a ring and communicate with their immediate neighbors
- **One-Peer Exponential**: Workers communicate with peers at exponentially increasing distances
- **Alternating Exponential-Ring**: Alternates between exponential and ring communication patterns

### Parameter Bucketing
Decent-DP automatically groups model parameters into buckets based on size, optimizing communication efficiency during training.

### Gradient Accumulation
The framework handles gradient accumulation seamlessly, making it easy to simulate larger batch sizes across multiple workers.



## 📖 Documentation

Code of experiments conducted in the paper: 	🔍 **[WangZesen/Decentralized-Training-Exp](https://github.com/WangZesen/Decentralized-Training-Exp)**

Comprehensive documentation, including tutorials, API references, and performance tips, is available on the Github page: **[Decent-DP Documentation](https://wangzesen.github.io/Decent-DP)**


## 📝 Citation

If you use Decent-DP in your research, please cite our work:


```bibtex
@article{wang2025decentralized,
  title={From Promise to Practice: Realizing High-Performance Decentralized Training},
  author={Wang, Zesen and Zhang, Jiaojiao and Wu, Xuyang and Johansson, Mikael},
  journal={arXiv preprint arXiv:2410.11998},
  year={2025}
}
```


## 🤝 Contributing

We welcome contributions from the community!  
To get involved:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a clear description of your changes.
4. For any issues or feature requests, please open an issue on GitHub.

---

## 🧾 License

Decent-DP is released under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

The computations and storage resources were enabled by resources provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS), partially funded by the Swedish Research Council through grant agreement no. 2022-06725.

---

🚀 Happy training with Decent-DP!
