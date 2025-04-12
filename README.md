# Decentralized Data Parallel (Decent-DP)

**Decent-DP** is a cutting-edge PyTorch extension designed to simplify and accelerate decentralized data parallel training. As the official implementation of the paper [**[ICLR'25] From Promise to Practice: Realizing High-performance Decentralized Training**](https://github.com/WangZesen/Decentralized-Training-Exp), Decent-DP empowers you to scale multi-worker training efficiently—eliminating centralized bottlenecks and streamlining your deep learning pipelines.

[![arXiv](https://img.shields.io/badge/arXiv-2401.11998-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2410.11998) 
[![OpenReview](https://img.shields.io/badge/OpenReview-Paper-blue)](https://openreview.net/forum?id=lo3nlFHOft)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyTorch Extension](https://img.shields.io/badge/PyTorch-Extension-brightgreen.svg)](https://pytorch.org/)

<p align="center">
  <img src="./doc/source/_static/icon-light.png" alt="Icon created by zero_wing - Flaticon"/>
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

Here is a pseudocode exmaple of how to use Decent-DP to train a model

```python

  import torch.distributed as dist
  from decent_dp.ddp import DecentralizedDataParallel as DecentDP

  # Initialize process group
  dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo', init_method='env://')

  # Initialize model (move to device before wrapping with DecentDP)
  model = ...
  model = model.to(device)

  # Wrap model with DecentDP
  model = DecentDP(model,
                    # optimizer constructor function which takes List[Tuple[str, Tensor]] as input and returns an optimizer
                    # examples could be found in `decent_dp.optim` module
                    optim_fn=<optimizer constructor function>,
                    # lr scheduler constructor function which takes an optimizer as input and returns a lr scheduler.
                    # None if no lr scheduler is used
                    # examples could be found in `decent_dp.optim` module
                    lr_scheduler_fn=<lr scheduler constructor function>,
                    # topology of the network which is a string
                    # supported topologies are 'ring', 'exp', 'complete', 'alternating-exp-ring'
                    # see Section `Communication topology` for more details
                    topology=<topology>)
  
  # Training loop
  for epoch in range(num_epochs):
      model.train()
      for batch in data_loader:
          loss = model(batch)
          model.zero_grad()
          loss.backward()
          # no need for optimizer.step() as it is handled by DecentDP

      model.eval()
      for batch in val_data_loader:
          with torch.no_grad():
              loss = model(batch)
```

Launch the script on multiple processes/nodes using [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html).



## 📖 Documentation

Code of experiments conducted in the paper: 	🔍 **[WangZesen/Decentralized-Training-Exp](https://github.com/WangZesen/Decentralized-Training-Exp)**

Comprehensive documentation, including tutorials, API references, and performance tips, is available on the Github page: **[Decent-DP Documentation](https://wangzesen.github.io/Decent-DP)**


## 📝 Citation

If you use Decent-DP in your research, please cite our work:


```bibtex
@article{wang2024decentralized,
  title={From Promise to Practice: Realizing High-Performance Decentralized Training},
  author={Wang, Zesen and Zhang, Jiaojiao and Wu, Xuyang and Johansson, Mikael},
  journal={arXiv preprint arXiv:2410.11998},
  year={2024}
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

