.. Decent-DP documentation master file, created by
   sphinx-quickstart on Sat Feb 15 12:01:04 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Decent-DP documentation
=======================

The repository is the official implementation of the paper **From Promise to Practice: Realizing High-performance Decentralized Training** (`arXiv version <https://arxiv.org/abs/2410.11998>`_, `OpenReview <https://openreview.net/forum?id=lo3nlFHOft>`_) accepted by **ICLR 2025**.

The package is an PyTorch extension that faciliates efficient multi-worker decentralized data parallel training which fits in certain algorithm schemas.

Quick Start
-----------

Installation
^^^^^^^^^^^^

* Install PyTorch (See `PyTorch Installation Guide <https://pytorch.org/get-started/locally/>`_ for platform-specific instructions)
   .. code-block:: bash

      pip3 install torch torchvision torchaudio

* Install Decent-DP
   .. code-block:: bash

      pip3 install decent-dp

Basic Usage
^^^^^^^^^^^

Here is a pseudocode exmaple of how to use Decent-DP to train a model

.. code-block:: python

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

Citation
--------

If you find this repository helpful, please consider citing the following paper:

.. code-block:: bibtex

    @inproceedings{wang2024promise,
        title={From Promise to Practice: Realizing High-performance Decentralized Training},
        author={Zesen Wang, Jiaojiao Zhang, Xuyang Wu, and Mikael Johansson},
        booktitle={International Conference on Learning Representations},
        year={2025},
        url={https://openreview.net/forum?id=lo3nlFHOft},
    }

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   API Documentation <modules>
   Algorithm Schema <schema>
   Communication Topology <topology>
   Benchmark Tests <benchmark>
