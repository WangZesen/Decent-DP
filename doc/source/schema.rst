Algorithm schemas
==================

As described in the paper, the extension is designed for communicate-while-adapt decentralized algorithms for the best efficiency. To be more specific, the extension is designed for the following algorithm schema:

.. math::

    x_i^{(t+1)} = -\alpha d_i^{(t)}(x_i^{(t)}) + \sum_{j\in\mathcal{N}(i)} w_{ij} x_j^{(t)}

where :math:`x_i^{(t)}` is the model parameter of worker :math:`i` at iteration :math:`t`, :math:`d_i^{(t)}` is the local update of worker :math:`i` at iteration :math:`t`, :math:`\mathcal{N}(i)` is the neighbors of worker :math:`i`, and :math:`w_{ij}` is the weight of the edge between worker :math:`i` and worker :math:`j`.

The extension also provides calling to customized "pre_average_hook" of the optimizers, which allows users to include the communication result in the local update.