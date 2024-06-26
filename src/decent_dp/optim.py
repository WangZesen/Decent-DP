import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import List, Tuple, Union

def optim_fn_adam(params: List[Tuple[Tensor, str]],
                  lr: float = 1e-3,
                  beta1: float = 0.9,
                  beta2: float = 0.999,
                  eps: float = 1e-8) -> Optimizer:
    return torch.optim.Adam([x for x, n in params], lr=lr, betas=(beta1, beta2), eps=eps)


def lr_scheduler_fn_cosine_with_warmup(optimizer: Optimizer,
                                       t_max: int,
                                       t_warmup: int,
                                       cosine_eta_min: float = 1e-6,
                                       warmup_decay: float = 0.01) -> LRScheduler:
    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=cosine_eta_min)
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_decay, total_iters=t_warmup)
    return torch.optim.lr_scheduler.SequentialLR(optimizer,
                                                 schedulers=[warmup_lr_scheduler, main_lr_scheduler],
                                                 milestones=[t_warmup])


def accum_adam_foreach(params: List[torch.Tensor],
                       grads: List[torch.Tensor],
                       exp_avgs: List[torch.Tensor],
                       exp_avg_sqs: List[torch.Tensor],
                       accum_grads: List[torch.Tensor],
                       state_steps: List[torch.Tensor],
                       beta1: float,
                       beta2: float,
                       lr: Union[float, torch.Tensor],
                       weight_decay: float,
                       eps: float,
                       accum_iter: int):
    torch._foreach_add_(state_steps, 1)
    if weight_decay != 0:
        torch._foreach_add_(grads, params, alpha=weight_decay)

    step = state_steps[0].item()
    torch._foreach_add_(accum_grads, grads, alpha=1./accum_iter)

    _exp_avgs = torch._foreach_add(exp_avgs, grads, alpha=1-beta1)
    _exp_avg_sqs = torch._foreach_addcmul(exp_avg_sqs, grads, grads, value=1-beta2)

    bias_correction1 = 1 - beta1 ** ((step + accum_iter - 1) // accum_iter)
    bias_correction2 = 1 - beta2 ** ((step + accum_iter - 1) // accum_iter)
    step_size = lr / bias_correction1
    bias_correction2_sqrt = math.sqrt(bias_correction2)

    torch._foreach_sqrt_(_exp_avg_sqs)
    torch._foreach_div_(_exp_avg_sqs, bias_correction2_sqrt)
    torch._foreach_add_(_exp_avg_sqs, eps)
    torch._foreach_addcdiv_(params, _exp_avgs, _exp_avg_sqs, value=-step_size)

    if step % accum_iter == 0:
        torch._foreach_add_(exp_avgs, accum_grads, alpha=1-beta1)
        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_addcmul_(exp_avg_sqs, accum_grads, accum_grads, value=1-beta2)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_zero_(accum_grads)


class AccumAdam(torch.optim.Optimizer):
    def __init__(self,
                 params: List[torch.Tensor],
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 accum_iter: int = 4):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            accum_iter=accum_iter
        )
        super().__init__(params, defaults)

    def _init_group(self,
                    group,
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    accum_grads,
                    state_steps):
        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                grads.append(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.tensor(0, dtype=torch.int64)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['accum_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                accum_grads.append(state['accum_grad'])
                state_steps.append(state['step'])

    @torch.no_grad()
    def step(self, closure=None):
        self._cuda_graph_capture_health_check()
        assert closure is None

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            accum_grads = []
            state_steps = []
            beta1, beta2 = group["betas"]

            self._init_group(
                group, params_with_grad, grads, exp_avgs, exp_avg_sqs, accum_grads, state_steps
            )

            accum_adam_foreach(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                accum_grads,
                state_steps,
                beta1,
                beta2,
                group['lr'],
                group['weight_decay'],
                group['eps'],
                group['accum_iter']
            )


def accum_adam_foreach(params: List[torch.Tensor],
                       grads: List[torch.Tensor],
                       exp_avgs: List[torch.Tensor],
                       exp_avg_sqs: List[torch.Tensor],
                       accum_grads: List[torch.Tensor],
                       state_steps: List[torch.Tensor],
                       beta1: float,
                       beta2: float,
                       lr: Union[float, torch.Tensor],
                       weight_decay: float,
                       eps: float,
                       accum_iter: int):
    torch._foreach_add_(state_steps, 1)
    if weight_decay != 0:
        torch._foreach_add_(grads, params, alpha=weight_decay)

    step = state_steps[0].item()
    torch._foreach_add_(accum_grads, grads, alpha=1./accum_iter)

    _exp_avgs = torch._foreach_add(exp_avgs, grads, alpha=1-beta1)
    _exp_avg_sqs = torch._foreach_addcmul(exp_avg_sqs, grads, grads, value=1-beta2)

    bias_correction1 = 1 - beta1 ** ((step + accum_iter - 1) // accum_iter)
    bias_correction2 = 1 - beta2 ** ((step + accum_iter - 1) // accum_iter)
    step_size = lr / bias_correction1
    bias_correction2_sqrt = math.sqrt(bias_correction2)

    torch._foreach_sqrt_(_exp_avg_sqs)
    torch._foreach_div_(_exp_avg_sqs, bias_correction2_sqrt)
    torch._foreach_add_(_exp_avg_sqs, eps)
    torch._foreach_addcdiv_(params, _exp_avgs, _exp_avg_sqs, value=-step_size)

    if step % accum_iter == 0:
        torch._foreach_add_(exp_avgs, accum_grads, alpha=1-beta1)
        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_addcmul_(exp_avg_sqs, accum_grads, accum_grads, value=1-beta2)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_zero_(accum_grads)


class AccumAdam(torch.optim.Optimizer):
    def __init__(self,
                 params: List[torch.Tensor],
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 accum_iter: int = 4):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            accum_iter=accum_iter
        )
        super().__init__(params, defaults)

    def _init_group(self,
                    group,
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    accum_grads,
                    state_steps):
        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                grads.append(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.tensor(0, dtype=torch.int64)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['accum_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                accum_grads.append(state['accum_grad'])
                state_steps.append(state['step'])

    @torch.no_grad()
    def step(self, closure=None):
        self._cuda_graph_capture_health_check()
        assert closure is None

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            accum_grads = []
            state_steps = []
            beta1, beta2 = group["betas"]

            self._init_group(
                group, params_with_grad, grads, exp_avgs, exp_avg_sqs, accum_grads, state_steps
            )

            accum_adam_foreach(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                accum_grads,
                state_steps,
                beta1,
                beta2,
                group['lr'],
                group['weight_decay'],
                group['eps'],
                group['accum_iter']
            )

