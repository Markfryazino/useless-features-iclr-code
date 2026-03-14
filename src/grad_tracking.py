from dataclasses import dataclass
import torch
from omegaconf import DictConfig
from typing import List, Tuple, Dict
import numpy as np
from collections import defaultdict

from src.data_gen import Sampler
from src.model import myopic_forward


@dataclass
class GradTracker:
    frozen_model: torch.nn.Module
    sampler: Sampler

    # (layer, feature_id, seq_idx) -> probe
    probes: Dict[Tuple[int, int, int], torch.Tensor]


def get_grad_tracker(
    model: torch.nn.Module,
    probes: List[List[np.array]],  # probes[layer][seq_idx]
    sampler: Sampler,
    cfg: DictConfig,
) -> GradTracker:
    if not cfg["verify_probes"]:
        return None
    
    probes_dict = {}

    for feature_name, layer, seq_idx in cfg["verify_probes"]:
        feature_id = sampler.get_feature_id(feature_name)
        probe = probes[layer][seq_idx]
        weight = torch.tensor(probe.coef_[feature_id], dtype=torch.float32, device=cfg["device"])
        probes_dict[(layer, feature_id, seq_idx)] = weight

    return GradTracker(frozen_model=model, sampler=sampler, probes=probes_dict)


def compute_adam_param_delta(gradients: List[torch.Tensor], full_gradients: torch.Tensor, moments: List[torch.Tensor], optimizer: torch.optim.Adam,):
    """
    Compute Adam parameter deltas without updating the optimizer state.
    
    Args:
        gradients: List of tensors, each tensor is gradients for one loss component
        optimizer: Adam optimizer with stored state
    
    Returns:
        List of parameter deltas (changes that would be applied to parameters), learning rate
    """
    param_deltas = [
        torch.zeros_like(g) for g in gradients
    ]

    if len(optimizer.param_groups) > 1:
        raise ValueError("Only one parameter group is supported")
    
    group = optimizer.param_groups[0]
    # Get hyperparameters
    beta1, beta2 = group['betas']
    lr = group['lr']
    eps = group['eps']
    weight_decay = group['weight_decay']

    if weight_decay != 0:
        raise ValueError("Weight decay is not supported")

    def compute_for_component(component_idx: int):
        offset = 0
        for param in group['params']:
            grad = gradients[component_idx][offset:offset+param.numel()]
            moment = moments[component_idx][offset:offset+param.numel()]
            combined_grad_param = full_gradients[offset:offset+param.numel()]
            
            # Get current state (don't modify it)
            state = optimizer.state[param]

            if len(state) == 0:
                exp_avg_sq = torch.zeros_like(param).flatten()
                step = 0
            else:
                exp_avg_sq = state['exp_avg_sq'].flatten()
                step = state['step']

            new_step = step + 1
            new_exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * combined_grad_param * combined_grad_param
            
            new_moment = beta1 * moment + (1 - beta1) * grad
            moments[component_idx][offset:offset+param.numel()] = new_moment

            corrected_new_moment = new_moment / (1 - beta1 ** new_step)
            corrected_second_moment = new_exp_avg_sq / (1 - beta2 ** new_step)
            
            # Compute parameter delta: Δθ = -α * m̂_t / (√v̂_t + ε)
            denom = corrected_second_moment.sqrt().add_(eps)
            param_delta = lr * corrected_new_moment / denom
            
            param_deltas[component_idx][offset:offset+param.numel()] = param_delta

            offset += param.numel()

    for i in range(len(gradients)):
        compute_for_component(i)

    return param_deltas, lr


@torch.no_grad()
def get_frozen_repr(x, frozen_model):
    logits, hiddens = frozen_model(x, return_hidden=True)
    return hiddens


def flatten_grads(model, optimizer):
    n_params = sum(p.numel() for p in model.parameters())

    if hasattr(model, "transformer"):
        device = next(model.transformer.parameters()).device
    else:
        device = next(model.transformers[0].parameters()).device
    
    grads = torch.zeros(n_params, device=device)
    
    offset = 0
    for p in model.parameters():
        new_offset = offset + p.numel()

        if p.grad is not None:
            grads[offset:new_offset] = p.grad.detach().flatten()
        offset = new_offset

    optimizer.zero_grad()

    return grads


def compute_probe_prediction(repr, layer, seq_idx, weight):
    return repr[layer][:, seq_idx] @ weight


def get_grad_r(x, hiddens, model, optimizer, grad_tracker: GradTracker, features_type: torch.Tensor, use_features_type: bool):
    frozen_repr = get_frozen_repr(x, grad_tracker.frozen_model)

    r_grads = {}
    risks = {}

    for (layer, feature_id, seq_idx), weight in grad_tracker.probes.items():
        probe_name = f"{grad_tracker.sampler.get_feature_name(feature_id)}_layer_{layer}_seq_{seq_idx}"
        frozen_prediction = compute_probe_prediction(frozen_repr, layer, seq_idx, weight)
        current_prediction = compute_probe_prediction(hiddens, layer, seq_idx, weight)
        risk = torch.nn.functional.mse_loss(current_prediction, frozen_prediction)
        risk.backward(retain_graph=True)
        r_grads[probe_name] = flatten_grads(model, optimizer)
        risks[probe_name] = risk.item()

        if use_features_type:
            unique_ft = torch.unique(features_type)
            for ft in unique_ft:
                ft_risk = torch.nn.functional.mse_loss(current_prediction[features_type[:, seq_idx, feature_id] == ft], frozen_prediction[features_type[:, seq_idx, feature_id] == ft])
                ft_risk.backward(retain_graph=True)
                ft_grads = flatten_grads(model, optimizer)
                r_grads[f"{probe_name}-ft-{ft}"] = ft_grads
                risks[f"{probe_name}-ft-{ft}"] = ft_risk.item()

    return r_grads, risks


def init_adam_moments(model, device):
    probe_grads_adam.moments = defaultdict(lambda: [torch.zeros(sum(p.numel() for p in model.parameters()), device=device) for _ in range(3)])

def probe_grads_adam(criterion, tokens, y_true, model, optimizer, grad_tracker: GradTracker, features_type: torch.Tensor, use_features_type: bool):
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    optimizer.zero_grad()

    seq_len = tokens.shape[1]

    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        no_reduction_criterion = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=0)
    else:
        raise ValueError(f"Unsupported criterion: {criterion}")

    _, non_myopic_hiddens = myopic_forward(model, myopic_on=False, x=tokens, return_hidden=True)
    logits = model(tokens)

    r_grads, risks = get_grad_r(tokens, non_myopic_hiddens, model, optimizer, grad_tracker, features_type, use_features_type)

    possible_seq_idx_plus_layer = list(set((seq_idx, layer) for layer, _, seq_idx in grad_tracker.probes.keys()))

    normal_loss = no_reduction_criterion(logits[:, :-1].transpose(1, 2), y_true).mean(dim=0)
    normal_loss_pooled = normal_loss.mean()
    normal_loss_pooled.backward(retain_graph=True)
    normal_grads = flatten_grads(model, optimizer)

    direct_proj = {}
    pre_cached_proj = {}
    shared_proj = {}
    full_proj = {}

    metrics = {"loss": normal_loss_pooled.item()}

    for k, v in risks.items():
        metrics[f"risk-{k}"] = v

    for seq_idx, layer in possible_seq_idx_plus_layer:
        sg_logits = model(tokens, detach_position=seq_idx, detach_layer=layer)
        sg_loss = no_reduction_criterion(sg_logits[:, :-1].transpose(1, 2), y_true).mean(dim=0)
        sg_loss_pooled = sg_loss.mean()
        sg_loss_pooled.backward(retain_graph=True)
        sg_grads = flatten_grads(model, optimizer)

        only_seq_idx_loss_normal = normal_loss[seq_idx] / (seq_len - 1)
        only_seq_idx_loss_normal.backward(retain_graph=True)
        only_seq_idx_grads_normal = flatten_grads(model, optimizer)

        only_seq_idx_loss_sg = sg_loss[seq_idx] / (seq_len - 1)
        only_seq_idx_loss_sg.backward(retain_graph=False)
        only_seq_idx_grads_sg = flatten_grads(model, optimizer)

        shared_grads = sg_grads
        direct_grads = only_seq_idx_grads_normal - only_seq_idx_grads_sg
        pre_cached_grads = normal_grads - sg_grads - direct_grads

        [adam_direct_delta, adam_pre_cached_delta, adam_shared_delta], lr = compute_adam_param_delta(
            [direct_grads, pre_cached_grads, shared_grads],
            normal_grads,
            probe_grads_adam.moments[(seq_idx, layer)],
            optimizer
        )

        deltas_by_type = {
            "sgd": [direct_grads * lr, pre_cached_grads * lr, shared_grads * lr],
            "adam": [adam_direct_delta, adam_pre_cached_delta, adam_shared_delta],
        }

        for type, deltas in deltas_by_type.items():
            direct_delta, pre_cached_delta, shared_delta = deltas

            total = direct_delta + pre_cached_delta + shared_delta
            for k in r_grads.keys():
                full_proj[f"{k}-{type}"] = (total @ r_grads[k]).item()

            metrics[f"{type}--grad-norm-layer-{layer}-{seq_idx}-shared"] = torch.norm(shared_delta).item()
            metrics[f"{type}--grad-norm-layer-{layer}-{seq_idx}-direct"] = torch.norm(direct_delta).item()
            metrics[f"{type}--grad-norm-layer-{layer}-{seq_idx}-pre-cached"] = torch.norm(pre_cached_delta).item()

            for (probe_layer, feature_id, probe_seq_idx), weight in grad_tracker.probes.items():
                if probe_seq_idx != seq_idx or probe_layer != layer:
                    continue

                probe_name = f"{grad_tracker.sampler.get_feature_name(feature_id)}_layer_{probe_layer}_seq_{probe_seq_idx}"
                shared_proj[f"{probe_name}-{type}"] = (shared_delta @ r_grads[probe_name]).item()
                direct_proj[f"{probe_name}-{type}"] = (direct_delta @ r_grads[probe_name]).item()
                pre_cached_proj[f"{probe_name}-{type}"] = (pre_cached_delta @ r_grads[probe_name]).item()

                if use_features_type:
                    unique_ft = torch.unique(features_type)
                    for ft in unique_ft:
                        ft_probe_name = f"{probe_name}-ft-{ft}"
                        shared_proj[f"{ft_probe_name}-{type}"] = (shared_delta @ r_grads[ft_probe_name]).item()
                        direct_proj[f"{ft_probe_name}-{type}"] = (direct_delta @ r_grads[ft_probe_name]).item()
                        pre_cached_proj[f"{ft_probe_name}-{type}"] = (pre_cached_delta @ r_grads[ft_probe_name]).item()

    for probe_name in direct_proj.keys():
        metrics[f"{probe_name}--direct"] = direct_proj[probe_name]
        metrics[f"{probe_name}--pre-cached"] = pre_cached_proj[probe_name]
        metrics[f"{probe_name}--shared"] = shared_proj[probe_name]
        metrics[f"{probe_name}--full"] = full_proj[probe_name]

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    return metrics