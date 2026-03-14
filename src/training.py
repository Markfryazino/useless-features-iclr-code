import torch
import logging
import wandb
import tqdm
from collections import defaultdict
from omegaconf import DictConfig
from typing import Optional, Dict

from src.data_gen import TrainEvalDataset
from src.grad_tracking import GradTracker, probe_grads_adam, init_adam_moments
from src.probing import extract_hidden_states, probing_by_position

from torch.utils.data import DataLoader
from typing import Iterator


def avg_with_mask(x, mask):
    return (x * mask).sum() / mask.sum()


# this is our main function that takes inputs and predictions and returns metrics
def get_loss_acc(criterion, logits, tokens, legal_tokens, loss_mask):

    tokens_y = tokens[:, 1:].clone()
    # Set ignored targets to -100, matching CrossEntropyLoss(ignore_index=-100)
    tokens_y[loss_mask[:, 1:] == 0] = -100

    loss = criterion(logits[:, :-1].transpose(1, 2), tokens_y)

    if legal_tokens.numel() > 0:
        prediction = logits[:, :-1].argmax(dim=-1)
        legal_pred = torch.gather(legal_tokens, 2, prediction.unsqueeze(-1)).squeeze(-1)
        legal_acc = avg_with_mask((legal_pred == 1).float(), loss_mask[:, 1:])
    else:
        legal_acc = 0.0

    return loss, legal_acc


def get_new_batch(data_iter: Iterator[Dict], train_dataloader: DataLoader, epoch: int):
    try:
        batch = next(data_iter)
    except StopIteration:
        epoch += 1
        data_iter = iter(train_dataloader)
        batch = next(data_iter)

    return batch, data_iter, epoch


def add_prefix_to_metrics(metrics: Dict[str, float], prefix: str):
    return {f"{prefix}/{k}": v for k, v in metrics.items()}


def log_training_metrics(
    step: int,
    epoch: int,
    logging_loss: float,
    legal_acc: float,
    batch_size: int,
    custom_metrics: Dict[str, float]
):
    if wandb.run:
        wandb.log({
            "train/loss": logging_loss,
            "train/epoch": epoch,
            "train/accuracy_legal": legal_acc,
            "processed_samples": (step + 1) * batch_size,
            "global_step": step + 1,
        }, step=step + 1)

        wandb.log(add_prefix_to_metrics(custom_metrics, "train"), step=step + 1)

    logging.info(f"Step {step + 1}: loss {logging_loss:10.3f}, "
            f"accuracy_legal: {legal_acc:5.2f}, "
            f"custom_metrics: {custom_metrics}, "
            f"epoch: {epoch}")

    return 0


def process_grad_tracking_step(
    criterion: torch.nn.Module,
    x: torch.Tensor,
    y_true: torch.Tensor,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_tracker: GradTracker,
    cumulative_grad_tracking_metrics: Dict[str, float],
    features_type: torch.Tensor,
    use_features_type: bool,
    step: int
):
    grad_tracking_metrics = probe_grads_adam(
        criterion, x, y_true, model, optimizer, grad_tracker, features_type, use_features_type
    )
    
    for k, v in grad_tracking_metrics.items():
        cumulative_grad_tracking_metrics[k] += v

    if wandb.run:
        wandb.log(add_prefix_to_metrics(grad_tracking_metrics, "grad_tracking"), step=step + 1)
        wandb.log(add_prefix_to_metrics(cumulative_grad_tracking_metrics, "grad_tracking_total"), step=step + 1)


def process_probing_step(
    model: torch.nn.Module,
    validation_dataloader: DataLoader,
    device: str,
    dataset: TrainEvalDataset,
    probe_alpha: float,
    step: int
):
    all_hiddens, all_features, all_features_type = extract_hidden_states(model, validation_dataloader, device)
    metrics, probes = probing_by_position(
        all_features, all_hiddens, all_features_type, dataset.sampler.get_max_len(), 
        dataset.sampler._get_feature_names(), probe_alpha
    )
    if wandb.run:
        wandb.log(add_prefix_to_metrics(metrics, "probing"), step=step + 1)
    
    return probes


def eval(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    dataset: TrainEvalDataset,
) -> Dict[str, float]:
    model.eval()
    
    total_loss = 0.0
    total_legal_acc = 0.0
    num_batches = 0
    custom_metrics = defaultdict(float)
    
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Evaluating"):
            token_ids = batch["token_ids"].to(device)
            legal_tokens = batch["legal_tokens"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            
            logits = model(token_ids)
            loss, legal_acc = get_loss_acc(criterion, logits, token_ids, legal_tokens, loss_mask)
            
            total_loss += loss.item()
            total_legal_acc += legal_acc
            num_batches += 1

            for k, v in dataset.sampler.get_custom_metrics(logits, token_ids, legal_tokens).items():
                custom_metrics[k] += v
    
    metrics = {
        "loss": total_loss / num_batches,
        "accuracy_legal": total_legal_acc / num_batches,
    }

    for k, v in custom_metrics.items():
        metrics[k] = v / num_batches

    model.train()
    
    return metrics


def process_eval_step(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    step: int,
    dataset: TrainEvalDataset,
) -> Dict[str, float]:
    metrics = eval(model, criterion, dataloader, device, dataset)
    
    if wandb.run:
        wandb.log(add_prefix_to_metrics(metrics, "eval"), step=step + 1)
    
    logging.info(f"Step {step + 1} - Eval: loss {metrics['loss']:10.3f}, "
                f"accuracy_legal: {metrics['accuracy_legal']:5.2f}")
    
    return metrics


def final_grad_tracking_step(
    criterion: torch.nn.Module,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_tracker: GradTracker,
    cumulative_grad_tracking_metrics: Dict[str, float],
    data_iter: Iterator[Dict],
    train_dataloader: DataLoader,
    epoch: int,
    step: int,
    device: str,
    use_features_type: bool
):
    batch, data_iter, epoch = get_new_batch(data_iter, train_dataloader, epoch)
    x = batch["token_ids"].to(device)
    y_true = x[:, 1:].clone()
    y_true[batch["loss_mask"][:, 1:] == 0] = -100
    features_type = batch["features_type"].to(device)

    process_grad_tracking_step(
        criterion, x, y_true, model, optimizer, grad_tracker, 
        cumulative_grad_tracking_metrics, features_type, use_features_type, step
    )


def train(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset: TrainEvalDataset,
    cfg: DictConfig,
    grad_tracker: Optional[GradTracker] = None,
):
    device = cfg["device"]
    probes = None

    model.train()
    logging.info(f"Training model on {device}")

    logging_loss = legal_acc = epoch = 0
    custom_metrics = defaultdict(float)
    cumulative_grad_tracking_metrics = defaultdict(float)

    is_iterable_train = isinstance(dataset.train_dataset, torch.utils.data.IterableDataset)
    train_dataloader = torch.utils.data.DataLoader(
        dataset.train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False if is_iterable_train else True,
        drop_last=True
    )
    validation_dataloader = torch.utils.data.DataLoader(
        dataset.eval_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False
    )
    data_iter = iter(train_dataloader)

    if grad_tracker is not None:
       init_adam_moments(model, device)

    for step in tqdm.trange(cfg["num_steps"]):
        batch, data_iter, epoch = get_new_batch(data_iter, train_dataloader, epoch)

        x = batch["token_ids"].to(device)
        legal = batch["legal_tokens"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        features_type = batch["features_type"].to(device)

        y_true = x[:, 1:].clone()
        y_true[loss_mask[:, 1:] == 0] = -100

        if grad_tracker is not None:
            process_grad_tracking_step(
                criterion, x, y_true, model, optimizer, grad_tracker, 
                cumulative_grad_tracking_metrics, features_type, cfg["grad_track_feature_types"], step
            )

        logits = model(x)
        current_loss, current_legal_acc = get_loss_acc(criterion, logits, x, legal, loss_mask)
        logging_loss += current_loss.item() / cfg["log_interval"]
        legal_acc += current_legal_acc / cfg["log_interval"]

        for k, v in dataset.sampler.get_custom_metrics(logits, x, legal).items():
            custom_metrics[k] += v / cfg["log_interval"]

        current_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (step + 1) % cfg["log_interval"] == 0:
            logging_loss = legal_acc = log_training_metrics(
                step, epoch, logging_loss, legal_acc, cfg["batch_size"], custom_metrics
            )
            custom_metrics = defaultdict(float)

        if (step + 1) % cfg["probe_interval"] == 0:
            probes = process_probing_step(
                model, validation_dataloader, device, dataset, cfg["probe_alpha"], step
            )

        if (step + 1) % cfg["eval_interval"] == 0:
            process_eval_step(
                model, criterion, validation_dataloader, device, step, dataset
            )
    
    if grad_tracker is not None:
        final_grad_tracking_step(
            criterion, model, optimizer, grad_tracker, cumulative_grad_tracking_metrics, 
            data_iter, train_dataloader, epoch, cfg["num_steps"], device, cfg["grad_track_feature_types"]
        )

    return model, probes
