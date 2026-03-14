import torch
import tqdm
from typing import Tuple

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import numpy as np


def extract_hidden_states(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.train()

    all_features_list = []
    all_hiddens_list = []
    all_features_type_list = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Extracting hidden states"):
            token_ids = batch["token_ids"].to(device)
            features = batch["features"].to(device)
            features_type = batch["features_type"].to(device)

            _, hiddens = model(token_ids, return_hidden=True)

            all_hiddens_list.append(hiddens)
            all_features_list.append(features)
            all_features_type_list.append(features_type)

    all_hiddens = torch.cat(all_hiddens_list, dim=1)
    all_features = torch.cat(all_features_list, dim=0)
    all_features_type = torch.cat(all_features_type_list, dim=0)

    return all_hiddens, all_features, all_features_type


def fit_predict(X_train, y_train, X_test, y_test, y_train_type, y_test_type, model=None, alpha=100.0):
    if model is None:
        model = Ridge(alpha=alpha).fit(X_train, y_train)
    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)

    pearsons = []
    if len(pred.shape) == 1:
        pred = pred.reshape(-1, 1)

    for dim in range(y_test.shape[1]):
        pearsons.append(pearsonr(y_test[:, dim], pred[:, dim])[0])

    type_pearsons = []
    # now aggregated over feature types
    for dim in range(y_test.shape[1]):
        type_pearsons.append({})
        for type in np.unique(y_test_type):
            mask = y_test_type[:, dim] == type
            if len(np.unique(y_test[mask, dim])) > 1:
                type_pearsons[dim][type] = pearsonr(y_test[mask, dim], pred[mask, dim])[0]

    return mse, np.mean(pearsons), pearsons, type_pearsons, model


def probing_by_position(features, hiddens, features_type, seq_len, coord_names, probe_alpha=100.0):
    test_ratio = 0.3
    train_size = int(len(features) * (1 - test_ratio))

    y_train = features[:train_size].cpu().numpy()
    y_test = features[train_size:].cpu().numpy()
    y_train_type = features_type[:train_size].cpu().numpy()
    y_test_type = features_type[train_size:].cpu().numpy()

    metrics = {}
    probes = []

    for layer in range(hiddens.size(0)):
        probes.append([None])
        X_train = hiddens[layer, :train_size].cpu().numpy()
        X_test = hiddens[layer, train_size:].cpu().numpy()

        # do probing (omitting seq_idx=0 because it's the <bos> token)
        for seq_idx in range(1, seq_len):
            mse, pearson, pearsons, type_pearsons, probe = fit_predict(X_train[:, seq_idx], y_train[:, seq_idx], X_test[:, seq_idx], y_test[:, seq_idx], y_train_type[:, seq_idx], y_test_type[:, seq_idx], alpha=probe_alpha)
            probes[-1].append(probe)
            metrics[f"mse_layer_{layer}_{seq_idx}"] = mse
            metrics[f"pearson_layer_{layer}_{seq_idx}"] = pearson

            for current_pearson, coord_name in zip(pearsons, coord_names):
                metrics[f"pearson_{coord_name}_layer_{layer}_{seq_idx}"] = current_pearson

            for current_pearson, coord_name in zip(type_pearsons, coord_names):
                for type, pearson in current_pearson.items():
                    metrics[f"pearson_{coord_name}_layer_{layer}_{seq_idx}_type_{type}"] = pearson

        # also log averaged metrics
        metrics[f"mse_layer_{layer}_avg"] = np.mean([metrics[f"mse_layer_{layer}_{seq_idx}"] for seq_idx in range(2, seq_len)])
        metrics[f"pearson_layer_{layer}_avg"] = np.mean([metrics[f"pearson_layer_{layer}_{seq_idx}"] for seq_idx in range(2, seq_len)])

    return metrics, probes
