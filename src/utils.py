import torch
import numpy as np
import pandas as pd
import random
import torch_geometric
import copy
import pytorch_lightning as L
import time


def train_val_test(graphs):
    with open("IPA_results/train_set.txt") as f:
        train_ids_ipa = f.read().splitlines()
    with open("IPA_results/validation_set.txt") as f:
        val_ids_ipa = f.read().splitlines()
    with open("IPA_results/test_set.txt") as f:
        test_ids_ipa = f.read().splitlines()

    train_graphs = []
    val_graphs = []
    test_graphs = []
    for graph in graphs:
        if graph.mat_id in train_ids_ipa:
            train_graphs.append(graph)
        elif graph.mat_id in val_ids_ipa:
            val_graphs.append(graph)
        elif graph.mat_id in test_ids_ipa:
            test_graphs.append(graph)
        else:
            print("Something went wrong when building the sets!")
    return train_graphs, val_graphs, test_graphs


def model_eval(model, test_loader, data_df):
    # evaluate a model on a given test set
    mse_names = []
    mse_vals = []
    mae_vals = []
    mape_vals = []
    simils = []
    formulas = []
    ipa_rpa = []
    model.eval()
    if not test_loader.batch_size == 1:
        print("The Loader size is not equal to 1, this function will not work!")
        return
    for mini in test_loader:
        out = model(mini).flatten()
        curr_mse = torch.nn.functional.mse_loss(out, mini.y)
        curr_mse = curr_mse.cpu().detach().numpy()
        curr_mae = torch.nn.functional.l1_loss(out, mini.y)
        curr_mae = curr_mae.cpu().detach().numpy()
        curr_mape = torch.mean(
            torch.abs((mini.y - model(mini).flatten()) / (mini.y + 1e-16))
        )
        curr_mape = curr_mape.cpu().detach().numpy()
        mse_vals.append(curr_mse)
        mae_vals.append(curr_mae)
        mape_vals.append(curr_mape)
        mse_names.append(mini.mat_id)
        sc = 1 - np.trapz(
            np.abs(out.cpu().detach().numpy() - mini.y.cpu().detach().numpy())
        ) / np.trapz(abs(mini.y.cpu().detach().numpy()))
        simils.append(sc)
        try:
            sc_rpa_ipa = 1 - np.trapz(
                np.abs(mini.ipa.cpu().detach().numpy() - mini.y.cpu().detach().numpy())
            ) / np.trapz(abs(mini.y.cpu().detach().numpy()))
        except:
            sc_rpa_ipa = np.nan
        ipa_rpa.append(sc_rpa_ipa)
        formulas.append(
            data_df.loc[data_df["mat_id"] == mini.mat_id[0]].formula.values[0]
        )
    mse_df = pd.DataFrame(
        list(zip(mse_names, mse_vals, mae_vals, mape_vals, simils, formulas, ipa_rpa)),
        index=range(len(mse_names)),
        columns=["name", "mse", "mae", "mape", "sc", "formulas", "ipa_rpa"],
    )
    mse_df["mse"] = mse_df["mse"].astype(np.float32)
    mse_df["mae"] = mse_df["mae"].astype(np.float32)
    mse_df["mape"] = mse_df["mape"].astype(np.float32)
    mse_df["sc"] = mse_df["sc"].astype(np.float32)
    mse_df["ipa_rpa"] = mse_df["ipa_rpa"].astype(np.float32)
    return mse_df


def train_val_test_formula(data_df, graphs, seed):
    unique_formulas = np.unique(data_df.formula.values)
    np.random.seed(seed)
    np.random.shuffle(unique_formulas)
    length = len(unique_formulas)
    train_form = unique_formulas[: int(0.8 * length)]
    val_form = unique_formulas[int(0.8 * length) : int(0.9 * length)]
    test_form = unique_formulas[int(0.9 * length) :]
    len(train_form) + len(val_form) + len(test_form)
    train_list = []
    val_list = []
    test_list = []
    for graph in graphs:
        graph_id = graph.mat_id
        graph_form = data_df[data_df["mat_id"] == graph_id].formula.values[0]
        if graph_form in train_form:
            train_list.append(graph)
        elif graph_form in val_form:
            val_list.append(graph)
        elif graph_form in test_form:
            test_list.append(graph)
        else:
            print("Something went wrong in building the train/val/test sets")
    TrainLoader = torch_geometric.loader.DataLoader(
        train_list, batch_size=256, shuffle=True
    )
    ValLoader = torch_geometric.loader.DataLoader(val_list, batch_size=10, shuffle=True)
    TestLoader = torch_geometric.loader.DataLoader(
        test_list, batch_size=10, shuffle=True
    )
    return TrainLoader, ValLoader, TestLoader, train_list, val_list, test_list
