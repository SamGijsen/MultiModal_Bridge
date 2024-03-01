import fusilli
import os, argparse
import sys
import importlib
import ast

import numpy as np
import pandas as pd

from datetime import datetime
from copy import deepcopy

from fusilli.fusionmodels.tabularfusion.crossmodal_att import TabularCrossmodalMultiheadAttention
from fusilli.fusionmodels.tabularfusion.channelwise_att import TabularChannelWiseMultiAttention
from fusilli.fusionmodels.tabularfusion.concat_data import ConcatTabularData
from fusilli.fusionmodels.tabularfusion.concat_feature_maps import ConcatTabularFeatureMaps
from fusilli.fusionmodels.unimodal.image import ImgUnimodal
from fusilli.fusionmodels.unimodal.tabular1 import Tabular1Unimodal
from fusilli.fusionmodels.unimodal.tabular2 import Tabular2Unimodal
from fusilli.data import prepare_fusion_data
from fusilli.train import train_and_save_models, train_and_test
from fusilli.eval import RealsVsPreds, ParentPlotter

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, ParameterGrid

import pytorch_lightning as pl 
from pytorch_lightning.callbacks import EarlyStopping


def early_stopping_via_train_loss(threshold):
    custom_early_stop = EarlyStopping(
        monitor="train_loss",
        stopping_threshold=threshold,
        verbose=True,
        strict=True,
    )
    return custom_early_stop

def get_sorted_dict_str(d):
    return str(sorted(d.items()))

def dict_to_str(d):
    return '_'.join(f"{key}_{value}" for key, value in d.items())

def training_modifications(GPU_devices):
    if GPU_devices:
        if not isinstance(GPU_devices, list):
            GPU_devices = [GPU_devices]
        training_mod = dict(
            accelerator= "gpu",
            devices= GPU_devices
        )
    else:
        training_mod = None
    return training_mod
        
def main():
    # load the config file
    parser = argparse.ArgumentParser('runMLpipelines Setting')
    parser.add_argument('config', type=str, help='please add the configuration name')
    option = parser.parse_args()
    name = option.config
    pkg = importlib.import_module(f'config.{name}')
    cfg = pkg.Config
    nested = cfg.N_INNER_CV>1

    start_time = datetime.now()

    tab1_name = os.path.basename(cfg.DATA_PATHS["tabular1"]).split(".")[-2]

    # Create the folder in which to save the results
    SAVE_DIR = f'{cfg.OUTPUT_PATH}output/{tab1_name}/{start_time.strftime("%Y%m%d-%H%M")}/'
    if not os.path.isdir(SAVE_DIR): os.makedirs(SAVE_DIR)
    # Save the configuration file in the folder
    # cpy_cfg = f'{SAVE_DIR}/{name}_{start_time.strftime("%Y%m%d_%H%M")}.py'
    # os.system(f"cp {cfg_path} {cpy_cfg}")

    pl.seed_everything(cfg.SEED, workers=True)
    rng = np.random.default_rng(seed=cfg.SEED)
    seeds = rng.integers(0, 1000, cfg.N_REPEATS)

    # load data and fetch the labels
    tab1_path = cfg.DATA_PATHS["tabular1"]
    df = pd.read_csv(tab1_path)

    for analysis in cfg.ANALYSIS:
        params = cfg.ANALYSIS[analysis]
        
        training_mod = training_modifications(params["GPU"])
        
        y = params["LABEL"]

        labels = df["prediction_label"].values

        labels_nan = np.isnan(labels)
        if np.sum(labels_nan)>0:
            print(f"Removing {np.sum(labels_nan)} subjects due to NaNs in label {y}")
            labels = labels[~labels_nan]
        X_n_samples = np.zeros(len(labels))
        
        for nreps in range(cfg.N_REPEATS):
            if nreps==1:
                break
            
            # set up the outer split
            splitter = KFold(n_splits=cfg.N_OUTER_CV, shuffle=True, random_state=seeds[nreps])
            # train_idxs = [test_idx for _,test_idx in splitter.split(X_n_samples, y=labels)]
            # test_idxs = [test_idx for _,test_idx in splitter.split(X_n_samples, y=labels)]
            train_idxs, test_idxs = zip(*[(train_idx, test_idx) for train_idx, test_idx in splitter.split(X_n_samples, y=labels)])
            
            for fold, (train_idx, test_idx) in enumerate(zip(train_idxs, test_idxs)):
                data_mask = np.zeros(len(labels_nan))
                data_mask[test_idx] = True
                train_labels = labels[train_idx]
                
                # Split the datafiles into a train+val and test set-up and save their locations.
                # Only necessary if we're doing nested-cv (i.e., N_INNER_CV>1)
                # train_data_paths, test_data_paths = deepcopy(cfg.DATA_PATHS), deepcopy(cfg.DATA_PATHS)
                # for key, data_path in cfg.DATA_PATHS.items():
                #     data = pd.read_csv(data_path)
                #     data = data.iloc[~labels_nan]
                #     data_test = data.iloc[data_mask]
                #     data_train = data.drop(~data_mask)
                    
                #     test_path = f"{cfg.INTERMEDIARY_DATA_PATH}+{data_path.split('/')[-1]}".replace(".csv", "_test.csv")
                #     train_path = f"{cfg.INTERMEDIARY_DATA_PATH}+{data_path.split('/')[-1]}".replace(".csv", "_train.csv")
                #     data_test.to_csv(test_path)
                #     data_train.to_csv(train_path)
                #     test_data_paths[key] = test_path
                #     train_data_paths[key] = train_path
            
                fusion_model = getattr(sys.modules[__name__], params["FUSILLI_MODEL"])

                print("-"*39)
                print(f"Repetition {nreps} | Fold {fold}")
                print("method_name:", fusion_model.method_name)
                print("modality_type:", fusion_model.modality_type)
                print("fusion_type:", fusion_model.fusion_type)
                    
                # prepare the 'inner_loop' train/val splits
                seeds_inner = rng.integers((fold+1)*1000, (fold+2)*1000, cfg.N_INNER_CV)
                if nested:

                    if params["PREDICTION_TASK"] == "binary":
                        kf = StratifiedKFold(n_splits=cfg.N_INNER_CV, shuffle=True, random_state=seeds_inner[0])
                    else:
                        kf = KFold(n_splits=cfg.N_INNER_CV, shuffle=True, random_state=seeds_inner[0], stratify=train_labels)
                    own_kfold_indices = [
                        (train_indices, test_indices) for train_indices, test_indices in kf.split(np.zeros(len(train_labels)), y=train_labels)]
                else: 
                    inner_train_idx, inner_val_idx = train_test_split(train_idx, test_size=0.25, 
                                                                    random_state=seeds_inner[0], shuffle=True, stratify=train_labels)
                    own_kfold_indices = [(inner_train_idx, inner_val_idx)]
                    

                # data_paths = train_data_paths #if nested else cfg.DATA_PATHS
                # test_paths = train_data_paths #if nested else cfg.DATA_PATHS
            
                # Generate the parameter grid
                param_grid = ParameterGrid(params["GRID"])
                if not list(param_grid):
                    # If the GRID is empty, use the default params for a single iteration
                    param_combinations = [params]
                else:
                    param_combinations = [{**params, **hp_combo} for hp_combo in param_grid]
                    
                # Keep track of the cumulative loss for each parameter combination
                cumulative_val_loss = {get_sorted_dict_str(hp_combo): 0 for hp_combo in param_grid}
                cumulative_train_loss = {get_sorted_dict_str(hp_combo): 0 for hp_combo in param_grid}
                cumulative_epochs = {get_sorted_dict_str(hp_combo): 0 for hp_combo in param_grid}

                for hp_iter, hp_combo in enumerate(param_combinations):
                    extra_suffix_dict = {k: hp_combo[k] for k in params["GRID"].keys() if k in hp_combo}
                    # each parameter condition needs to be tested on each inner fold
                    
                    for inner_fold in range(len(own_kfold_indices)):
                        
                        # set paths for this fold
                        m_name = f"{analysis}_{params['FUSILLI_MODEL']}"
                        base_output_path = f"{SAVE_DIR}/{m_name}/rep_{nreps}/outfold_{fold}"
                        output_paths = { # do I want to name these based on n_rep?
                            "losses": f"{base_output_path}/infold_{inner_fold}/val/loss_logs",
                            "checkpoints": f"{base_output_path}/infold_{inner_fold}/val/checkpoints",
                            "figures": f"{base_output_path}/infold_{inner_fold}/val/figures",
                        }
                        for path in output_paths.values():
                            os.makedirs(path, exist_ok=True)
                
                        dm = prepare_fusion_data(prediction_task=params["PREDICTION_TASK"],
                                                fusion_model=fusion_model,
                                                data_paths=cfg.DATA_PATHS, 
                                                output_paths=output_paths,
                                                kfold=True,
                                                max_epochs=params["MAX_EPOCHS"],
                                                num_folds=cfg.N_INNER_CV,
                                                own_kfold_indices=own_kfold_indices, 
                                                batch_size=params["BATCH_SIZE"],
                                                extra_log_string_dict=extra_suffix_dict,
                                                learning_rate=hp_combo["LEARNING_RATE"],
                                                weight_decay=hp_combo["WEIGHT_DECAY"],
                                                num_workers=params["NUM_WORKERS"])
                        
                        trained_models = train_and_save_models(
                            data_module=dm,
                            max_epochs=params["MAX_EPOCHS"],
                            fusion_model=fusion_model,
                            extra_log_string_dict=extra_suffix_dict,
                            enable_checkpointing=True,
                            show_loss_plot=False,
                            training_modifications=training_mod
                        )
                        del dm, trained_models
                        
                        # fetch the metrics and keep a record of model performance and training stats
                        key = get_sorted_dict_str(extra_suffix_dict)
                        metrics_path = os.path.join(os.getcwd(), output_paths["losses"], 
                                f"{params['FUSILLI_MODEL']}_fold_{inner_fold}_{dict_to_str(extra_suffix_dict)}",  "metrics.csv")
                        metrics = pd.read_csv(metrics_path)
                        cumulative_val_loss[key] += metrics["val_loss"].min()
                        cumulative_epochs[key] += metrics["epoch"][metrics["val_loss"].idxmin()]
                        cumulative_train_loss[key] += metrics["train_loss"].min()
                        
                # if non-nested, we just need to check what our target loss is.
                # if nested, we also need to check our best hyper-parameter
                best_val_loss = np.inf
                best_hp_key = None
                for k,v in cumulative_val_loss.items():
                    if v < best_val_loss:
                        best_val_loss = v 
                        best_hp_key = k
                        target_epoch = cumulative_epochs[k]/cfg.N_INNER_CV
                        target_train_loss = cumulative_train_loss[k]/cfg.N_INNER_CV
                                                
                best_hp_dict = dict(ast.literal_eval(best_hp_key))
                hp_dict = {**params, **best_hp_dict}                    
                
                # Refit on training + validation data
                
                # !!! THIS IS NOT POSSIBLE UNTIL WE CAN SAVE AND LOAD THE BEST_VAL MODEL
                # Check the training loss on the checkpoint, which will be our early stop threshold
                # checkpoint_path = os.path.join(os.getcwd(), output_paths["checkpoints"])
                # checkpoints = os.listdir(checkpoint_path)
                # target_loss = metrics[(metrics["epoch"] == target_epoch) & (metrics["train_loss"].notnull())]["train_loss"].values[0]
                # metrics_path = os.path.join(os.getcwd(), output_paths["losses"], "metrics.csv")
                # metrics = pd.read_csv(metrics)

                # !!! FOR THE MOMENT, WE FULLY RETRAIN THE MODEL
                # But for how long? Until train loss associated with best val loss is reached
                # metrics_path = os.path.join(os.getcwd(), output_paths["losses"], 
                #                             f"{params['FUSILLI_MODEL']}_fold_{inner_fold}",  "metrics.csv")
                # # TODO, instead of fold_0, we need to select the 'best' hyperparameter fold actually
                # metrics = pd.read_csv(metrics_path)
                # best_epoch = metrics["epoch"][metrics["val_loss"].idxmin()] # based on lowest val loss
                # target_train_loss = metrics[metrics["epoch"]==best_epoch]["train_loss"].values[-1]
                # print("INNER FOLD", inner_fold)
                # print("best epoch and target loss", best_epoch, target_train_loss)
                # print("max epochs:", 2*best_epoch)
                
                output_paths = { 
                    "losses": f"{SAVE_DIR}/{m_name}/rep_{nreps}/outfold_{fold}/loss_logs",
                    "checkpoints": f"{SAVE_DIR}/{m_name}/rep_{nreps}/outfold_{fold}/checkpoints",
                    "figures": f"{SAVE_DIR}/{m_name}/rep_{nreps}/outfold_{fold}/figures",
                }
                for path in output_paths.values():
                    os.makedirs(path, exist_ok=True)
                    
                # custom early stopping with KFold is not yet supported. Thus, we need to use custom test_indices
                            
                dm_test = prepare_fusion_data(prediction_task=params["PREDICTION_TASK"],
                            fusion_model=fusion_model,
                            data_paths=cfg.DATA_PATHS, 
                            output_paths=output_paths,
                            max_epochs=max(2, int(target_epoch*2)), # twice the amount of epochs needed previously
                            test_indices = test_idx,
                            batch_size=params["BATCH_SIZE"],
                            extra_log_string_dict=best_hp_dict,
                            own_early_stopping=early_stopping_via_train_loss(target_train_loss),
                            learning_rate=hp_dict["LEARNING_RATE"],
                            weight_decay=hp_dict["WEIGHT_DECAY"],
                            num_workers=params["NUM_WORKERS"])
                
                test_models = train_and_save_models(
                    data_module=dm_test,
                    max_epochs=max(2, int(target_epoch*2)),
                    fusion_model=fusion_model,
                    enable_checkpointing=False,
                    extra_log_string_dict=extra_suffix_dict,
                    show_loss_plot=False,
                    learning_rate=hp_dict["LEARNING_RATE"],
                    weight_decay=hp_dict["WEIGHT_DECAY"],
                    training_modifications=training_mod)
                
if __name__ == "__main__": main()