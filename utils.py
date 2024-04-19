#from pytorch_lightning.callbacks import EarlyStopping
# for some reason the above import doesn't work but the one below does...
from lightning.pytorch.callbacks import EarlyStopping

def early_stopping_via_train_loss(threshold, patience):
    custom_early_stop = EarlyStopping(
        monitor="train_loss",
        stopping_threshold=threshold,
        patience=patience,
        verbose=True,
        strict=True,
    )
    return custom_early_stop

def early_stopping_via_val_loss(patience):
    custom_early_stop = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0,
        patience=patience,
        verbose=True,
        mode="min",
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


# def impute_data(data_paths, temp_path, indices):
#     # Load data and impute within each fold
    
#     suffix = random.randint(1, 10000000)
#     new_data_paths = deepcopy(data_paths)
    
#     for tab in ["tabular1", "tabular2"]:
#         if data_paths[tab] != "":
#             df = pd.read_csv(data_paths[tab])
            
#             imputed_df = pd.DataFrame(index=df.index, columns=df.columns)
            
#             df_parts = []
#             for i, idx in enumerate(indices):
#                 df_part = df.iloc[idx]
#                 if i == 0:
#                     medians = df_part.median()
#                 df_part_filled = df_part.fillna(medians)
                
#                 imputed_df.iloc[idx] = df_part_filled
                
#             # save the imputed df with a new name
#             new_name = data_paths[tab].split("/")[-1].replace(".csv", f"_{str(suffix)}.csv")
#             imputed_df.to_csv(temp_path + new_name)
#             new_data_paths[tab] = temp_path + new_name
            
#     return new_data_paths

# def remove_imputed_data(data_paths):
#     for _, path in data_paths.items():
#         if path:  # Checks if the path is not empty
#             if os.path.isfile(path):  # Checks if the file exists
#                 os.remove(path)  # Deletes the file
