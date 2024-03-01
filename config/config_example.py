class Config:

    # TABULAR
    DATA_PATHS = {
        "tabular1": "/ritter/share/projects/sam/temp/mnist1_binary.csv",
        "tabular2": "/ritter/share/projects/sam/temp/mnist2_binary.csv",
        "image": "",
        "confs": "" # [ID, name_conf1, name_conf2, name_conf3]
    }
    
    # UNIMODAL 3D-IMAGES
    # DATA_PATHS = {
    #     "tabular1": "/ritter/share/projects/sam/temp/ADNI_labels_60.csv",
    #     "tabular2": "",
    #     "image": "/ritter/share/projects/sam/temp/ADNI_images_smaller_60.pt",
    #     "confs": "" # [ID, name_conf1, name_conf2, name_conf3]
    # }
    
    #OUTPUT_PATH = "/ritter/share/projects/sam/fusilli/"
    OUTPUT_PATH = "/home/sam/code/Multimodal_Bridge/notebooks/"
    
    ANALYSIS = {
        'classification_TABULAR' : dict(
            # LABEL='',
            # TASK_TYPE='classification',
            # METRICS='balanced_accuracy',
            
            # Fusilli parameters
            GPU=[0], # Set to None in case of CPU
            PREDICTION_TASK="binary",
            FUSILLI_MODEL= "TabularCrossmodalMultiheadAttention", #"Tabular2Unimodal"
            BATCH_SIZE=256,
            LEARNING_RATE=1e-4,
            WEIGHT_DECAY=1e-8,
            MAX_EPOCHS=100,
            NUM_WORKERS=4,
            
            GRID= dict(
                WEIGHT_DECAY= [1e-4, 1e-8, 1e-12, 0.],
            )
            )
    }
    
    # ANALYSIS = {
    #     'classification_IMAGES' : dict(
    #         LABEL='',
    #         TASK_TYPE='classification',
    #         METRICS='balanced_accuracy',

    #         # Fusilli parameters
    #         GPU=[0],
    #         PREDICTION_TASK="binary",
    #         FUSILLI_MODEL= "ImgUnimodal", 
    #         BATCH_SIZE=6,
    #         LEARNING_RATE=1e-4,
    #         WEIGHT_DECAY=0.,
    #         MAX_EPOCHS=100,
    #         NUM_WORKERS=4,
            
    #         GRID= dict(
    #             #WEIGHT_DECAY= [1e-4, 1e-8, 1e-12, 0.],
    #             LEARNING_RATE=[1e-2, 3e-3, 1e-3, 3e-4]
    #         )
    #         )
    # }
    
    
    # SETTING
    SEED = 42
    N_REPEATS = 2
    N_INNER_CV = 2
    N_OUTER_CV = 3