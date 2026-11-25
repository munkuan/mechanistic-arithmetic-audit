import torch

class Config:
    # List of Qwen models to test 
    model_name = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B-Instuct-2507"
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # DAS hyperparameters
    # Layers will need to be specified for each model in the model_name list
    token_positions = [-1, -2, -3]
    low_rank_dimension = 1          
    learning_rate = 1e-2
    batch_size = 16
    num_warmup_steps = 100
    num_train_steps = 1000
    temperature_start = 50.0
    temperature_end = 0.1

    # Data
    data_file = "data_pairs.csv"

    seed = 42  # For reproducibility    
    