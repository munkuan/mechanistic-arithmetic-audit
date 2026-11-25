import torch
import os
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd


from config import Config
from run_das import DASDataset, custom_collate_fn, build_intervenable, train_das
from evaluate import evaluate_iia

if __name__ == "__main__":
    config = Config()
    torch.manual_seed(config.seed)
    
    for model_name in config.model_name:
        print(f"\n{'='*50}")
        print(f"Loading Model: {model_name}")
        print(f"{'='*50}")
        
        # 1. Load Tokenizer and Model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=config.dtype, device_map="auto"
        )
        model.eval()
        
        # 2. Dynamically sweep layers based on model depth
        num_layers = model.config.num_hidden_layers
        start_layer = int(num_layers * 0.25)
        end_layer = int(num_layers * 0.75)
        layers_to_sweep = list(range(start_layer, end_layer))
        
        # 3. Setup Data
        train_dataset = DASDataset("train.csv", tokenizer)
        test_dataset = DASDataset("test.csv", tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate_fn)
        
        # 4. Sweep Configuration
        results = []
        for layer in layers_to_sweep:
            for pos in config.token_positions:
                print(f"\n--- Layer {layer}, Token Position {pos} ---")
                
                intervenable = build_intervenable(model, config, layer, pos, config.low_rank_dimension)
                
                # Train
                intervenable = train_das(intervenable, train_loader, config, layer, pos, tokenizer)
                
                # Evaluate
                test_iia = evaluate_iia(intervenable, test_loader, config, pos)
                print(f"Test IIA: {test_iia:.4f}")
                
                results.append({
                    "model": model_name,
                    "layer": layer, 
                    "pos": pos, 
                    "test_iia": test_iia
                })
                
                # Clean up memory for next iteration
                del intervenable
                torch.cuda.empty_cache()
                
        # 5. Save Results for this model
        os.makedirs("outputs", exist_ok=True)
        df = pd.DataFrame(results)
        df.to_csv(f"outputs/das_results_{model_name.split('/')[-1]}.csv", index=False)
        print(f"Saved results for {model_name}")