import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from tqdm import tqdm
import pandas as pd

from pyvene import (
    IntervenableModel,
    IntervenableConfig,
    RepresentationConfig,
    BoundlessRotatedSpaceIntervention,
)

from config import Config

class DASDataset(Dataset):
    def __init__(self, data_file, tokenizer):
        self.data = pd.read_csv(data_file)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        max_length = 128
        base_prompt = self.data.iloc[idx]["Base prompt"]
        source_prompt = self.data.iloc[idx]["Source prompt"]

        base_encoding = self.tokenizer(base_prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
        source_encoding = self.tokenizer(source_prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)

        expected_output_str = str(self.data.iloc[idx]["Expected Output After Carry"])
        expected_token_ids = self.tokenizer(expected_output_str, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)

        return {
            "base_input_ids": base_encoding["input_ids"].squeeze(),
            "base_attention_mask": base_encoding["attention_mask"].squeeze(),
            "source_input_ids": source_encoding["input_ids"].squeeze(),
            "source_attention_mask": source_encoding["attention_mask"].squeeze(),
            "expected_base_output": expected_token_ids
        }

def custom_collate_fn(batch):
    base_input_ids = torch.stack([item["base_input_ids"] for item in batch])
    base_attention_mask = torch.stack([item["base_attention_mask"] for item in batch])
    source_input_ids = torch.stack([item["source_input_ids"] for item in batch])
    source_attention_mask = torch.stack([item["source_attention_mask"] for item in batch])
    expected_base_output = torch.stack([item["expected_base_output"][0] for item in batch])
    return {
        "base_input_ids": base_input_ids,
        "base_attention_mask": base_attention_mask,
        "source_input_ids": source_input_ids,
        "source_attention_mask": source_attention_mask,
        "expected_base_output": expected_base_output
    }

def build_intervenable(model, config, layer, token_position, rank_dim):
    intervenable_config = IntervenableConfig(
        model_type=type(model),
        representations=[
            RepresentationConfig(
                layer=layer,
                component="block_output",
                unit="pos",
                max_number_of_units=1,
                low_rank_dimension=rank_dim,
                intervention_type=BoundlessRotatedSpaceIntervention,
            )
        ],
        intervention_types=BoundlessRotatedSpaceIntervention,
    )
    intervenable = IntervenableModel(intervenable_config, model)
    intervenable.set_device(config.device)
    intervenable.disable_model_gradients()
    return intervenable

def train_das(intervenable, train_loader, config, layer, token_position, tokenizer):
    params = []
    for k,v in intervenable.interventions.items():
        params.extend(v.parameters())

    optimizer = torch.optim.Adam(params, lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=len(train_loader) * config.num_train_steps
    )
    criterion = nn.CrossEntropyLoss()
    temp_scheduler = torch.linspace(
        config.temperature_start, config.temperature_end, config.num_train_steps
    ).to(config.device)

    global_step = 0
    total_correct = 0
    total_count = 0

    for epoch in range(1000):
        for batch in train_loader:
            if global_step >= config.num_train_steps:
                break

            base_input_ids = batch["base_input_ids"].to(config.device)
            base_attention_mask = batch["base_attention_mask"].to(config.device)
            source_input_ids = batch["source_input_ids"].to(config.device)
            source_attention_mask = batch["source_attention_mask"].to(config.device)

            expected_token_ids = batch["expected_base_output"].to(config.device)

            intervenable.set_temperature(temp_scheduler[global_step])

            seq_len = base_input_ids.shape[1]
            abs_pos = token_position if token_position >= 0 else seq_len + token_position

            _, counterfactual_outputs = intervenable(
                {"input_ids": base_input_ids, "attention_mask": base_attention_mask},
                [{"input_ids": source_input_ids, "attention_mask": source_attention_mask}],
                {"sources->base": ([[[abs_pos]]], [[[abs_pos]]])},
            )

            logits = counterfactual_outputs.logits[:, -1, :]
            loss = criterion(logits, expected_token_ids)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            preds = logits.argmax(dim=-1)
            total_correct += (preds == expected_token_ids).sum().item()
            total_count += expected_token_ids.size(0)
            global_step += 1
            
            if global_step % 200 == 0:
                print(f"  Step {global_step}: loss={loss.item():.4f}, IIA={total_correct/total_count:.4f}")

    return intervenable, total_correct / total_count
