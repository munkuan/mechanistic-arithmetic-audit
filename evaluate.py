import torch

def evaluate_iia(intervenable, test_loader, config, token_position):
    """Evaluates the Interchange Intervention Accuracy on held-out data."""
    total_correct = 0
    total_count = 0
    
    with torch.no_grad():
        for batch in test_loader:
            base_input_ids = batch["base_input_ids"].to(config.device)
            base_attention_mask = batch["base_attention_mask"].to(config.device)
            source_input_ids = batch["source_input_ids"].to(config.device)
            source_attention_mask = batch["source_attention_mask"].to(config.device)
            expected_token_ids = batch["expected_base_output"].to(config.device)

            seq_len = base_input_ids.shape[1]
            abs_pos = token_position if token_position >= 0 else seq_len + token_position

            _, counterfactual_outputs = intervenable(
                {"input_ids": base_input_ids, "attention_mask": base_attention_mask},
                [{"input_ids": source_input_ids, "attention_mask": source_attention_mask}],
                {"sources->base": ([[[abs_pos]]], [[[abs_pos]]])},
            )

            logits = counterfactual_outputs.logits[:, -1, :]
            preds = logits.argmax(dim=-1)

            # Print the predicted number and expected number for the first 5 examples
            # Remove later - debugging
            for i in range(min(5, preds.size(0))):
                predicted_number = intervenable.tokenizer.decode(preds[i].unsqueeze(0), skip_special_tokens=True)
                expected_number = intervenable.tokenizer.decode(expected_token_ids[i].unsqueeze(0), skip_special_tokens=True)
                print(f"Predicted: {predicted_number}, Expected: {expected_number}")
            
            total_correct += (preds == expected_token_ids).sum().item()
            total_count += expected_token_ids.size(0)
            
    return total_correct / total_count