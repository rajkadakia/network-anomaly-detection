import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import f1_score
import torch
import os
os.environ['HF_HOME'] = './hf_cache'
os.environ['HF_HUB_OFFLINE'] = '1'  # Enforce offline mode globally
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def prepare_dataset(data_path, test_size=0.2):
    """Convert network features to text prompts with full progress tracking"""
    with tqdm(total=4, desc="🚀 Dataset Pipeline") as main_pbar:
        main_pbar.set_postfix_str("Loading CSV")
        df = pd.read_csv(data_path)
        main_pbar.update(1)
        
        text_prompts = []
        main_pbar.set_postfix_str("Processing logs")
        with tqdm(total=len(df), desc="📊 Network Logs", leave=False) as log_pbar:
            for _, row in df.drop(columns=['label']).iterrows():
                features = ", ".join([f"{col}: {val}" for col, val in row.items()])
                text_prompts.append(f"Analyze network logs: {features}")
                log_pbar.update(1)
        main_pbar.update(1)
        
        main_pbar.set_postfix_str("Encoding labels")
        labels = df['label'].astype(str).tolist()
        main_pbar.update(1)
        
        main_pbar.set_postfix_str("Splitting data")
        splits = train_test_split(
            text_prompts, labels, 
            test_size=test_size, 
            random_state=42
        )
        main_pbar.update(1)
        
    return splits

def fine_tune_model(train_texts, train_labels, eval_texts, eval_labels, model_name="../models/flan-t5-base"):
    """Fine-tune FLAN-T5 with LoRA using local model files"""
    # Device configuration
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    with tqdm(total=6, desc="🎯 Model Pipeline") as main_pbar:
        main_pbar.set_postfix_str("Loading model")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=True
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                local_files_only=True
            )
            # Force embeddings to CPU
            model.base_model.shared = model.base_model.shared.to('cpu')
            model.base_model.encoder.embed_tokens = model.base_model.encoder.embed_tokens.to('cpu')
            model.base_model.decoder.embed_tokens = model.base_model.decoder.embed_tokens.to('cpu')
            
            # Rest of model stays on MPS
            model = model.to(device)
            print(f"Model loaded successfully from {model_name}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        main_pbar.update(1)

        main_pbar.set_postfix_str("Configuring LoRA")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q", "v"]
        )
        model = get_peft_model(model, peft_config)
        main_pbar.update(1)

        main_pbar.set_postfix_str("Tokenizing data")
        with tqdm(total=2, desc="🔡 Tokenization", leave=False) as token_pbar:
            train_encodings = tokenizer(
                train_texts, 
                truncation=True, 
                padding="max_length", 
                max_length=512,
                add_special_tokens=True
            )
            token_pbar.update(1)
            
            train_label_encodings = tokenizer(
                train_labels, 
                truncation=True, 
                padding="max_length", 
                max_length=4
            )
            token_pbar.update(1)
        main_pbar.update(1)

        main_pbar.set_postfix_str("Tokenizing eval data")
        with tqdm(total=2, desc="🔡 Eval Tokenization", leave=False) as eval_token_pbar:
            eval_encodings = tokenizer(
                eval_texts, 
                truncation=True, 
                padding="max_length", 
                max_length=512,
                add_special_tokens=True
            )
            eval_token_pbar.update(1)
            
            eval_label_encodings = tokenizer(
                eval_labels, 
                truncation=True, 
                padding="max_length", 
                max_length=4
            )
            eval_token_pbar.update(1)
        main_pbar.update(1)

        main_pbar.set_postfix_str("Creating datasets")
        class NetworkDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels.input_ids
                
            def __getitem__(self, idx):
                return {
                    'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
                    'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
                    'labels': torch.tensor(self.labels[idx])
                }
            
            def __len__(self):
                return len(self.labels)
        
        train_dataset = NetworkDataset(train_encodings, train_label_encodings)
        eval_dataset = NetworkDataset(eval_encodings, eval_label_encodings)
        main_pbar.update(1)

        main_pbar.set_postfix_str("Configuring trainer")
        training_args = Seq2SeqTrainingArguments(
            output_dir="../results",
            eval_strategy="epoch",
            learning_rate=3e-4,
            per_device_train_batch_size=2,  # Reduced for M2 GPU memory
            per_device_eval_batch_size=2,
            num_train_epochs=5,
            weight_decay=0.01,
            save_total_limit=3,
            predict_with_generate=True,
            fp16=False,  # Disable FP16 for MPS
            bf16=False, # Disable BF16 for MPS
            dataloader_pin_memory=False,  # Avoid MPS pinning warnings
            report_to="none",
            logging_steps=50,
            disable_tqdm=False
        )
        
        data_collator = DataCollatorForSeq2Seq(tokenizer)
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,  # Pass the evaluation dataset
            data_collator=data_collator,
        )
        main_pbar.update(1)

    print("\n🔥 Training Progress:")
    model.config.use_cache = False
    trainer.train()
    return model, tokenizer, device

def validate_model(model, tokenizer, device, test_texts, test_labels):
    """Validate model with hierarchical progress tracking"""
    original_device = next(model.parameters()).device
    model.to('cpu')  # Validate on CPU
    with tqdm(total=2, desc="🧪 Validation Pipeline") as main_pbar:
        main_pbar.set_postfix_str("Initializing metrics")
        main_pbar.update(1)
        
        main_pbar.set_postfix_str("Generating predictions")
        predictions = []
        with tqdm(total=len(test_texts), desc="🔮 Predictions", leave=False) as pred_pbar:
            for text in test_texts:
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                outputs = model.generate(**inputs)
                predictions.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
                pred_pbar.update(1)
        main_pbar.update(1)
        model.to(original_device)  # Restore original device

    macro_f1 = f1_score(
        y_true=test_labels, 
        y_pred=predictions, 
        average="macro"
    )
    print(f"\n✅ Validation Complete | Macro F1: {macro_f1:.4f}")
    return predictions

def main():
    # Step 1: Prepare data
    train_texts, test_texts, train_labels, test_labels = prepare_dataset("../datasets/clean_data.csv")
    
    # Step 2: Fine-tune model (with LoRA)
    model, tokenizer, device = fine_tune_model(train_texts, train_labels, test_texts, test_labels)
    
    # Step 3: Validate model
    predictions = validate_model(model, tokenizer, device, test_texts, test_labels)
    
    # Step 4: Save LoRA-adapted model
    print("\n💾 Saving LoRA model...")
    model.save_pretrained("../models/network_anomaly_detector")         # LoRA adapters
    tokenizer.save_pretrained("../models/network_anomaly_detector")     # Tokenizer used

    # Step 5: Merge LoRA with base model and save as a standalone Hugging Face model
    print("🔁 Merging LoRA into base model...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained("../models/merged_model")
    tokenizer.save_pretrained("../models/merged_model")  # Reuse same tokenizer

    print("🎉 Training complete!")
    print("📦 LoRA model saved to: models/network_anomaly_detector/")
    print("📦 Merged full model saved to: models/merged_model/")

if __name__ == "__main__":
    main()
