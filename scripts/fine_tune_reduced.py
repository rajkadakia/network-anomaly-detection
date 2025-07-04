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

def reduce_dataset_size(input_path, output_path, target_size=50000, min_samples=100):
    """Reduce dataset size with stratified sampling and minimum samples per class"""
    print("\n🔻 Reducing dataset size...")
    
    # Get total rows and class distribution
    with tqdm(desc="📊 Counting rows", unit="rows") as pbar:
        total_rows = sum(1 for _ in open(input_path)) - 1
        pbar.total = total_rows
        pbar.refresh()

    # Read in chunks and sample proportionally
    chunk_size = 10000
    reduced_df = pd.DataFrame()
    
    with tqdm(total=target_size, desc="🎯 Sampling data") as pbar:
        for chunk in pd.read_csv(input_path, chunksize=chunk_size):
            # Stratified sampling per chunk
            sample = chunk.groupby('label', group_keys=False).apply(
                lambda x: x.sample(frac=target_size/total_rows, random_state=42)
            )
            reduced_df = pd.concat([reduced_df, sample])
            
            if len(reduced_df) >= target_size:
                break
            pbar.update(len(sample))

    # Enforce minimum samples per class
    print("\n⚖️  Balancing classes...")
    class_counts = reduced_df['label'].value_counts()
    with tqdm(total=len(class_counts), desc="🔁 Oversampling rare classes") as pbar:
        for cls, count in class_counts.items():
            if count < min_samples:
                additional = reduced_df[reduced_df['label'] == cls].sample(
                    min_samples - count, replace=True, random_state=42
                )
                reduced_df = pd.concat([reduced_df, additional])
            pbar.update(1)

    reduced_df.to_csv(output_path, index=False)
    print(f"\n✅ Reduced dataset saved to {output_path} (n={len(reduced_df)})")
    return reduced_df

def prepare_dataset(data_path, test_size=0.2):
    """Convert network features to text prompts with progress tracking"""
    print("\n🚀 Preparing dataset...")
    
    # Read reduced dataset
    with tqdm(desc="📥 Loading reduced data") as pbar:
        df = pd.read_csv(data_path)
        pbar.update(1)

    # Convert features to text prompts
    text_prompts = []
    with tqdm(total=len(df), desc="📝 Generating prompts") as pbar:
        for _, row in df.drop(columns=['label']).iterrows():
            features = ", ".join([f"{col}: {val}" for col, val in row.items()])
            text_prompts.append(f"Analyze network logs: {features}")
            pbar.update(1)

    # Convert labels
    labels = df['label'].astype(str).tolist()

    # Stratified split
    print("\n🔀 Splitting dataset...")
    return train_test_split(
        text_prompts, labels, 
        test_size=test_size, 
        random_state=42,
        stratify=labels
    )

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
    # Reduce dataset size first
    reduced_path = "../datasets/reduced_data.csv"
    if not os.path.exists(reduced_path):
        reduce_dataset_size("../datasets/historical_data.csv", reduced_path, target_size=50000, min_samples=100)
    
    # Prepare data
    train_texts, test_texts, train_labels, test_labels = prepare_dataset(reduced_path)
    
    # Fine-tune model
    model, tokenizer, device = fine_tune_model(train_texts, train_labels, test_texts, test_labels)
    
    # Validate model
    predictions = validate_model(model, tokenizer, device, test_texts, test_labels)
    
    # Save model
    print("\n💾 Saving model...")
    model.save_pretrained("../models/network_anomaly_detector")
    tokenizer.save_pretrained("../models/network_anomaly_detector")
    print("🎉 Training complete! Model saved to models/network_anomaly_detector/")

if __name__ == "__main__":
    main()