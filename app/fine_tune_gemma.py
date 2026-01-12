"""
Offline fine-tuning example (Gemma model)
"""
from transformers import AutoTokenizer,AutoModelForCausalLM,Trainer,TrainingArguments,TextDataset,DataCollatorForLanguageModeling

MODEL_NAME="google/gemma-2b"
tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)
model=AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def load_dataset(path,tokenizer,block_size=128):
    return TextDataset(tokenizer=tokenizer,file_path=path,block_size=block_size)

train_ds=load_dataset("train_data.txt",tokenizer)
collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)

args=TrainingArguments(output_dir="./gemma_finetuned",overwrite_output_dir=True,
                       num_train_epochs=1,per_device_train_batch_size=2,
                       save_steps=500,save_total_limit=2)

trainer=Trainer(model=model,args=args,data_collator=collator,train_dataset=train_ds)
trainer.train()
model.save_pretrained("./gemma_finetuned"); tokenizer.save_pretrained("./gemma_finetuned")
