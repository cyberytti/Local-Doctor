from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig


MAX_SEQ_LENGTH = 8000
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
EPOCHS = 2
LEARNING_RATE = 2e-4
SYSTEM_PROMPT = """You are a medical assistant AI trained to identify possible diseases based on given symptoms.

You will be provided with a list of symptoms as input. Your task is to:
1. Predict the most likely disease.
2. Suggest appropriate precautions or basic treatments based on the prediction."""
DATASET_PATH = "/content/combined_disease_dataset.csv"
OUTPUT_DIR = "outputs"
load_in_4bit = False


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None,
    load_in_4bit = load_in_4bit,
    full_finetuning=False
)


dataset = load_dataset("csv", data_files=DATASET_PATH)["train"]


def format_example(example):
    precautions = example["precautions"] if example["precautions"] else "Consult a doctor for proper diagnosis."

    assistant_reply = f"""
POSSIBLE DISEASE: {example["disease"]}

POSSIBLE PRECAUTIONS: {precautions}"""
    return {
        "text": (
            f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
            f"<|im_start|>user\n{example['symptoms']}\n<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_reply}\n<|im_end|>"
        )
    }


dataset = dataset.map(format_example)
dataset = dataset.shuffle(seed=42)
print ("Dataset loaded properly")
print (f"First row of the dataset:\n {dataset[0]["text"]}")


model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 64,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = True,
)


FastLanguageModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",          
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=30,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        remove_unused_columns=False,
    ),
)


trainer_stats = trainer.train()
print("\n✅ Training complete!")
print(f"   Runtime : {trainer_stats.metrics['train_runtime']:.0f}s")
print(f"   Loss    : {trainer_stats.metrics['train_loss']:.4f}")
