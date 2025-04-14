#!/usr/bin/env python3
import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# Check CUDA availability
if not torch.cuda.is_available():
    print("CUDA is not available. Please ensure you have a GPU and CUDA installed.")
    exit(1)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current CUDA device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name()}")
print(f"GPU memory available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Set default CUDA device
torch.cuda.set_device(0)
device = torch.device("cuda")

# Read HuggingFace token
with open('hf_token.txt', 'r') as f:
    token = f.read().strip()
os.environ["HF_TOKEN"] = token

def create_translation_pairs():
    """Create English-Saxon translation pairs for training."""
    with open('ds/input.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create instruction format for each text
    formatted_data = []
    
    # Example translations to seed the model
    example_pairs = [
        {
            "english": "Hello, how are you?",
            "saxon": "Nu, wie gieht's dr denn?"
        },
        {
            "english": "The weather is beautiful today",
            "saxon": "'s Watter is heit schie"
        },
        {
            "english": "Good morning everyone",
            "saxon": "Gutn Morgn ihr Leit"
        },
        {
            "english": "I am going to the market",
            "saxon": "Iech gieh ofs'n Markt"
        },
        {
            "english": "My grandmother baked a cake",
            "saxon": "Mei Oma hot an Kuchn gebackn"
        },
        {
            "english": "The children are playing in the garden",
            "saxon": "De Kinner spieln im Gartn"
        },
        {
            "english": "It's snowing outside",
            "saxon": "'s schneit draußn"
        },
        {
            "english": "Would you like some coffee?",
            "saxon": "Willste an Kaffee hobn?"
        },
        {
            "english": "I love my family very much",
            "saxon": "Iech hob mei Familie racht lieb"
        },
        {
            "english": "The sun is shining brightly",
            "saxon": "De Sunn scheint hell"
        },
        {
            "english": "Let's go for a walk in the forest",
            "saxon": "Komm, mir gieh in Wald spazieren"
        },
        {
            "english": "What time is it?",
            "saxon": "Wie spät is'n?"
        },
        {
            "english": "I'm very tired today",
            "saxon": "Iech bie heit racht miede"
        },
        {
            "english": "The food tastes delicious",
            "saxon": "'s Assn schmeckt gut"
        },
        {
            "english": "It's getting cold",
            "saxon": "'s werd kalt"
        },
        {
            "english": "Have a nice weekend",
            "saxon": "Schiens Wochenend"
        },
        {
            "english": "Where are you going?",
            "saxon": "Wu giste hie?"
        },
        {
            "english": "I need to go to work",
            "saxon": "Iech muss of Arbt"
        },
        {
            "english": "The flowers are blooming",
            "saxon": "De Blume blühn"
        },
        {
            "english": "Can you help me please?",
            "saxon": "Kannste mir mol helfn?"
        },
        {
            "english": "I'm coming home late",
            "saxon": "Iech kumm spät ham"
        },
        {
            "english": "The birds are singing",
            "saxon": "De Vöchel singn"
        },
        {
            "english": "My head hurts",
            "saxon": "Mei Kopp tut weh"
        },
        {
            "english": "The water is very cold",
            "saxon": "'s Wasser is racht kalt"
        },
        {
            "english": "I don't understand",
            "saxon": "Iech versteh net"
        }
    ]
    
    # Add example pairs first
    for pair in example_pairs:
        formatted_data.append({
            'text': f"<s>[INST] Translate to Saxon dialect:\n{pair['english']} [/INST] {pair['saxon']}</s>"
        })
    
    # Then add the Saxon texts as examples of the target style
    for item in data['texts']:
        saxon_text = item['text']
        # Add as style example
        formatted_data.append({
            'text': f"<s>[INST] This is an example of Saxon dialect:\n{saxon_text} [/INST] Thank you for showing me this example of Saxon dialect.</s>"
        })
    
    return Dataset.from_list(formatted_data)

# Model configuration
model_name = "mistralai/Mistral-7B-v0.1"
local_model_path = "./mistral-base"

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

try:
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
        token=token,
        torch_dtype=torch.float16
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=token,
        model_max_length=2048,
        padding_side="right"
    )
    
    # Set padding token to EOS token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    print("Model loaded successfully on:", next(model.parameters()).device)
    print("CUDA memory allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")

except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Prepare model for training
model = prepare_model_for_kbit_training(model)

# LoRA configuration for Mistral
lora_config = LoraConfig(
    r=8,  # Reduced rank for memory efficiency
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Create and process dataset
print("Preparing dataset...")
dataset = create_translation_pairs()
print(f"Dataset size: {len(dataset)}")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors="pt"
    )

# Tokenize dataset
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names
)

# Training arguments optimized for Mistral
training_args = TrainingArguments(
    output_dir="./mistral-saxon-translator",
    num_train_epochs=10,
    per_device_train_batch_size=2,  # Reduced batch size
    gradient_accumulation_steps=8,  # Increased gradient accumulation
    warmup_ratio=0.03,
    learning_rate=1e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    remove_unused_columns=False,
    report_to="none"
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

# Train the model
print("Starting training...")
trainer.train()

# Save the model
model.save_pretrained("./mistral-saxon-translator")
tokenizer.save_pretrained("./mistral-saxon-translator")
print("Training completed and model saved!")

def test_translation(text, max_length=512):
    """Test the model's translation capabilities."""
    prompt = f"<s>[INST] Translate the following English text to Saxon dialect:\n{text} [/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        temperature=0.7,
        num_return_sequences=1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()

# Test cases
print("\nTesting the model with example translations:")
test_cases = [
    "Our team consists of more than 180 people, including renowned international researchers as well as highly skilled professionals in administrative and communicative roles. With more than 60 principal investigators, two Humboldt Professorships and up to twelve planned AI Professorships, we support excellence in research and teaching in Leipzig and Dresden. Promoting young talent is also an important part of our work, therefore we have established four Junior Research Groups that meaningfully complement our current research topics. Furthermore, we are welcoming Associated Members who contribute their expertise to our center."
]

print("\nTranslation Results:")
print("=" * 50)
for test_case in test_cases:
    print(f"\nEnglish: {test_case}")
    try:
        translation = test_translation(test_case)
        print(f"Saxon: {translation}")
    except Exception as e:
        print(f"Error during translation: {e}")
    print("-" * 50)

# Interactive testing mode
def interactive_translation_mode():
    print("\nEntering interactive translation mode.")
    print("Type 'quit' or 'exit' to end the session.")
    print("-" * 50)
    
    while True:
        text = input("\nEnter English text to translate: ").strip()
        if text.lower() in ['quit', 'exit']:
            break
        
        if not text:
            continue
            
        try:
            translation = test_translation(text)
            print("\nTranslation:")
            print(f"English: {text}")
            print(f"Saxon: {translation}")
            print("-" * 50)
        except Exception as e:
            print(f"Error during translation: {e}")

# Ask user if they want to enter interactive mode
response = input("\nWould you like to enter interactive translation mode? (yes/no): ").strip().lower()
if response in ['y', 'yes']:
    interactive_translation_mode()
print("\nTesting completed.") 