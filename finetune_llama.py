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
    
    formatted_data = []
    
    # Example translations including formal/complex examples
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
        },
        {
            "english": "Our research team is conducting important studies",
            "saxon": "Unner Forschergrupp macht wichtige Unnersuchungen"
        },
        {
            "english": "The university's main building is located in the city center",
            "saxon": "'s Hauptgebäude vun dr Uni stieht mittich in dr Stadt"
        },
        {
            "english": "The conference will be held next month with international participants",
            "saxon": "De Konferenz werd nächstn Monat mit Leit aus aller Welt abgehaltn"
        },
        {
            "english": "We are proud to announce our latest research findings",
            "saxon": "Mir sein stolz, dass mr eich unner neistn Forschungsergebnisse zoign kenne"
        },
        {
            "english": "The collaboration between different departments has been successful",
            "saxon": "De Zusammenarbt zwischn de verschiedn Abteilungen is gut geloffn"
        }
    ]
    
    # Add example pairs with consistent translation prompt
    for pair in example_pairs:
        formatted_data.append({
            'text': f"<s>[INST] You are a translator for the Saxon German dialect. Translate this English text to Saxon. Keep the Saxon style informal and authentic:\n\n{pair['english']}\n\nSaxon translation: [/INST] {pair['saxon']}</s>",
            'english': pair['english'],
            'saxon': pair['saxon']
        })
    
    # Extract patterns from Saxon texts to create more training examples
    for item in data['texts']:
        saxon_text = item['text']
        formatted_data.append({
            'text': f"<s>[INST] You are a translator for the Saxon German dialect. Translate this English text to Saxon. Keep the Saxon style informal and authentic:\n\nThe text must be translated to authentic Saxon dialect, not standard German. Here's the text:\n\n{saxon_text}\n\nSaxon translation: [/INST] {saxon_text}</s>",
            'english': "Text sample",  # Placeholder
            'saxon': saxon_text
        })
    
    # Create dataset and split
    full_dataset = Dataset.from_list(formatted_data)
    
    # Split dataset into train, validation, and test
    splits = full_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    train_test = splits['train'].train_test_split(test_size=0.15, shuffle=True, seed=42)
    
    train_dataset = train_test['train']
    val_dataset = train_test['test']
    test_dataset = splits['test']
    
    print(f"Total examples: {len(full_dataset)}")
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    print(f"Test examples: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset

# Model configuration
model_name = "mistralai/Mistral-7B-v0.1"
local_model_path = "./mistral-base"

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    yesad_in_4bit=True,
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

# Create and process datasets
print("Preparing datasets...")
train_dataset, val_dataset, test_dataset = create_translation_pairs()

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors="pt"
    )

# Tokenize all datasets
train_tokenized = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

val_tokenized = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=val_dataset.column_names
)

test_tokenized = test_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=test_dataset.column_names
)

# Training arguments optimized for Mistral
training_args = TrainingArguments(
    output_dir="./mistral-saxon-translator",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    warmup_ratio=0.05,
    learning_rate=5e-5,
    fp16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    eval_strategy="steps",
    eval_steps=50,
    save_total_limit=3,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    remove_unused_columns=False,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    push_to_hub=False
)

# Create trainer with evaluation dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    compute_metrics=lambda eval_pred: {
        "loss": eval_pred.predictions.mean()
    }
)

# Train the model
print("Starting training...")
trainer.train()

# Evaluate on test set
print("\nEvaluating on test set...")
test_results = trainer.evaluate(test_tokenized)
print(f"Test results: {test_results}")

# Save the best model
model.save_pretrained("./mistral-saxon-translator")
tokenizer.save_pretrained("./mistral-saxon-translator")
print("Training completed and model saved!")

def test_translation(text, max_length=512):
    """Test the model's translation capabilities."""
    prompt = f"<s>[INST] You are a translator for the Saxon German dialect. Translate this English text to Saxon. Keep the Saxon style informal and authentic. Do not use standard German, use proper Saxon dialect:\n\n{text}\n\nSaxon translation: [/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        temperature=0.8,  # Slightly increased for more dialectal variation
        top_p=0.9,
        num_return_sequences=1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2
    )
    
    # Extract only the translation part
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Saxon translation:" in translation:
        translation = translation.split("Saxon translation:")[-1].strip()
    return translation

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