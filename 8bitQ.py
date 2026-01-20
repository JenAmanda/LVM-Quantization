import torch
import csv
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from tqdm import tqdm
#29193/41370 correct: 70.56562726613488% for 8bit, using bitesandbytes, suspect its not being completely quanitized, try another way
#28939/41370 correct: 69.95165578921923% for eval2

quant_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True
)

# Load model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", 
    quantization_config = quant_config_8bit, 
    device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
print(model.get_memory_footprint()) #2444551504
# Chat prompt template
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "The image contains a name, output the name, no spaces, and no other text. DO NOT output NOM or PRENOM or any codes"},
        ],
    }
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# --- Batch processing parameters ---
BATCH_SIZE = 512  # tune this depending on GPU VRAM

# --- Load dataset filenames ---
image_paths = []
solutions = []
with open("./dataset/written_name_test_v2.csv", "r") as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        if row[0] == "FILENAME":
            continue
        image_paths.append(f"./dataset/test_v2/test/{row[0]}")
        solutions.append(row[1])

total = 0
correct = 0
bar = tqdm(range(0, len(image_paths), BATCH_SIZE))
# --- Process in batches ---
for i in bar:
    batch_paths = image_paths[i:i+BATCH_SIZE]
    batch_solutions = solutions[i:i+BATCH_SIZE]
    images = [Image.open(p).convert("RGB") for p in batch_paths]

    # Prepare batched inputs
    inputs = processor(
        text=[text] * len(images),  # same text for each image
        images=images,
        videos=None,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # Generate outputs for the batch
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64)

    # Remove prompt tokens from outputs
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode outputs
    output_texts = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    
    # Print results
    for (path, output), solution in zip(zip(batch_paths, output_texts), batch_solutions):
        total += 1
        output = output.strip().upper()
        if (output == solution):
            correct += 1
        bar.write(f"{path}: {output}: {solution}")
        per = correct/total * 100
        bar.set_description(f"{correct}/{total} correct: {per}%")
print(f"{correct}/{total} correct: {per}%")
