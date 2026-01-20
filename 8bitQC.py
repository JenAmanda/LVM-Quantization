import torch
import csv
import os
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from tqdm import tqdm
from quant_int8 import quantize_model_int8  

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,   # keep fp16 for activations
    device_map="cpu"
)
#4213 MB
for name, module in model.named_modules():
    print(name)

# print("Model size:", model.get_memory_footprint() / (1024**2), "MB")
# for name, module in model.named_modules():
#     if isinstance(module, nn.Linear):
#         print("FOUND FP16 LAYER before:", name)

model = quantize_model_int8(model,None)  # tune group_size {64,128,256}
model.to("cuda")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
after = model.get_memory_footprint()
# print("==============================Size:", model.get_memory_footprint())
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        print("FOUND FP16 LAYER after:", name)

# exit(0)
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
