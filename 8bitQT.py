import torch
import csv
import os
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from quant_int8_triton import replace_linear_with_triton
import Levenshtein


model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    dtype=torch.float16,
    device_map="cpu",  # load on CPU first
)

print("Model size:", model.get_memory_footprint() / (1024**2), "MB")

# quantize all safe linear layers to int8 weights and move to GPU
# 
# replace_linear_with_triton(model, prefix="", exclude_prefixes=["placeHolder"])

replace_linear_with_triton(model, prefix="", exclude_prefixes=["lm_head", "model.embed_tokens", "model.visual"])
model.to("cuda")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
after = model.get_memory_footprint() #get size after quantization


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
BATCH_SIZE = 16  # smaller batch for testing

# --- Load dataset filenames ---
image_paths = []
solutions = []
with open("./dataset/written_name_test_v2.csv", "r") as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        if row[0] == "FILENAME":
            continue
        img_path = f"./dataset/test_v2/testMini/{row[0]}"
        # img_path = f"./dataset/test_v2/test/{row[0]}"
        if os.path.exists(img_path):
            image_paths.append(img_path)
            solutions.append(row[1])
        # else:
            # print(f"Skipping missing file: {img_path}")

total = 0
correct = 0
lratio = 0
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
        if output == solution:
            correct += 1
        bar.write(f"{path}: {output} | Expected: {solution}")
        per = correct / total * 100
        bar.set_description(f"{correct}/{total} correct: {per:.2f}%")
        check_sim = Levenshtein.distance(output, solution)
        lratio += check_sim
        print(f"The distance between {output} and {solution}: {check_sim}")
print(f"{correct}/{total} correct: {per}%")
print(f"The l dist for the set is ",(lratio/total),"")
print("Model size:",after / (1024**2),"MB")
