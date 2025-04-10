from PIL import Image

import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import argparse

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# Option 1: Move model to GPU (if you have enough VRAM)
model.to("cuda:1")


# Option 2: Or alternatively keep everything on CPU
# Just remove the .to("cuda:0") from the inputs processing below

def chat_with_llava(image_path, conversation):
    """
    Have a multi-turn conversation with LLaVA about an image

    Args:
        image_path: Path to the image file
        conversation: List of conversation turns in LLaVA format

    Returns:
        Updated conversation with model response
    """
    # Load the image
    raw_image = Image.open(image_path)

    # Format the conversation using the chat template
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Process inputs - keep device consistent with model
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(model.device)

    # Generate response
    output = model.generate(**inputs, max_new_tokens=300, do_sample=False)

    # Decode response
    response = processor.decode(output[0][2:], skip_special_tokens=True)

    # Add the assistant's response to the conversation
    conversation.append({
        "role": "assistant",
        "content": [{"type": "text", "text": response}]
    })

    return conversation, response


# Define a chat history and use ⁠ apply_chat_template ⁠ to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text",
             "text": "What items are present on this shelf? Only tell the items in which you have high confidence. Give me a small set of very short tags for this shelf, that you have high confidence in.\n"},
            {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# Editing to prompt image path when running script in the terminal:
parser = argparse.ArgumentParser(description="Image to interact with")
parser.add_argument("--image_path", required=True, help="Path to image", type=str)
args = parser.parse_args()

local_image_path = args.image_path

# Load the image from the local path
raw_image = Image.open(local_image_path)

# Keep device consistent with model
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(model.device)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))