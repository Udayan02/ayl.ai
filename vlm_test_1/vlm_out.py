# NOTE: you will have to figure out the paths of the individual frames and their crops.

from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import argparse

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# depends on whether you have 16gb of cuda ram or not
# model.to("cuda:0")

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
    
    # Process inputs
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to("cuda:0", torch.float16)
    
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

# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "What items are present on this shelf? Only tell the items in which you have high confidence. Give me a broad set of very short tags for this shelf, in order of relevance.\n"},
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
#
# # Replace with your local image path
# local_image_path = "image_path.png" # <--- change this line.

# Load the image from the local path
raw_image = Image.open(local_image_path)

inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))

# NOTE: I have not included logic on picking only the top output (topmost tag) given by LLaVA.
