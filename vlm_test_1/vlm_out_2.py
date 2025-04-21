from PIL import Image
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import argparse
import re

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
# ONE GPU IS ENOUGH BROTHER
model.to("cuda:0")

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

# Define a chat history and use apply_chat_template to get correctly formatted prompt
conversation = [
    {
      "role": "user",
      "content": [
          {"type": "text", "text": "What items are present on this shelf? Only tell the items in which you have high confidence. Give me a small set of very short tags for this shelf, that you have high confidence in. List them as numbered items (1. 2. 3.)\n"},
          {"type": "image"},
        ],
    },
]

# Parse arguments
parser = argparse.ArgumentParser(description="Image to interact with")
parser.add_argument("--image_path", required=True, help="Path to image", type=str)
parser.add_argument("--output", default="vlm_output.txt", type=str, help="Path to output file")
args = parser.parse_args()

# Load the image from the local path
raw_image = Image.open(args.image_path)

# Format the conversation
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# Process inputs
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(model.device)

# Generate response
output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

# Get the full response
full_response = processor.decode(output[0][2:], skip_special_tokens=True)

# Print the full response (for debugging)
print(full_response)

# Extract the top 3 items using regex
# This will look for patterns like "1. Item", "2. Item", "3. Item" 
# Or "1) Item", "2) Item", "3) Item"
# Or bullet points like "• Item", "- Item", "* Item"
numbered_items = re.findall(r'(?:^|\n)(?:\d+[\.\)]\s*|\*\s*|\-\s*|•\s*)(.+?)(?=$|\n)', full_response)

# If we didn't find numbered items, try other formats (like comma-separated)
if not numbered_items:
    # Try to find items after a colon
    colon_match = re.search(r':(.+?)(?=$|\n)', full_response)
    if colon_match:
        # Split by commas if found
        comma_items = colon_match.group(1).split(',')
        numbered_items = [item.strip() for item in comma_items if item.strip()]
    else:
        # Just take the whole text as one item
        numbered_items = [full_response.strip()]

# Take only the top 3 (or fewer if less were found)
top_items = numbered_items[:10]

# Write the top items to the output file
with open(args.output, 'w') as f:
    for i, item in enumerate(top_items, 1):
        f.write(f"{item.strip()}\n")

print(f"Top {len(top_items)} items saved to {args.output}")
