# imports
import os
import sys
import argparse

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# model definition
model = AutoModelForZeroShotObjectDetection.from_pretrained("google/owlvit-base-patch32")
processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")


# functions
def detect_shelf_items(image_path, candidate_labels, threshold):
    # Open the image
    image = Image.open(image_path)

    # Prepare inputs
    inputs = processor(
        text=candidate_labels,
        images=image,
        return_tensors="pt"
    )

    # Perform object detection
    outputs = model(**inputs)

    # Process results
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=target_sizes
    )[0]

    # return results
    # Prepare results
    detected_items = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        detected_items.append({
            "label": candidate_labels[label],
            "score": round(score.item(), 3),
            "box": box
        })

    return detected_items


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (list or tuple): [x_min, y_min, x_max, y_max] of the first box.
        box2 (list or tuple): [x_min, y_min, x_max, y_max] of the second box.

    Returns:
        float: The IoU value (between 0 and 1).
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate intersection coordinates
    x_intersect_min = max(x1_min, x2_min)
    y_intersect_min = max(y1_min, y2_min)
    x_intersect_max = min(x1_max, x2_max)
    y_intersect_max = min(y1_max, y2_max)

    # Calculate intersection area
    intersection_area = max(0, x_intersect_max - x_intersect_min) * max(0, y_intersect_max - y_intersect_min)

    # Calculate area of each box
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_max)

    # Calculate union area
    union_area = area1 + area2 - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


def is_contained(inner_box, outer_box, containment_threshold=0.9):
    """
    Checks if the inner bounding box is highly contained within the outer bounding box.

    Args:
        inner_box (list or tuple): [x_min, y_min, x_max, y_max] of the inner box.
        outer_box (list or tuple): [x_min, y_min, x_max, y_max] of the outer box.
        containment_threshold (float): Minimum IoU for considering containment.

    Returns:
        bool: True if the inner box is highly contained within the outer box, False otherwise.
    """
    iou = calculate_iou(inner_box, outer_box)

    # Calculate the area of the inner box
    inner_area = (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1])

    # Calculate the area of the intersection
    x_intersect_min = max(inner_box[0], outer_box[0])
    y_intersect_min = max(inner_box[1], outer_box[1])
    x_intersect_max = min(inner_box[2], outer_box[2])
    y_intersect_max = min(inner_box[3], outer_box[3])
    intersection_area = max(0, x_intersect_max - x_intersect_min) * max(0, y_intersect_max - y_intersect_min)

    # Check if the intersection area is a large fraction of the inner box's area
    if inner_area > 0 and intersection_area / inner_area >= containment_threshold:
        return True
    return False


def remove_overlapping_and_contained(detections, iou_threshold=0.5, containment_threshold=0.9):
    """
    Removes bounding boxes with high IoU or those highly contained within
    a higher-confidence box of a potentially different class.

    Args:
        detections (list of lists/tuples): List of [x_min, y_min, x_max, y_max, confidence, class_label (optional)].
        iou_threshold (float): The IoU threshold above which boxes are considered highly overlapping.
        containment_threshold (float): Minimum IoU for considering containment.

    Returns:
        list: A new list of bounding boxes (with confidence and optional class) with overlaps and containments removed.
    """
    if not detections:
        return []

    # Sort detections by confidence in descending order
    sorted_detections = sorted(detections, key=lambda x: x[4], reverse=True)
    filtered_detections = []
    processed_indices = set()

    for i in range(len(sorted_detections)):
        if i in processed_indices:
            continue

        best_detection = sorted_detections[i]
        filtered_detections.append(best_detection)
        processed_indices.add(i)

        for j in range(i + 1, len(sorted_detections)):
            if j in processed_indices:
                continue

            other_detection = sorted_detections[j]
            iou = calculate_iou(best_detection[:4], other_detection[:4])

            # Remove if high IoU with a higher confidence box
            if iou > iou_threshold:
                processed_indices.add(j)
            # Also remove if highly contained within a higher confidence box
            elif is_contained(other_detection[:4], best_detection[:4], containment_threshold):
                processed_indices.add(j)

    return filtered_detections


# Replace the IPython import with a more direct approach
# from IPython import get_ipython  # Remove this line

def plot_bounding_boxes(image_path, detections):
    try:
        img = Image.open(image_path)
        fig, ax = plt.subplots(1, figsize=(10, 8))
        ax.imshow(img)
        cmap = cm.get_cmap('tab20', len(detections))

        for i, detection in enumerate(detections):
            x_min, y_min, x_max, y_max, confidence = detection[:5]
            width = x_max - x_min
            height = y_max - y_min
            color = cmap(i)
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            text = f"{confidence:.2f}%"
            ax.text(x_min, y_min - 5, text, fontsize=8, color='white',
                    bbox=dict(facecolor=color, edgecolor=color, alpha=0.7))

        ax.axis('off')
        plt.tight_layout()

        # Always save the visualization
        output_dir = 'visualizations'
        os.makedirs(output_dir, exist_ok=True)

        basename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(basename)[0]
        output_path = f"{output_dir}/{name_without_ext}_detection.png"

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to: {output_path}")

    except FileNotFoundError as e:
        print(f"Error: Image not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


parser = argparse.ArgumentParser(description="Image to interact with")
parser.add_argument("--img_path", type=str, required=True, help="Path to the image")
args = parser.parse_args()
# image_path = "craft_store_frames/frame_0395.png"
image_path = args.img_path

candidate_labels = ["shelf"]

# Detect items
results = detect_shelf_items(image_path, candidate_labels, 0.04)

# Print results
print("Detected Items:")
for item in results:
    print(f"- {item['label']}: Confidence {item['score'] * 100:.2f}%, Box: {item['box']}")

# Example usage
# Replace with your actual image path and coordinates
box_coordinates_list = [r['box'] + [r['score'] * 100] for r in results]
# [x_min, y_min, x_max, y_max]

# Call the function
filtered_detections = remove_overlapping_and_contained(box_coordinates_list, iou_threshold=0.5)
print("Filtered Detections:", filtered_detections)
plot_bounding_boxes(image_path, filtered_detections)


# below code is to crop the bounding boxes and save them in separate images
def crop_and_save(image_path, bbox_coordinates, output_path="cropped_image.png"):
    """
    Crops a specific region from an image based on bounding box coordinates
    and saves it as a new image.

    Args:
        image_path (str): Path to the original image file.
        bbox_coordinates (tuple or list): A tuple or list of (x_min, y_min, x_max, y_max)
                                         defining the cropping region.
        output_path (str): Path where the cropped image will be saved.
                           Defaults to "cropped_image.png" in the current directory.
    """
    try:
        img = Image.open(image_path)
        cropped_img = img.crop(bbox_coordinates)
        cropped_img.save(output_path)
        print(f"Cropped image saved to: {output_path}")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


# may change as required
os.makedirs('temp_crops', exist_ok=True)

original_image = image_path  # Replace with the path to your image
print(original_image)
filename = os.path.basename(image_path)
filename_without_ext = os.path.splitext(filename)[0]
for i, fd in enumerate(filtered_detections, 1):
    bounding_box = fd[0:-1]
    cropped_output = f"temp_crops/{filename_without_ext}crop{i}.png"

    if os.path.exists(original_image):
        crop_and_save(original_image, bounding_box, cropped_output)
    else:
        print(f"Error: The example image '{original_image}' was not found. Please replace it with your image path.")