import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import defaultdict


class MultiProductDetector:
    def __init__(self, model_path=None):
        """
        Initialize the multi-product detector.

        Args:
            model_path: Path to a pre-trained YOLO model. If None, will use YOLOv8n.
        """
        # Load the YOLO model only if we need it for shelf segmentation
        # We're not using it for product detection - only for segmenting shelf areas
        self.model = YOLO('yolov8n.pt') if not model_path else YOLO(model_path)

        # For feature matching
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()
        self.min_match_count = 10

        # Store multiple reference products
        self.reference_products = []

    def add_reference_product(self, image_path, product_name=None):
        """
        Add a reference product image to the detection list.

        Args:
            image_path: Path to the reference product image.
            product_name: Name of the product (optional).

        Returns:
            product_id: ID of the added product for later reference.
        """
        reference_image = cv2.imread(image_path)
        if reference_image is None:
            raise ValueError(f"Could not read reference image at {image_path}")

        # Set product name if provided, otherwise use filename
        if not product_name:
            product_name = os.path.splitext(os.path.basename(image_path))[0]

        # Convert to grayscale for feature extraction
        gray_reference = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

        # Extract keypoints and descriptors
        keypoints, descriptors = self.sift.detectAndCompute(gray_reference, None)

        # Add to reference products list
        product_id = len(self.reference_products)
        color = self._get_display_color(product_id)

        self.reference_products.append({
            'id': product_id,
            'name': product_name,
            'image': reference_image,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'color': color
        })

        print(f"Added reference product '{product_name}' (ID: {product_id}) with {len(keypoints)} keypoints")
        return product_id

    def _get_display_color(self, product_id):
        """Generate a distinct color for each product based on ID."""
        colors = [
            (0, 0, 255),  # Red
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (0, 128, 255),  # Pink
            (128, 255, 0)  # Lime
        ]
        return colors[product_id % len(colors)]

    def process_video(self, video_path, output_path, confidence_threshold=0.6):
        """
        Process a video to detect multiple reference products on shelves.

        Args:
            video_path: Path to the input video.
            output_path: Path to save the output video.
            confidence_threshold: Minimum confidence for product detection.
        """
        if not self.reference_products:
            raise ValueError("No reference products added. Use add_reference_product first.")

        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video at {video_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            print(f"Processing frame {frame_count}")

            # Process frame
            processed_frame = self.process_frame(frame, confidence_threshold)

            # Write the processed frame
            out.write(processed_frame)

            # Display progress every 10 frames
            if frame_count % 10 == 0:
                cv2.imshow('Processing shelf video...', cv2.resize(processed_frame, (width // 2, height // 2)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Clean up
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Video processing complete. Output saved to {output_path}")

    def process_frame(self, frame, confidence_threshold=0.6):
        """
        Process a single frame to detect all reference products.

        Args:
            frame: Input frame from video.
            confidence_threshold: Minimum confidence for product detection.

        Returns:
            Processed frame with detections drawn.
        """
        if not self.reference_products:
            raise ValueError("No reference products added. Use add_reference_product first.")

        # Create a copy for drawing
        result_frame = frame.copy()

        # Convert frame to grayscale for feature matching
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Step 1: Detect all keypoints in the frame once (more efficient than detecting per ROI)
        frame_keypoints, frame_descriptors = self.sift.detectAndCompute(gray_frame, None)

        if frame_descriptors is None or len(frame_keypoints) < self.min_match_count:
            return result_frame  # No features found in frame

        # Create a dictionary to store detected products
        detected_products = defaultdict(list)

        # Step 2: For each reference product, find matches in the frame
        for product in self.reference_products:
            # Skip if the product has no descriptors
            if product['descriptors'] is None or len(product['descriptors']) < 2:
                continue

            # Match features between reference product and frame
            matches = self.bf.knnMatch(product['descriptors'], frame_descriptors, k=2)

            # Apply ratio test to find good matches
            good_matches = []
            for match_pair in matches:
                if len(match_pair) < 2:
                    continue
                m, n = match_pair
                if m.distance < 0.7 * n.distance:  # Adjust ratio as needed
                    good_matches.append(m)

            # If enough matches are found, find the object
            if len(good_matches) >= self.min_match_count:
                # Extract locations of matched keypoints
                src_pts = np.float32([product['keypoints'][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Find homography
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if M is not None:
                    # Get dimensions of reference image
                    h, w = product['image'].shape[:2]

                    # Define corners of reference image
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

                    # Transform corners to coordinates in the current frame
                    dst = cv2.perspectiveTransform(pts, M)

                    # Calculate match quality (percentage of inliers)
                    match_quality = np.sum(mask) / len(mask) if len(mask) > 0 else 0

                    # Only show high quality matches
                    if match_quality > confidence_threshold:
                        # Store the detection
                        detected_products[product['id']].append({
                            'points': dst,
                            'quality': match_quality,
                            'center': (int(np.mean(dst[:, 0, 0])), int(np.mean(dst[:, 0, 1]))),
                            'product': product
                        })

        # Step 3: Draw the detected products
        for product_id, detections in detected_products.items():
            for detection in detections:
                product = detection['product']
                points = detection['points']
                quality = detection['quality']
                center_x, center_y = detection['center']

                # Draw the bounding box with product-specific color
                result_frame = cv2.polylines(result_frame, [np.int32(points)], True, product['color'], 3)

                # Add background for text
                text_bg_size = cv2.getTextSize(product['name'], cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(result_frame,
                              (center_x - text_bg_size[0] // 2 - 5, center_y - 25),
                              (center_x + text_bg_size[0] // 2 + 5, center_y - 5),
                              product['color'], -1)

                # Add product name
                cv2.putText(result_frame, product['name'],
                            (center_x - text_bg_size[0] // 2, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Show match quality
                cv2.putText(result_frame, f"{quality * 100:.1f}%",
                            (center_x, center_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, product['color'], 2)

        # Add detection summary
        summary_y = 30
        for product in self.reference_products:
            count = len(detected_products[product['id']])
            status = f"{product['name']}: {count} found"
            cv2.putText(result_frame, status, (10, summary_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, product['color'], 2)
            summary_y += 30

        return result_frame

    def analyze_image(self, image_path, output_path=None, confidence_threshold=0.6):
        """
        Process a single image to detect multiple reference products.

        Args:
            image_path: Path to the input image.
            output_path: Path to save the output image. If None, will display the result.
            confidence_threshold: Minimum confidence for product detection.
        """
        if not self.reference_products:
            raise ValueError("No reference products added. Use add_reference_product first.")

        # Read the image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image at {image_path}")

        # Process the frame
        result_frame = self.process_frame(frame, confidence_threshold)

        # Save or display the result
        if output_path:
            cv2.imwrite(output_path, result_frame)
            print(f"Result saved to {output_path}")
            return output_path
        else:
            # Convert to RGB for matplotlib
            result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(12, 8))
            plt.imshow(result_rgb)
            plt.title("Multi-Product Detection Result")
            plt.axis('off')
            plt.tight_layout()
            plt.show()


def main():
    """Example usage of the MultiProductDetector class."""
    import argparse

    parser = argparse.ArgumentParser(description="Detect multiple products on supermarket shelves")
    parser.add_argument("--references", required=True, nargs="+", help="Paths to reference product images")
    parser.add_argument("--product_names", nargs="+", help="Names of the products (optional, same order as references)")
    parser.add_argument("--input", required=True, help="Path to input video or image of shelf")
    parser.add_argument("--output", default=None, help="Path to save output (optional)")
    parser.add_argument("--confidence", type=float, default=0.6, help="Confidence threshold (default: 0.6)")

    args = parser.parse_args()

    # Initialize detector
    detector = MultiProductDetector()

    # Add reference products
    for i, ref_path in enumerate(args.references):
        # Get product name if provided, otherwise None
        product_name = args.product_names[i] if args.product_names and i < len(args.product_names) else None
        detector.add_reference_product(ref_path, product_name)

    # Check if input is video or image
    if args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Process video
        output_path = args.output if args.output else os.path.join(os.getcwd(),
                                                                   "output_" + os.path.basename(args.input))
        detector.process_video(args.input, output_path, args.confidence)
    else:
        # Process single image
        output_path = args.output if args.output else os.path.join(os.getcwd(),
                                                                   "output_" + os.path.basename(args.input))
        detector.analyze_image(args.input, output_path, args.confidence)


if __name__ == "__main__":
    main()
