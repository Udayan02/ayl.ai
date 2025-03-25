#!/bin/bash

# Input parameters
VIDEO_FILE="$1"
OUTPUT_DIR="${2:-orb_slam_data}"
KEYFRAMES_DIR="$OUTPUT_DIR/keyframes"
FRAME_RATE="${3:-1}"  # Extract 1 frame per second by default

# Create directories
mkdir -p "$KEYFRAMES_DIR"

# Get video information
VIDEO_FPS=$(ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate "$VIDEO_FILE" | bc -l)
echo "Original video frame rate: $VIDEO_FPS fps"
echo "Extracting frames at: $FRAME_RATE fps"

# Extract frames at specified frame rate
ffmpeg -i "$VIDEO_FILE" -vf "fps=$FRAME_RATE" -q:v 2 "$KEYFRAMES_DIR/frame_%06d.png"

# Generate rgb.txt file
RGB_FILE="$OUTPUT_DIR/rgb.txt"
> "$RGB_FILE" # Create or clear the file

# List all extracted frames in order
FRAMES=$(find "$KEYFRAMES_DIR" -name "frame_*.png" | sort)

# For each frame, calculate the timestamp
FRAME_COUNT=0
for FRAME in $FRAMES; do
    FRAME_NAME=$(basename "$FRAME")
    FRAME_COUNT=$((FRAME_COUNT + 1))
    
    # Calculate timestamp (seconds since start)
    TIMESTAMP=$(echo "scale=6; ($FRAME_COUNT - 1) / $FRAME_RATE" | bc)
    
    # Write to rgb.txt
    echo "$TIMESTAMP keyframes/$FRAME_NAME" >> "$RGB_FILE"
    
    echo "Processed frame $FRAME_NAME with timestamp $TIMESTAMP"
done

echo "Completed! Extracted $FRAME_COUNT frames at $FRAME_RATE fps"
echo "Output in $OUTPUT_DIR"
