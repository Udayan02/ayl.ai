#!/bin/bash

# Input parameters
VIDEO_FILE="$1"
OUTPUT_DIR="${2:-orb_slam_data}"
FRAME_RATE="${3:-1}"  # Extract 1 frame per second by default
MAX_FRAMES="${4:-0}"  # Maximum number of frames to extract (0 means no limit)

# Check if required parameters are provided
if [ -z "$VIDEO_FILE" ]; then
    echo "Usage: $0 <video_file> [output_dir] [frame_rate] [max_frames]"
    echo "  video_file: Path to input video file"
    echo "  output_dir: Directory to store extracted frames (default: orb_slam_data)"
    echo "  frame_rate: Frame rate for extraction in fps (default: 1)"
    echo "  max_frames: Maximum number of frames to extract (default: 0, no limit)"
    exit 1
fi

KEYFRAMES_DIR="$OUTPUT_DIR/keyframes"

# Create directories
mkdir -p "$KEYFRAMES_DIR"

# Get video information
VIDEO_FPS=$(ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate "$VIDEO_FILE" | bc -l)
echo "Original video frame rate: $VIDEO_FPS fps"
echo "Extracting frames at: $FRAME_RATE fps"

if [ "$MAX_FRAMES" -gt 0 ]; then
    echo "Maximum number of frames to extract: $MAX_FRAMES"
    # Calculate duration based on max frames and frame rate
    DURATION=$(echo "scale=6; $MAX_FRAMES / $FRAME_RATE" | bc)
    # Extract frames with duration limit
    ffmpeg -i "$VIDEO_FILE" -vf "fps=$FRAME_RATE" -q:v 2 -frames:v "$MAX_FRAMES" "$KEYFRAMES_DIR/frame_%06d.png"
else
    echo "No maximum frame limit set"
    # Extract frames without frame limit
    ffmpeg -i "$VIDEO_FILE" -vf "fps=$FRAME_RATE" -q:v 2 "$KEYFRAMES_DIR/frame_%06d.png"
fi

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
