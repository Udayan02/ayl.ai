#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
from datetime import datetime

class RealSenseRecorder:
    def __init__(self, output_folder="recordings"):
        # Create output folder if it doesn't exist
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Generate timestamp for this recording session
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording = False
        self.frame_count = 0
        
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable streams
        self.config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
        
        # Create align object to align depth frames to color frames
        self.align = rs.align(rs.stream.color)
        
        # Video writers - will be initialized when recording starts
        self.color_video_writer = None
        self.depth_colormap_video_writer = None
        self.fps = 30
        
    def start(self):
        """Start the RealSense pipeline"""
        print("Starting RealSense camera...")
        self.profile = self.pipeline.start(self.config)
        
        # Get depth scale for distance calculations
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        print(f"Depth scale: {self.depth_scale}")
        
        # Wait for camera to stabilize
        for _ in range(30):
            self.pipeline.wait_for_frames()
        
        print("Camera ready!")
        
    def start_recording(self):
        """Start recording data"""
        if not self.recording:
            # Create a unique folder for this recording
            self.session_folder = os.path.join(self.output_folder, self.timestamp)
            os.makedirs(self.session_folder, exist_ok=True)
            
            # Create folders for different types of data
            self.color_folder = os.path.join(self.session_folder, "color")
            self.depth_folder = os.path.join(self.session_folder, "depth")
            self.depth_colormap_folder = os.path.join(self.session_folder, "depth_colormap")
            
            os.makedirs(self.color_folder, exist_ok=True)
            os.makedirs(self.depth_folder, exist_ok=True)
            os.makedirs(self.depth_colormap_folder, exist_ok=True)
            
            # Create metadata file to store depth values over time
            self.metadata_file = open(os.path.join(self.session_folder, "metadata.csv"), "w")
            self.metadata_file.write("frame,timestamp,avg_depth,min_depth,max_depth\n")
            
            # Initialize video writers
            color_video_path = os.path.join(self.session_folder, "color_video.mp4")
            depth_video_path = os.path.join(self.session_folder, "depth_video.mp4")
            
            # Define codec (H.264)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # Create video writers for color and depth
            self.color_video_writer = cv2.VideoWriter(
                color_video_path, fourcc, self.fps, (1920, 1080))
            self.depth_colormap_video_writer = cv2.VideoWriter(
                depth_video_path, fourcc, self.fps, (1024, 768))
            
            # Reset frame counter
            self.frame_count = 0
            self.recording = True
            print(f"Recording started. Saving to {self.session_folder}")
            
    def stop_recording(self):
        """Stop recording data"""
        if self.recording:
            self.recording = False
            self.metadata_file.close()
            
            # Release video writers
            if self.color_video_writer is not None:
                self.color_video_writer.release()
            if self.depth_colormap_video_writer is not None:
                self.depth_colormap_video_writer.release()
                
            print(f"Recording stopped. Saved {self.frame_count} frames and video files.")
            
    def process_frames(self):
        """Process the latest frames from the camera"""
        # Wait for frames
        frames = self.pipeline.wait_for_frames()
        
        # Align depth frame to color frame
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None, None
            
        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Create colormap from depth image for visualization
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        if self.recording:
            # Save frames as images
            frame_filename = f"frame_{self.frame_count:06d}"
            cv2.imwrite(os.path.join(self.color_folder, f"{frame_filename}.jpg"), color_image)
            
            # Save depth as 16-bit PNG (preserves actual depth values)
            cv2.imwrite(os.path.join(self.depth_folder, f"{frame_filename}.png"), depth_image)
            
            # Save colorized depth for visualization
            cv2.imwrite(os.path.join(self.depth_colormap_folder, f"{frame_filename}.jpg"), depth_colormap)
            
            # Write frames to video files
            self.color_video_writer.write(color_image)
            
            # Ensure depth colormap is the correct size for the video writer
            if depth_colormap.shape[:2] != (768, 1024):
                resized_depth_colormap = cv2.resize(depth_colormap, (1024, 768))
                self.depth_colormap_video_writer.write(resized_depth_colormap)
            else:
                self.depth_colormap_video_writer.write(depth_colormap)
            
            # Calculate depth statistics (in meters)
            valid_depth = depth_image[depth_image != 0] * self.depth_scale
            if valid_depth.size > 0:
                avg_depth = np.mean(valid_depth)
                min_depth = np.min(valid_depth)
                max_depth = np.max(valid_depth)
            else:
                avg_depth = min_depth = max_depth = 0
                
            # Write metadata
            timestamp = time.time()
            self.metadata_file.write(f"{self.frame_count},{timestamp},{avg_depth},{min_depth},{max_depth}\n")
            self.metadata_file.flush()  # Ensure data is written immediately
            
            self.frame_count += 1
            
        return color_image, depth_image, depth_colormap
        
    def stop(self):
        """Stop the RealSense pipeline"""
        if self.recording:
            self.stop_recording()
        self.pipeline.stop()
        
        # Make sure all OpenCV windows and resources are properly released
        if self.color_video_writer is not None and self.color_video_writer.isOpened():
            self.color_video_writer.release()
        if self.depth_colormap_video_writer is not None and self.depth_colormap_video_writer.isOpened():
            self.depth_colormap_video_writer.release()
            
        print("Camera stopped.")

def main():
    # Create recorder instance
    recorder = RealSenseRecorder()
    
    try:
        # Start camera
        recorder.start()
        
        # Create window for display
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        
        print("Press 'r' to start/stop recording, 'q' to quit")
        
        while True:
            # Process frames
            color_image, depth_image, depth_colormap = recorder.process_frames()
            
            if color_image is None:
                continue
                
            # Create combined display image
            if depth_colormap.shape[0] != color_image.shape[0]:
                # Resize depth colormap to match color image dimensions
                depth_colormap = cv2.resize(depth_colormap, 
                                           (color_image.shape[1], color_image.shape[0]))
                                
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            
            # Add recording indicator
            if recorder.recording:
                # Add red circle to indicate recording
                cv2.circle(images, (30, 30), 15, (0, 0, 255), -1)
                
                # Add frame counter
                cv2.putText(images, f"Frame: {recorder.frame_count}", 
                            (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Show images
            cv2.imshow('RealSense', images)
            
            # Check for keyboard input
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:  # q or ESC
                break
            elif key & 0xFF == ord('r'):
                if recorder.recording:
                    recorder.stop_recording()
                else:
                    recorder.start_recording()
                    
    finally:
        # Stop camera and close windows
        recorder.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
