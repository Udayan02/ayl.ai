#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
from datetime import datetime

class L515OrbSlamRecorder:
    def __init__(self, output_folder="rgbd_dataset"):
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
        
        # Enable streams - using lower resolution for better compatibility
        self.config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)
        
        # Create align object to align depth frames to color frames
        self.align = rs.align(rs.stream.color)
        
    def start(self):
        """Start the RealSense pipeline"""
        print("Starting RealSense L515 camera...")
        try:
            self.profile = self.pipeline.start(self.config)
            
            # Get depth scale for distance calculations
            self.depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = self.depth_sensor.get_depth_scale()
            print(f"Depth scale: {self.depth_scale}")
            
            # Get camera intrinsics
            self.get_camera_intrinsics()
            
            # Wait for camera to stabilize
            for _ in range(30):
                self.pipeline.wait_for_frames()
            
            print("Camera ready!")
            return True
        except RuntimeError as e:
            print(f"Error starting camera: {str(e)}")
            return False
            
    def get_camera_intrinsics(self):
        """Get camera intrinsics for ORB_SLAM3 configuration"""
        color_stream = self.profile.get_stream(rs.stream.color)
        depth_stream = self.profile.get_stream(rs.stream.depth)
        
        self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        
        # Save intrinsics to file for later use with ORB_SLAM3
        with open(os.path.join(self.output_folder, "camera_intrinsics.txt"), "w") as f:
            f.write("# L515 Camera Parameters for ORB_SLAM3\n\n")
            f.write("# RGB Camera Parameters (Pinhole model)\n")
            f.write(f"Camera.fx: {self.color_intrinsics.fx}\n")
            f.write(f"Camera.fy: {self.color_intrinsics.fy}\n")
            f.write(f"Camera.cx: {self.color_intrinsics.ppx}\n")
            f.write(f"Camera.cy: {self.color_intrinsics.ppy}\n")
            
            f.write("\n# Distortion Parameters (k1, k2, p1, p2, k3)\n")
            f.write(f"Camera.k1: {self.color_intrinsics.coeffs[0]}\n")
            f.write(f"Camera.k2: {self.color_intrinsics.coeffs[1]}\n")
            f.write(f"Camera.p1: {self.color_intrinsics.coeffs[2]}\n")
            f.write(f"Camera.p2: {self.color_intrinsics.coeffs[3]}\n")
            f.write(f"Camera.k3: 0.0\n")
            
            f.write("\n# Image dimensions\n")
            f.write(f"Camera.width: {self.color_intrinsics.width}\n")
            f.write(f"Camera.height: {self.color_intrinsics.height}\n")
            
            f.write("\n# ORB_SLAM3 Parameters (adjust as needed)\n")
            f.write("Camera.fps: 30.0\n")
            f.write("Camera.RGB: 1\n")
            f.write("\n# ORB Extractor Parameters\n")
            f.write("ORBextractor.nFeatures: 1000\n")
            f.write("ORBextractor.scaleFactor: 1.2\n")
            f.write("ORBextractor.nLevels: 8\n")
            f.write("ORBextractor.iniThFAST: 20\n")
            f.write("ORBextractor.minThFAST: 7\n")
            
            f.write("\n# Depth Parameters\n")
            f.write("Camera.bf: 40.0\n")  # This is a placeholder, adjust if needed
            f.write(f"Camera.depthScale: {1.0 / self.depth_scale}\n")  # Inverse of depth scale
        
        print("Camera intrinsics saved to camera_intrinsics.txt")
            
    def start_recording(self):
        """Start recording data"""
        if not self.recording:
            # Create folders for RGB and depth images
            self.dataset_folder = os.path.join(self.output_folder, self.timestamp)
            self.rgb_folder = os.path.join(self.dataset_folder, "rgb")
            self.depth_folder = os.path.join(self.dataset_folder, "depth")
            
            os.makedirs(self.dataset_folder, exist_ok=True)
            os.makedirs(self.rgb_folder, exist_ok=True)
            os.makedirs(self.depth_folder, exist_ok=True)
            
            # Create association file for ORB_SLAM3
            self.association_file = open(os.path.join(self.dataset_folder, "associations.txt"), "w")
            
            # Reset frame counter
            self.frame_count = 0
            self.recording = True
            print(f"Recording started. Saving to {self.dataset_folder}")
            
    def stop_recording(self):
        """Stop recording data"""
        if self.recording:
            self.recording = False
            if hasattr(self, 'association_file') and self.association_file:
                self.association_file.close()
                
            # Copy camera_intrinsics.txt to the dataset folder
            with open(os.path.join(self.output_folder, "camera_intrinsics.txt"), "r") as src:
                with open(os.path.join(self.dataset_folder, "camera_intrinsics.txt"), "w") as dst:
                    dst.write(src.read())
                    
            print(f"Recording stopped. Saved {self.frame_count} frames.")
            print(f"Data saved to {self.dataset_folder} in ORB_SLAM3 compatible format.")
    
    def process_frames(self):
        """Process frames and save them in ORB_SLAM3 format"""
        try:
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
            
            # Convert RGB to BGR for OpenCV display
            color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            
            # Create colormap from depth image for display
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            if self.recording:
                # Get current timestamp in seconds
                timestamp = time.time()
                
                # Save RGB image (for ORB_SLAM3, we use the timestamp as the filename)
                rgb_filename = f"{timestamp:.6f}.png"
                rgb_path = os.path.join(self.rgb_folder, rgb_filename)
                cv2.imwrite(rgb_path, color_image_bgr)
                
                # Convert depth to meters and scale to 16-bit for ORB_SLAM3
                # ORB_SLAM3 expects depth values in meters saved as 16-bit PNG
                # with values = depth_in_meters * depth_factor
                depth_factor = 5000  # Standard factor used by many RGB-D datasets
                depth_meters = depth_image.astype(np.float32) * self.depth_scale
                depth_scaled = (depth_meters * depth_factor).astype(np.uint16)
                
                # Save depth image
                depth_filename = f"{timestamp:.6f}.png"
                depth_path = os.path.join(self.depth_folder, depth_filename)
                cv2.imwrite(depth_path, depth_scaled)
                
                # Write to association file
                # Format: rgb_timestamp rgb_filename depth_timestamp depth_filename
                self.association_file.write(f"{timestamp:.6f} rgb/{rgb_filename} {timestamp:.6f} depth/{depth_filename}\n")
                self.association_file.flush()
                
                self.frame_count += 1
                
            return color_image_bgr, depth_image, depth_colormap
            
        except Exception as e:
            print(f"Error processing frames: {str(e)}")
            return None, None, None
            
    def stop(self):
        """Stop the RealSense pipeline"""
        try:
            if self.recording:
                self.stop_recording()
                
            # Only stop the pipeline if it's already running
            if hasattr(self, 'profile') and self.profile:
                self.pipeline.stop()
                
            print("Camera stopped.")
        except Exception as e:
            print(f"Error stopping camera: {str(e)}")

def main():
    try:
        # Check if RealSense devices are connected
        ctx = rs.context()
        devices = []
        
        for i in range(ctx.devices.size()):
            dev = ctx.devices[i]
            devices.append({
                'name': dev.get_info(rs.camera_info.name),
                'serial': dev.get_info(rs.camera_info.serial_number)
            })
        
        if not devices:
            print("No Intel RealSense devices detected!")
            return
            
        print(f"Found {len(devices)} RealSense device(s):")
        for i, dev in enumerate(devices):
            print(f"  Device {i}: {dev['name']} (S/N: {dev['serial']})")
        
        # Create recorder
        recorder = L515OrbSlamRecorder()
        
        # Start camera
        if not recorder.start():
            print("Failed to start camera.")
            return
        
        # Create window for display
        cv2.namedWindow('L515 ORB_SLAM3 Recorder', cv2.WINDOW_AUTOSIZE)
        
        print("Controls:")
        print("  Press 'r' to start/stop recording")
        print("  Press 'q' to quit")
        
        while True:
            # Process frames
            color_image, depth_image, depth_colormap = recorder.process_frames()
            
            if color_image is None:
                continue
                
            # Create display image
            display_image = np.hstack((color_image, depth_colormap))
            
            # Add recording indicator
            if recorder.recording:
                # Add red circle to indicate recording
                cv2.circle(display_image, (30, 30), 15, (0, 0, 255), -1)
                
                # Add frame counter
                cv2.putText(display_image, f"Frame: {recorder.frame_count}", 
                            (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Show images
            cv2.imshow('L515 ORB_SLAM3 Recorder', display_image)
            
            # Check for keyboard input
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:  # q or ESC
                break
            elif key & 0xFF == ord('r'):
                if recorder.recording:
                    recorder.stop_recording()
                else:
                    recorder.start_recording()
                    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Stop the camera and close windows
        if 'recorder' in locals():
            recorder.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
