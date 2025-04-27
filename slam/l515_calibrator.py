#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import json
import os
from datetime import datetime

def get_intrinsics_as_dict(intrinsics):
    """Convert rs.intrinsics to a dictionary for easier serialization"""
    return {
        'width': intrinsics.width,
        'height': intrinsics.height,
        'ppx': intrinsics.ppx,  # Principal point X
        'ppy': intrinsics.ppy,  # Principal point Y
        'fx': intrinsics.fx,    # Focal length X
        'fy': intrinsics.fy,    # Focal length Y
        'model': str(intrinsics.model),  # Distortion model
        'coeffs': list(intrinsics.coeffs)  # Distortion coefficients - convert to list directly
    }

def get_extrinsics_as_dict(extrinsics):
    """Convert rs.extrinsics to a dictionary for easier serialization"""
    # Handle the case where rotation and translation might already be lists
    rotation = list(extrinsics.rotation) if not isinstance(extrinsics.rotation, list) else extrinsics.rotation
    translation = list(extrinsics.translation) if not isinstance(extrinsics.translation, list) else extrinsics.translation
    
    return {
        'rotation': rotation,  # 3x3 rotation matrix
        'translation': translation  # Translation vector
    }

def main():
    try:
        print("Connecting to Intel RealSense L515 camera...")
        
        # Check for available devices
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
        
        # Create pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Enable device by serial number (if multiple cameras connected)
        if len(devices) > 1:
            selected = int(input("Multiple devices found. Enter device number to use: "))
            config.enable_device(devices[selected]['serial'])
            print(f"Using device: {devices[selected]['name']} (S/N: {devices[selected]['serial']})")
        
        # Enable any stream to get access to the device
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Start pipeline to access device properties
        print("Starting camera to access calibration data...")
        profile = pipeline.start(config)
        
        # Get device
        dev = profile.get_device()
        
        # Create a dictionary to store all calibration information
        calibration_data = {
            'device_info': {
                'name': dev.get_info(rs.camera_info.name),
                'serial_number': dev.get_info(rs.camera_info.serial_number),
                'firmware_version': dev.get_info(rs.camera_info.firmware_version),
                'physical_port': dev.get_info(rs.camera_info.physical_port) if dev.supports(rs.camera_info.physical_port) else "unknown",
                'product_id': dev.get_info(rs.camera_info.product_id),
                'camera_locked': dev.get_info(rs.camera_info.camera_locked) if dev.supports(rs.camera_info.camera_locked) else "unknown"
            },
            'intrinsics': {},
            'extrinsics': {}
        }
        
        # Get the active streams
        depth_sensor = dev.first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        calibration_data['depth_scale'] = depth_scale
        
        sensors = dev.query_sensors()
        print(f"\nDevice has {len(sensors)} sensors:")
        for i, sensor in enumerate(sensors):
            print(f"  Sensor {i}: {sensor.get_info(rs.camera_info.name)}")
            
            # Get all possible stream profiles for this sensor
            stream_profiles = sensor.get_stream_profiles()
            print(f"    Available streams: {len(stream_profiles)}")
            
        # Get the intrinsics and extrinsics of enabled streams
        print("\nCollecting calibration data for enabled streams...")
        depth_profile = pipeline.get_active_profile().get_stream(rs.stream.depth)
        depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        calibration_data['intrinsics']['depth'] = get_intrinsics_as_dict(depth_intrinsics)
        
        # Get extrinsics between streams
        streams = [rs.stream.depth, rs.stream.color, rs.stream.infrared]
        stream_names = {
            rs.stream.depth: 'depth',
            rs.stream.color: 'color',
            rs.stream.infrared: 'infrared'
        }
        
        # Try to get extrinsics for each combination of streams
        for i, source_stream in enumerate(streams):
            for target_stream in streams[i+1:]:
                try:
                    # Try to get the profiles for these streams
                    source_profile = None
                    target_profile = None
                    
                    for profile in pipeline.get_active_profile().get_streams():
                        if profile.stream_type() == source_stream:
                            source_profile = profile
                        elif profile.stream_type() == target_stream:
                            target_profile = profile
                    
                    if source_profile and target_profile:
                        extrinsics = source_profile.get_extrinsics_to(target_profile)
                        source_name = stream_names[source_stream]
                        target_name = stream_names[target_stream]
                        key = f"{source_name}_to_{target_name}"
                        calibration_data['extrinsics'][key] = get_extrinsics_as_dict(extrinsics)
                except Exception as e:
                    continue
        
        # Create output folder if it doesn't exist
        output_folder = "calibration_data"
        os.makedirs(output_folder, exist_ok=True)
        
        # Generate timestamp for this calibration
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save calibration data to JSON file
        calibration_file = os.path.join(output_folder, f"l515_calibration_{timestamp}.json")
        with open(calibration_file, 'w') as f:
            json.dump(calibration_data, f, indent=4)
            
        print(f"\nCalibration data saved to: {calibration_file}")
        
        # Also print a summary of the calibration data
        print("\nCalibration Data Summary:")
        print("-------------------------")
        print(f"Device: {calibration_data['device_info']['name']} (S/N: {calibration_data['device_info']['serial_number']})")
        print(f"Firmware: {calibration_data['device_info']['firmware_version']}")
        print(f"Depth Scale: {calibration_data['depth_scale']} meters/unit")
        
        # Print camera intrinsics
        if 'depth' in calibration_data['intrinsics']:
            depth_intr = calibration_data['intrinsics']['depth']
            print("\nDepth Camera Intrinsics:")
            print(f"  Resolution: {depth_intr['width']}x{depth_intr['height']}")
            print(f"  Principal Point: ({depth_intr['ppx']}, {depth_intr['ppy']})")
            print(f"  Focal Length: ({depth_intr['fx']}, {depth_intr['fy']})")
            print(f"  Distortion Model: {depth_intr['model']}")
            print(f"  Distortion Coefficients: {depth_intr['coeffs']}")
            
        # Stop the pipeline
        pipeline.stop()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
if __name__ == "__main__":
    main()
