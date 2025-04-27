#!/usr/bin/env python3

import os
import csv
import sys
from tkinter import Tk, filedialog, Label, Button, Entry, StringVar

def create_associations(csv_file, rgb_folder, depth_folder):
    """
    Create associations.txt file from CSV timestamps and frame folders
    
    Args:
        csv_file: Path to CSV file with timestamps
        rgb_folder: Path to RGB frames folder
        depth_folder: Path to depth frames folder
    """
    
    # Get output directory (same as CSV location)
    output_dir = os.path.dirname(csv_file)
    associations_file = os.path.join(output_dir, "associations.txt")
    
    # Read timestamps from CSV
    timestamps = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        # Skip header
        header = next(reader)
        
        # Find timestamp column index
        try:
            timestamp_col = header.index("timestamp")
        except ValueError:
            print("Error: CSV file does not have a 'timestamp' column")
            return False
        
        # Read timestamps
        for row in reader:
            if len(row) > timestamp_col:
                timestamps.append(row[timestamp_col])
    
    # Check if we have timestamps
    if not timestamps:
        print("Error: No timestamps found in the CSV file")
        return False
    
    print(f"Found {len(timestamps)} timestamps in CSV file")
    
    # Create associations file
    with open(associations_file, 'w') as f:
        for i, timestamp in enumerate(timestamps):
            # Format frame number
            frame_num = f"{i:06d}"
            
            # Get the RGB and depth frame paths
            rgb_filename = f"frame_{frame_num}.jpg"
            depth_filename = f"frame_{frame_num}.png"
            
            rgb_path = os.path.join(rgb_folder, rgb_filename)
            depth_path = os.path.join(depth_folder, depth_filename)
            
            # Debug output
            if i < 5 or not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                print(f"Frame {frame_num} - RGB path: {rgb_path} (exists: {os.path.exists(rgb_path)})")
                print(f"Frame {frame_num} - Depth path: {depth_path} (exists: {os.path.exists(depth_path)})")
            
            # Skip this frame if files don't exist
            if not os.path.exists(rgb_path):
                print(f"Warning: RGB frame not found: {rgb_path}")
                # Try to list the files in the directory to see what's available
                try:
                    print(f"Files in {os.path.dirname(rgb_path)}:")
                    files = os.listdir(os.path.dirname(rgb_path))
                    print(", ".join(files[:5]) + (", ..." if len(files) > 5 else ""))
                except Exception as e:
                    print(f"Error listing directory: {str(e)}")
                continue
                
            if not os.path.exists(depth_path):
                print(f"Warning: Depth frame not found: {depth_path}")
                continue
            
            # Write association line - use path relative to dataset root for ORB_SLAM3
            rgb_rel_path = os.path.join("rgb", rgb_filename)
            depth_rel_path = os.path.join("depth", depth_filename)
            f.write(f"{timestamp} {rgb_rel_path} {timestamp} {depth_rel_path}\n")
    
    print(f"Successfully created associations file: {associations_file}")
    return True
    
    print(f"Successfully created associations file: {associations_file}")
    return True

def browse_file(entry_var):
    """Open file browser dialog and update entry"""
    filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if filename:
        entry_var.set(filename)

def browse_folder(entry_var):
    """Open folder browser dialog and update entry"""
    folder = filedialog.askdirectory()
    if folder:
        entry_var.set(folder)

def create_gui():
    """Create GUI interface"""
    root = Tk()
    root.title("CSV to Associations.txt Converter")
    root.geometry("600x250")
    
    # CSV file selection
    csv_var = StringVar()
    Label(root, text="CSV File with Timestamps:").grid(row=0, column=0, sticky="w", padx=10, pady=10)
    Entry(root, textvariable=csv_var, width=50).grid(row=0, column=1, padx=5, pady=10)
    Button(root, text="Browse", command=lambda: browse_file(csv_var)).grid(row=0, column=2, padx=5, pady=10)
    
    # RGB folder selection
    rgb_var = StringVar()
    Label(root, text="RGB Frames Folder:").grid(row=1, column=0, sticky="w", padx=10, pady=10)
    Entry(root, textvariable=rgb_var, width=50).grid(row=1, column=1, padx=5, pady=10)
    Button(root, text="Browse", command=lambda: browse_folder(rgb_var)).grid(row=1, column=2, padx=5, pady=10)
    
    # Depth folder selection
    depth_var = StringVar()
    Label(root, text="Depth Frames Folder:").grid(row=2, column=0, sticky="w", padx=10, pady=10)
    Entry(root, textvariable=depth_var, width=50).grid(row=2, column=1, padx=5, pady=10)
    Button(root, text="Browse", command=lambda: browse_folder(depth_var)).grid(row=2, column=2, padx=5, pady=10)
    
    # Status display
    status_var = StringVar()
    status_var.set("Ready")
    status_label = Label(root, textvariable=status_var, bd=1, relief="sunken", anchor="w")
    status_label.grid(row=4, column=0, columnspan=3, sticky="we", padx=10, pady=10)
    
    # Create button
    def on_create():
        csv_file = csv_var.get()
        rgb_folder = rgb_var.get()
        depth_folder = depth_var.get()
        
        if not csv_file or not os.path.exists(csv_file):
            status_var.set("Error: Invalid CSV file path")
            return
            
        if not rgb_folder or not os.path.exists(rgb_folder):
            status_var.set("Error: Invalid RGB frames folder path")
            return
            
        if not depth_folder or not os.path.exists(depth_folder):
            status_var.set("Error: Invalid depth frames folder path")
            return
        
        status_var.set("Creating associations.txt...")
        root.update()
        
        if create_associations(csv_file, rgb_folder, depth_folder):
            status_var.set("Success! Created associations.txt in same folder as CSV")
        else:
            status_var.set("Error creating associations.txt. Check console for details.")
    
    Button(root, text="Create Associations.txt", command=on_create).grid(row=3, column=0, columnspan=3, pady=10)
    
    root.mainloop()

def main():
    """Main function with argparse for command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create associations.txt file for ORB_SLAM3 from CSV timestamps')
    parser.add_argument('--csv', '-c', type=str, help='Path to CSV file with timestamps', required=True)
    parser.add_argument('--rgb', '-r', type=str, help='Path to RGB frames folder', required=True)
    parser.add_argument('--depth', '-d', type=str, help='Path to depth frames folder', required=True)
    parser.add_argument('--gui', '-g', action='store_true', help='Launch GUI interface')
    
    args = parser.parse_args()
    
    if args.gui:
        # GUI mode
        create_gui()
        return
    
    # Command line mode
    csv_file = args.csv
    rgb_folder = args.rgb
    depth_folder = args.depth
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        return
        
    if not os.path.exists(rgb_folder):
        print(f"Error: RGB folder not found: {rgb_folder}")
        return
        
    if not os.path.exists(depth_folder):
        print(f"Error: Depth folder not found: {depth_folder}")
        return
        
    create_associations(csv_file, rgb_folder, depth_folder)

if __name__ == "__main__":
    main()
