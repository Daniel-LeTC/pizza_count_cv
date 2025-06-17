```python
import os
import random
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
```


```python
# Parsing filename
def parse_video_filename(filename):
    """
    Extract CCTV video file names for metadata.
    Format: SEQUENCE_CHANNEL_YYYYMMDDHHMMSS_HHMMSS.mp4
    Example: 1461_CH01_20250607193711_203711.mp4
    """
    # Regular expression for file name component
    pattern = re.compile(r"(\d+)_CH(\d+)_(\d{14})_(\d{6})\.mp4")
    match = pattern.match(filename)

    if not match:
        return None

    sequence_id, channel_id, start_timestamp_str, end_time_str = match.groups()

    # Understanding timestamp
    start_datetime = datetime.strptime(start_timestamp_str, "%Y%m%d%H%M%S")

    # Combine start and end datetime
    end_datetime = datetime.strptime(start_datetime.strftime("%Y%m%d") + end_time_str, "%Y%m%d%H%M%S")

    return {
        "filename": filename,
        "sequence_id": sequence_id,
        "camera_id": f"CH{channel_id.zfill(2)}",  # Ensure format CH01, CH02
        "start_time": start_datetime,
        "end_time": end_datetime,
        "duration_minutes": (end_datetime - start_datetime).total_seconds() / 60,
    }
```


```python
# Read video file name
raw_data_path = "../data/raw/"
# Check if path exists and list files
if os.path.exists(raw_data_path):
    print(f"✓ Dir exists: {raw_data_path}")

    # Get all files in directory
    all_files = os.listdir(raw_data_path)
    print(f"Total files: {len(all_files)}")
    print("files list:")
    for file in all_files:
        print(f"  - {file}")
else:
    print(f"✗ Dir not existed: {raw_data_path}")
    print("Please check your stated raw data dir")
```

    ✓ Dir exists: ../data/raw/
    Total files: 6
    files list:
      - 1461_CH01_20250607193711_203711.mp4
      - 1462_CH03_20250607192844_202844.mp4
      - 1462_CH04_20250607210159_211703.mp4
      - 1464_CH02_20250607180000_190000.mp4
      - 1465_CH02_20250607170555_172408.mp4
      - 1467_CH04_20250607180000_190000.mp4



```python
# Filter to get only .mp4 files
mp4_files = [f for f in all_files if f.lower().endswith(".mp4")]
print(f"\nQuantity of file .mp4: {len(mp4_files)}")
print("List of file video:")
for video_file in mp4_files:
    print(f"  - {video_file}")
```

    
    Quantity of file .mp4: 6
    List of file video:
      - 1461_CH01_20250607193711_203711.mp4
      - 1462_CH03_20250607192844_202844.mp4
      - 1462_CH04_20250607210159_211703.mp4
      - 1464_CH02_20250607180000_190000.mp4
      - 1465_CH02_20250607170555_172408.mp4
      - 1467_CH04_20250607180000_190000.mp4



```python
# Parse metadata from each video file
parsed_data = []
failed_files = []

print("\n--- Metadata Extraction Process ---")
for video_file in mp4_files:
    print(f"\nProcessing: {video_file}")
    
    metadata = parse_video_filename(video_file)
    
    if metadata:
        parsed_data.append(metadata)
        print(f"✓ Success: {metadata['camera_id']} | {metadata['start_time']} -> {metadata['end_time']} | {metadata['duration_minutes']:.1f} minutes")
    else:
        failed_files.append(video_file)
        print("✗ Failed: Filename pattern mismatch")

print("\n--- Extraction Results ---")
print(f"Successful files: {len(parsed_data)}")
print(f"Failed files: {len(failed_files)}")

if failed_files:
    print("Files with pattern mismatch:")
    for failed_file in failed_files:
        print(f"  - {failed_file}")```
```
    
    --- Metadata Extraction Process ---
    
    Processing: 1461_CH01_20250607193711_203711.mp4
    ✓ Success: CH01 | 2025-06-07 19:37:11 -> 2025-06-07 20:37:11 | 60.0 minutes
    
    Processing: 1462_CH03_20250607192844_202844.mp4
    ✓ Success: CH03 | 2025-06-07 19:28:44 -> 2025-06-07 20:28:44 | 60.0 minutes
    
    Processing: 1462_CH04_20250607210159_211703.mp4
    ✓ Success: CH04 | 2025-06-07 21:01:59 -> 2025-06-07 21:17:03 | 15.1 minutes
    
    Processing: 1464_CH02_20250607180000_190000.mp4
    ✓ Success: CH02 | 2025-06-07 18:00:00 -> 2025-06-07 19:00:00 | 60.0 minutes
    
    Processing: 1465_CH02_20250607170555_172408.mp4
    ✓ Success: CH02 | 2025-06-07 17:05:55 -> 2025-06-07 17:24:08 | 18.2 minutes
    
    Processing: 1467_CH04_20250607180000_190000.mp4
    ✓ Success: CH04 | 2025-06-07 18:00:00 -> 2025-06-07 19:00:00 | 60.0 minutes
    
    --- Extraction Results ---
    Successful files: 6
    Failed files: 0



```python
# Create DataFrame from parsed data
if parsed_data:
    df_videos = pd.DataFrame(parsed_data)
    
    # Sort by start time for better visualization
    df_videos = df_videos.sort_values("start_time").reset_index(drop=True)
    
    print("=== VIDEO METADATA SUMMARY ===")
    print(df_videos.to_string(index=False))
    
    # Some basic statistics
    print("\n=== DATASET STATISTICS ===")
    print(f"Total video count: {len(df_videos)}")
    print(f"Total duration: {df_videos['duration_minutes'].sum():.1f} minutes")
    print(f"Average duration: {df_videos['duration_minutes'].mean():.1f} minutes")
    print(f"Camera channels used: {sorted(df_videos['camera_id'].unique())}")
    print(f"Time range: {df_videos['start_time'].min()} -> {df_videos['end_time'].max()}")
else:
    print("No files were successfully parsed!")
```

    
    === VIDEO METADATA SUMMARY ===
                               filename sequence_id camera_id          start_time            end_time  duration_minutes
    1465_CH02_20250607170555_172408.mp4        1465      CH02 2025-06-07 17:05:55 2025-06-07 17:24:08         18.216667
    1464_CH02_20250607180000_190000.mp4        1464      CH02 2025-06-07 18:00:00 2025-06-07 19:00:00         60.000000
    1467_CH04_20250607180000_190000.mp4        1467      CH04 2025-06-07 18:00:00 2025-06-07 19:00:00         60.000000
    1462_CH03_20250607192844_202844.mp4        1462      CH03 2025-06-07 19:28:44 2025-06-07 20:28:44         60.000000
    1461_CH01_20250607193711_203711.mp4        1461      CH01 2025-06-07 19:37:11 2025-06-07 20:37:11         60.000000
    1462_CH04_20250607210159_211703.mp4        1462      CH04 2025-06-07 21:01:59 2025-06-07 21:17:03         15.066667
    
    === DATASET STATISTICS ===
    Total video count: 6
    Total duration: 273.3 minutes
    verage duration: 45.5 minutes
    Camera channels used: ['CH01', 'CH02', 'CH03', 'CH04']
    Time range: 2025-06-07 17:05:55 -> 2025-06-07 21:17:03




```python
def create_camera_processing_plan(df_videos, df_overlaps):
    """
    Create processing plan for 6 independent cameras based on parsed data
    """
    print("=== CAMERA PROCESSING STRATEGY ===")
    
    # Create unique camera info from df_videos
    cameras = []
    for idx, video in df_videos.iterrows():
        camera_info = {
            "camera_id": f"CAM{idx+1:02d}",  # CAM01, CAM02, etc.
            "original_filename": video["filename"],
            "original_channel": video["camera_id"],
            "start_time": video["start_time"],
            "end_time": video["end_time"],
            "duration_minutes": video["duration_minutes"],
            "video_index": idx,
            "has_overlap": False,
            "overlap_partners": []
        }
        cameras.append(camera_info)
    
    # Mark cameras with overlap
    if not df_overlaps.empty:
        for _, overlap in df_overlaps.iterrows():
            cameras[overlap["video1_idx"]]["has_overlap"] = True
            cameras[overlap["video2_idx"]]["has_overlap"] = True
            
            cameras[overlap["video1_idx"]]["overlap_partners"].append({
                "partner_cam": f"CAM{overlap['video2_idx']+1:02d}",
                "overlap_start": overlap["overlap_start"],
                "overlap_end": overlap["overlap_end"],
                "overlap_minutes": overlap["overlap_minutes"]
            })
            
            cameras[overlap["video2_idx"]]["overlap_partners"].append({
                "partner_cam": f"CAM{overlap['video1_idx']+1:02d}",
                "overlap_start": overlap["overlap_start"],
                "overlap_end": overlap["overlap_end"],
                "overlap_minutes": overlap["overlap_minutes"]
            })
    
    df_cameras = pd.DataFrame(cameras)
    
    print(f"Defined {len(df_cameras)} independent cameras")
    for _, cam in df_cameras.iterrows():
        overlap_info = f" (len(cam['overlap_partners'])} overlap)" if cam["has_overlap"] else ""
        print(f"  • {cam['camera_id']}: {cam['original_filename']}{overlap_info}")
    
    return df_cameras

# Create camera processing plan
df_cameras = create_camera_processing_plan(df_videos, df_overlaps)
```

    === CAMERA PROCESSING STRATEGY ===
    Define 6 independent camera:
      • CAM_01: 1465_CH02_20250607170555_172408.mp4
      • CAM_02: 1464_CH02_20250607180000_190000.mp4 (1 overlap)
      • CAM_03: 1467_CH04_20250607180000_190000.mp4 (1 overlap)
      • CAM_04: 1462_CH03_20250607192844_202844.mp4 (1 overlap)
      • CAM_05: 1461_CH01_20250607193711_203711.mp4 (1 overlap)
      • CAM_06: 1462_CH04_20250607210159_211703.mp4



```python
def analyze_video_properties(df_cameras, raw_data_path):
    """
    Analyze technical properties from actual video files
    """
    print("=== ANALYZING VIDEO PROPERTIES ===")
    
    video_props = []
    for _, camera in df_cameras.iterrows():
        video_path = os.path.join(raw_data_path, camera["original_filename"])
        print(f"Analyzing {camera['camera_id']} - {camera['original_filename']}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"  Cannot open video: {camera['original_filename']}")
            continue
        
        # Extract properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_duration = frame_count / fps if fps > 0 else 0
        file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
        
        cap.release()
        
        props = {
            "camera_id": camera["camera_id"],
            "filename": camera["original_filename"],
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}",
            "actual_duration_minutes": actual_duration / 60,
            "file_size_mb": file_size,
            "has_overlap": camera["has_overlap"]
        }
        
        video_props.append(props)
        print(f"  {props['resolution']} @ {props['fps']:.1f}fps, {props['frame_count']:,} frames, {props['file_size_mb']:.1f}MB")
    
    df_props = pd.DataFrame(video_props)
    
    # Summary statistics
    print("\n=== VIDEO PROPERTIES SUMMARY ===")
    print(f"Resolutions: {sorted(df_props['resolution'].unique())}")
    print(f"FPS range: {df_props['fps'].min():.1f} - {df_props['fps'].max():.1f}")
    print(f"Total frames: {df_props['frame_count'].sum():,}")
    print(f"Total size: {df_props['file_size_mb'].sum():.1f} MB")
    
    return df_props

# Analyze video properties
df_props = analyze_video_properties(df_cameras, raw_data_path)
```

    === ANALYZING VIDEO PROPERTIES ===
    Analyzing: CAM_01 - 1465_CH02_20250607170555_172408.mp4
      ✓ 1920x1080 @ 25.0fps, 27,301 frames, 232.8MB
    Analyzing: CAM_02 - 1464_CH02_20250607180000_190000.mp4
      ✓ 1920x1080 @ 15.0fps, 53,999 frames, 432.2MB
    Analyzing: CAM_03 - 1467_CH04_20250607180000_190000.mp4
      ✓ 1920x1080 @ 12.0fps, 43,200 frames, 330.1MB
    Analyzing: CAM_04 - 1462_CH03_20250607192844_202844.mp4
      ✓ 1920x1080 @ 25.0fps, 89,992 frames, 766.6MB
    Analyzing: CAM_05 - 1461_CH01_20250607193711_203711.mp4
      ✓ 1920x1080 @ 10.0fps, 35,997 frames, 374.7MB
    Analyzing: CAM_06 - 1462_CH04_20250607210159_211703.mp4
      ✓ 1920x1080 @ 25.0fps, 22,589 frames, 192.5MB
    
    === VIDEO PROPERTIES SUMMARY ===
    Resolutions: ['1920x1080']
    FPS range: 10.0 - 25.0
    Total frames: 273,078
    Total size: 2328.9 MB



```python
def design_sampling_strategy(df_props, df_cameras):
    """
    Design optimal sampling strategy for training data preparation
    """
    print("=== FRAME SAMPLING STRATEGY ===")
    
    # Analyze FPS and resolution diversity
    min_fps = df_props["fps"].min()
    max_fps = df_props["fps"].max()
    
    # Recommended sampling rate to ensure consistency
    target_fps = 1.0  # 1 frame per second for training
    
    sampling_strategies = []
    total_extracted_frames = 0
    
    for _, props in df_props.iterrows():
        # Calculate sampling interval for each video
        sampling_interval = max(1, int(props["fps"] / target_fps))
        expected_frames = props["frame_count"] // sampling_interval
        
        # Find overlap info
        camera_info = df_cameras[df_cameras["camera_id"] == props["camera_id"]].iloc[0]
        
        strategy = {
            "camera_id": props["camera_id"],
            "filename": props["filename"],
            "original_fps": props["fps"],
            "sampling_interval": sampling_interval,
            "expected_frames": expected_frames,
            "target_resolution": "1280x720",  # Standardize to 720p
            "has_overlap": camera_info["has_overlap"],
            "overlap_partners": camera_info["overlap_partners"],
        }
        
        sampling_strategies.append(strategy)
        total_extracted_frames += expected_frames
        
        overlap_info = f" (overlap with {len(camera_info['overlap_partners'])} cam)" if camera_info["has_overlap"] else ""
        print(f"  • {props['camera_id']}: sample every {sampling_interval} frames → {expected_frames:,} frames{overlap_info}")
    
    print(f"\nTotal estimated extracted frames: {total_extracted_frames:,}")
    
    return pd.DataFrame(sampling_strategies)

# Design sampling strategy
df_sampling = design_sampling_strategy(df_props, df_cameras)
```

    === FRAME SAMPLING STRATEGY ===
      • CAM_01: sample every 24 frames → ~1,137 frames
      • CAM_02: sample every 15 frames → ~3,599 frames (overlap with 1 cam)
      • CAM_03: sample every 12 frames → ~3,600 frames (overlap with 1 cam)
      • CAM_04: sample every 24 frames → ~3,749 frames (overlap with 1 cam)
      • CAM_05: sample every 9 frames → ~3,999 frames (overlap with 1 cam)
      • CAM_06: sample every 24 frames → ~941 frames
    
    Total estimated extracted frames: 17,025



```python
def extract_frames_from_video(video_path, output_dir, sampling_interval, target_size=(1280, 720)):
    """
    Extract frames from video with specified sampling interval.
    
    Args:
        video_path: Path to source video file
        output_dir: Directory to save extracted frames
        sampling_interval: Extract every Nth frame
        target_size: Resize frames to this resolution
        
    Returns:
        list: Information about extracted frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    
    extracted_frames = []
    frame_idx = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame according to sampling interval
        if frame_idx % sampling_interval == 0:
            # Resize frame to target resolution
            frame_resized = cv2.resize(frame, target_size)
            
            # Generate filename with timestamp information
            timestamp_seconds = frame_idx / cap.get(cv2.CAP_PROP_FPS)
            frame_filename = f"frame_{extracted_count:06d}_t{timestamp_seconds:.2f}s.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            
            # Save frame
            cv2.imwrite(frame_path, frame_resized)
            
            extracted_frames.append({
                "frame_path": frame_path,
                "frame_number": frame_idx,
                "timestamp_seconds": timestamp_seconds,
                "extracted_index": extracted_count
            })
            
            extracted_count += 1
        
        frame_idx += 1
    
    cap.release()
    return extracted_frames

# Test extraction với một video đầu tiên
test_camera = df_sampling.iloc[0]
test_video_path = os.path.join(raw_data_path, test_camera["filename"])
test_output_dir = f"../data/processed/frames/{test_camera['camera_id']}"

print("=== TEST FRAME EXTRACTION ===")
print(f"Testing {test_camera['camera_id']}: {test_camera['filename']}")
print(f"Sampling interval: {test_camera['sampling_interval']}")

# Extract frames (uncomment để chạy test)
extracted_frames = extract_frames_from_video(test_video_path, test_output_dir, test_camera["sampling_interval"])
print(f"✓ Extracted {len(extracted_frames)} frames into {test_output_dir}")
```

    === TEST FRAME EXTRACTION ===
    Testing CAM_01: 1465_CH02_20250607170555_172408.mp4
    Sampling interval: 24
    ✓ Extracted 1138 frames into ../data/processed/frames/CAM_01



```python
# Execute frame extraction for each camera
def batch_extract_all_videos(df_cameras, raw_data_path):
    """
    Execute frame extraction for all videos in the dataset.
    
    Args:
        df_cameras: DataFrame containing camera information
        raw_data_path: Path to raw video files
        
    Returns:
        DataFrame: Extraction results summary
    """
    print("=== BATCH FRAME EXTRACTION ===")
    
    extraction_results = []
    
    for idx, camera in df_cameras.iterrows():
        video_path = os.path.join(raw_data_path, camera["original_filename"])
        output_dir = f"../data/processed/frames/{camera['camera_id']}"
        
        print(f"\nExtracting from {camera['camera_id']} - {camera['original_filename']}")
        
        # Calculate sampling interval based on video duration
        if camera["duration_minutes"] <= 5:
            sampling_interval = 30  # Extract every 30 frames for short videos
        elif camera["duration_minutes"] <= 10:
            sampling_interval = 60  # Extract every 60 frames for medium videos
        else:
            sampling_interval = 90  # Extract every 90 frames for long videos
        
        print(f"  Sampling interval: {sampling_interval} frames")
        
        extracted_frames = extract_frames_from_video(
            video_path, output_dir, sampling_interval
        )
        
        result = {
            "camera_id": camera["camera_id"],
            "filename": camera["original_filename"],
            "frames_extracted": len(extracted_frames),
            "output_directory": output_dir,
            "sampling_interval": sampling_interval,
            "extracted_frames_info": extracted_frames
        }
        
        extraction_results.append(result)
        print(f"  ✓ Extracted {len(extracted_frames)} frames → {output_dir}")
    
    df_extraction = pd.DataFrame(extraction_results)
    
    # Extraction summary
    total_frames = df_extraction["frames_extracted"].sum()
    print("\n=== EXTRACTION SUMMARY ===")
    print(f"Total cameras processed: {len(df_extraction)}")
    print(f"Total frames extracted: {total_frames:,}")
    
    return df_extraction

# Execute batch extraction
df_extraction = batch_extract_all_videos(df_cameras, raw_data_path)
```

    === BATCH FRAME EXTRACTION ===
    
    Extracting CAM_01: 1465_CH02_20250607170555_172408.mp4
      Sampling: every 24 frames
      ✓ Extracted 1138 frames → ../data/processed/frames/CAM_01
    
    Extracting CAM_02: 1464_CH02_20250607180000_190000.mp4
      Sampling: every 15 frames
      ✓ Extracted 3600 frames → ../data/processed/frames/CAM_02
    
    Extracting CAM_03: 1467_CH04_20250607180000_190000.mp4
      Sampling: every 12 frames
      ✓ Extracted 3600 frames → ../data/processed/frames/CAM_03
    
    Extracting CAM_04: 1462_CH03_20250607192844_202844.mp4
      Sampling: every 24 frames
      ✓ Extracted 3750 frames → ../data/processed/frames/CAM_04
    
    Extracting CAM_05: 1461_CH01_20250607193711_203711.mp4
      Sampling: every 9 frames
      ✓ Extracted 4000 frames → ../data/processed/frames/CAM_05
    
    Extracting CAM_06: 1462_CH04_20250607210159_211703.mp4
      Sampling: every 24 frames
      ✓ Extracted 942 frames → ../data/processed/frames/CAM_06
    
    === EXTRACTION SUMMARY ===
    Total cameras processed: 6
    Total frames extracted: 17,030
    Cameras with overlaps: 4





```python
def select_frames_from_directories(frames_base_path, output_path, frames_per_camera=50):
    """
    Select frames directly from CAM directories and copy to annotation folder
    """
    frames_base = Path(frames_base_path)
    output_base = Path(output_path)

    # Create output directory structure
    output_base.mkdir(parents=True, exist_ok=True)

    # Find all CAM directories
    cam_directories = sorted([d for d in frames_base.iterdir() if d.is_dir() and d.name.startswith("CAM_")])

    total_selected = 0

    for cam_dir in cam_directories:
        cam_name = cam_dir.name
        print(f"Processing {cam_name}...")

        # Get all frames in directory
        all_frames = sorted(list(cam_dir.glob("*.jpg")))

        if not all_frames:
            print(f"  ⚠️  No frames found in {cam_name}")
            continue

        # Selection strategy: evenly distributed
        if len(all_frames) > frames_per_camera:
            step = len(all_frames) // frames_per_camera
            selected = [all_frames[i] for i in range(0, len(all_frames), step)][:frames_per_camera]
        else:
            selected = all_frames

        # Create output directory for this camera
        output_cam_dir = output_base / cam_name
        output_cam_dir.mkdir(exist_ok=True)

        # Copy selected frames
        for frame_path in selected:
            dest_path = output_cam_dir / frame_path.name
            shutil.copy2(frame_path, dest_path)

        print(f"  ✓ {cam_name}: {len(selected)}/{len(all_frames)} frames copied")
        total_selected += len(selected)

    print(f"\n🎯 Total selected: {total_selected} frames")
    print(f"📁 Location: {output_base}")
    return total_selected


# Execute selection
frames_path = "../data/processed/frames"
annotation_path = "../data/processed/annotations/selected"

selected_count = select_frames_from_directories(frames_path, annotation_path, frames_per_camera=50)```
```
    Processing CAM_01...
      ✓ CAM_01: 50/1138 frames copied
    Processing CAM_02...
      ✓ CAM_02: 50/3600 frames copied
    Processing CAM_03...
      ✓ CAM_03: 50/3600 frames copied
    Processing CAM_04...
      ✓ CAM_04: 50/3750 frames copied
    Processing CAM_05...
      ✓ CAM_05: 50/4000 frames copied
    Processing CAM_06...
      ✓ CAM_06: 50/942 frames copied
    
    🎯 Total selected: 300 frames
    📁 Location: ../data/processed/annotation/selected



```python
def get_existing_selected_frames():
    """Get list of all frames already selected from selected folder"""
    selected_frames = set()
    selected_folder = "../data/processed/annotations/selected"

    if os.path.exists(selected_folder):
        # Walk through all files and subfolders
        for root, dirs, files in os.walk(selected_folder):
            for file in files:
                if file.endswith((".jpg", ".png", ".jpeg")):
                    selected_frames.add(file)

    return selected_frames


def get_all_source_frames():
    """Get all 17k source frames from processed/frames/CAM_XX"""
    all_frames = []

    for cam_id in range(1, 7):  # CAM_01 to CAM_06
        frames_folder = f"../data/processed/frames/CAM_{cam_id:02d}"

        if os.path.exists(frames_folder):
            print(f"Scanning {frames_folder}...")
            for frame_file in os.listdir(frames_folder):
                if frame_file.endswith((".jpg", ".png", ".jpeg")):
                    frame_info = {"filename": frame_file, "full_path": os.path.join(frames_folder, frame_file), "cam_id": f"CAM_{cam_id:02d}"}
                    all_frames.append(frame_info)
        else:
            print(f"Directory not found: {frames_folder}")

    return all_frames


def select_additional_300_frames():
    """Select additional 300 new frames for selected_2"""

    # Get list of frames already selected in selected
    existing_selected = get_existing_selected_frames()
    print(f"Already have {len(existing_selected)} frames in ../data/processed/annotations/selected")

    # Get all 17k source frames
    all_source_frames = get_all_source_frames()
    print(f"Total of {len(all_source_frames)} source frames in processed/frames/CAM_XX")

    # Filter out frames that haven't been selected yet
    available_frames = [frame for frame in all_source_frames if frame["filename"] not in existing_selected]

    print(f"Have {len(available_frames)} frames not yet selected")

    if len(available_frames) < 300:
        print(f"Warning: Only have {len(available_frames)} available frames, less than 300 frames requested")
        frames_to_select = available_frames
    else:
        # Randomly select 300 frames
        frames_to_select = random.sample(available_frames, 300)

    # Create destination directory selected_2
    output_folder = "../data/processed/annotations/selected_2"
    os.makedirs(output_folder, exist_ok=True)
    print(f"Created directory: {output_folder}")

    # Copy selected frames
    print("Copying frames to selected_2...")
    for i, frame in enumerate(frames_to_select):
        dest_path = os.path.join(output_folder, frame["filename"])
        shutil.copy2(frame["full_path"], dest_path)

        if (i + 1) % 50 == 0:
            print(f"Copied {i + 1}/{len(frames_to_select)} frames")

    # Statistics distribution by camera
    cam_distribution = {}
    for frame in frames_to_select:
        cam_id = frame["cam_id"]
        cam_distribution[cam_id] = cam_distribution.get(cam_id, 0) + 1

    print(f"\nSelected additional {len(frames_to_select)} frames for selected_2")
    print("Distribution by camera:")
    for cam_id, count in sorted(cam_distribution.items()):
        print(f"  {cam_id}: {count} frames")

    # Create log file for tracking
    log_file = os.path.join(output_folder, "selection_log.txt")
    with open(log_file, "w") as f:
        f.write(f"Selected {len(frames_to_select)} additional frames\n")
        f.write("Distribution by camera:\n")
        for cam_id, count in sorted(cam_distribution.items()):
            f.write(f"  {cam_id}: {count} frames\n")
        f.write("\nSelected files:\n")
        for frame in frames_to_select:
            f.write(f"{frame['filename']} (from {frame['cam_id']})\n")

    return frames_to_select


# Execute
if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)

    selected_frames = select_additional_300_frames()
    print(f"\nCompleted! {len(selected_frames)} frames have been saved to ../data/processed/annotations/selected_2/")
    print("Detailed log file created: ../data/processed/annotations/selected_2/selection_log.txt")

```

    Already have 241 frames in ../data/processed/annotations/selected
    Scanning ../data/processed/frames/CAM_01...
    Scanning ../data/processed/frames/CAM_02...
    Scanning ../data/processed/frames/CAM_03...
    Scanning ../data/processed/frames/CAM_04...
    Scanning ../data/processed/frames/CAM_05...
    Scanning ../data/processed/frames/CAM_06...
    Total of 17030 source frames in processed/frames/CAM_XX
    Have 16532 frames not yet selected
    Created directory: ../data/processed/annotations/selected_2
    Copying frames to selected_2...
    Copied 50/300 frame
    Copied 100/300 frame
    Copied 150/300 frame
    Copied 200/300 frame
    Copied 250/300 frame
    Copied 300/300 frame
    
    Selected additional 300 frames for selected_2
    Distribution by camera:
      CAM_01: 17 frame
      CAM_02: 80 frame
      CAM_03: 69 frame
      CAM_04: 60 frame
      CAM_05: 63 frame
      CAM_06: 11 frame
    
    Completed! 300 frames have been saved to ../data/processed/annotations/selected_2/
    Detailed log file created: ../data/processed/annotations/selected_2/selection_log.txt

