import tempfile

import streamlit as st

from core.robust_processor import RobustZoneProcessor


def process_video_with_real_config(video_path, video_name, camera_id, model_name, model, max_frames, start_frame, conf_threshold, probation_frames, dispatch_timer):
    """Process video with real camera configuration"""

    from config.camera_config import REAL_CAMERA_CONFIGS

    # Use real camera configuration
    camera_config = REAL_CAMERA_CONFIGS[camera_id].copy()
    camera_config.update({"source_video_path": video_path, "target_video_path": f"output/robust_processed_{camera_id}_{video_name}"})

    camera_configs = {camera_id: camera_config}

    # Progress tracking containers
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()

    try:
        # Initialize RobustZoneProcessor with real configurations
        processor = RobustZoneProcessor(camera_configs, model)
        processor.confidence_threshold = conf_threshold
        processor.min_frames_for_confirmation = probation_frames
        processor.dispatch_threshold = dispatch_timer

        status_text.text(f"🔄 Initializing Robust Zone Processor for {camera_id}...")
        progress_bar.progress(10)

        # Process video with robust tracking
        status_text.text(f"🎬 Processing {camera_id} video with probation system...")

        final_sales = processor.process_partial_video(camera_id, max_frames=max_frames, start_frame=start_frame)

        progress_bar.progress(100)
        status_text.text(f"✅ {camera_id} processing completed successfully!")

        return final_sales, processor

    except Exception as e:
        st.error(f"❌ Error during {camera_id} processing: {str(e)}")
        st.exception(e)
        return None, None


def save_uploaded_video(video_file):
    """Save uploaded video to temporary file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video_file.getbuffer())
        return tmp_file.name
