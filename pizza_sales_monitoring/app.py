import os

import streamlit as st
from ultralytics import YOLO

# Import configurations
from config.camera_config import REAL_CAMERA_CONFIGS
from config.model_config import DEFAULT_MODEL, MODEL_CONFIGS
from config.streamlit_config import configure_streamlit
from utils.feedback_handler import FeedbackHandler
from utils.ui_components import create_download_report, create_processing_settings, display_camera_info, display_model_info, display_results_with_camera_info

# Import utilities
from utils.video_processor import process_video_with_real_config, save_uploaded_video

# Configure Streamlit for large uploads
configure_streamlit()

st.title("🍕 Pizza Sales Tracking - Robust Zone Detection")
st.markdown("**Multi-camera system with 6 stores - Probation-based tracking + User Feedback**")

# Initialize feedback handler
if "feedback_handler" not in st.session_state:
    st.session_state.feedback_handler = FeedbackHandler()

# Sidebar - Model Configuration
st.sidebar.header("🤖 Model Configuration")
selected_model = st.sidebar.selectbox("Select Model Version", options=list(MODEL_CONFIGS.keys()), index=list(MODEL_CONFIGS.keys()).index(DEFAULT_MODEL))

# Display model information
model_info = MODEL_CONFIGS[selected_model]
display_model_info(model_info, selected_model)


# Load model with caching
@st.cache_resource
def load_model(model_path):
    """Load YOLO model with caching for performance"""
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        st.error(f"Model not found: {model_path}")
        return None


model = load_model(model_info["path"])

if model:
    st.sidebar.success(f"✅ Model loaded: {selected_model}")

    # Camera selection
    st.sidebar.header("📹 Camera Selection")
    selected_camera = st.sidebar.selectbox("Select Camera Store", options=list(REAL_CAMERA_CONFIGS.keys()), index=0, help="Choose which store camera to process")

    # Display camera info
    camera_info = REAL_CAMERA_CONFIGS[selected_camera]
    display_camera_info(camera_info, selected_camera)

    # User Feedback Interface
    feedback_type, user_comment = st.session_state.feedback_handler.create_feedback_interface()

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📹 Video Upload & Processing")

        # Enhanced video upload
        video_file = st.file_uploader(f"Upload Video for {selected_camera}", type=["mp4", "avi", "mov", "mkv"], help=f"Upload video file for {selected_camera} analysis (Large files supported in Docker)")

        if video_file:
            video_path = save_uploaded_video(video_file)
            st.success(f"✅ Video uploaded for {selected_camera}: {video_file.name}")

            # Display file info
            file_size_mb = len(video_file.getvalue()) / (1024 * 1024)
            st.info(f"📊 File size: {file_size_mb:.1f} MB")

            # Video preview
            st.video(video_file)

    with col2:
        # Enhanced processing settings
        processing_mode, max_frames, start_frame, confidence_threshold, probation_frames, dispatch_timer = create_processing_settings()

        # Process button
        if st.button("🚀 Start Robust Processing", type="primary"):
            if video_file:
                # Store current timestamp in session state for feedback
                st.session_state.current_timestamp = 0.0

                if processing_mode == "Entire Video":
                    import cv2

                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()

                    max_frames_to_process = total_frames
                    st.info(f"🎬 Processing ENTIRE video: {total_frames:,} frames")
                else:
                    # Partial Video - dùng max_frames từ slider
                    max_frames_to_process = max_frames
                    st.info(f"🎬 Processing PARTIAL video: {max_frames:,} frames")

                final_sales, processor = process_video_with_real_config(video_path, video_file.name, selected_camera, selected_model, model, max_frames_to_process, start_frame, confidence_threshold, probation_frames, dispatch_timer)

                if final_sales is not None:
                    # Handle feedback if provided
                    if feedback_type and user_comment:
                        feedback_id, feedback_file = st.session_state.feedback_handler.save_feedback(feedback_type, user_comment, selected_camera, st.session_state.get("current_timestamp", 0.0))
                        st.sidebar.success(f"✅ Feedback saved! ID: {feedback_id}")
                        st.sidebar.info("Your feedback will help improve the model in future versions.")

                    # Display comprehensive results
                    display_results_with_camera_info(final_sales, selected_camera, selected_model, max_frames or "All", start_frame, confidence_threshold, probation_frames, dispatch_timer, processor)

                    # Download report
                    create_download_report(final_sales, selected_camera, selected_model, video_file.name)
            else:
                st.error("⚠️ Please upload a video file first!")

    st.session_state.feedback_handler.display_feedback_summary()

else:
    st.error("❌ Cannot load model. Please check model configuration.")
    st.info("📁 Ensure model files exist in the models/ directory")

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666;'>
    <p>🍕 Pizza Sales Tracking System - Multi-Camera Robust Detection + User Feedback</p>
    <p>6 Store Cameras: CAM01-CAM06 • Real Zone Configurations • Probation-based Algorithm • User Feedback System</p>
</div>
""",
    unsafe_allow_html=True,
)
