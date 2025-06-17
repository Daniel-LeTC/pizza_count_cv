import streamlit as st


def display_model_info(model_info, selected_model):
    """Display model information in sidebar"""
    st.sidebar.info(f"""
    **Model: {selected_model}**
    - **Precision:** {model_info["precision"]}
    - **Recall:** {model_info["recall"]}  
    - **mAP50:** {model_info["map50"]}
    - **mAP50-95:** {model_info["map50_95"]}

    *{model_info["description"]}*
    """)


def display_camera_info(camera_info, selected_camera):
    """Display camera information in sidebar"""
    st.sidebar.info(f"""
    **{selected_camera} Configuration:**
    - **Resolution:** {camera_info["frame_resolution_wh"]}
    - **Zone Points:** {len(camera_info["zone_polygon"])} coordinates
    - **Zone Type:** Real staging area from PolygonZone tool
    """)


def create_processing_settings():
    """Create enhanced processing settings UI"""
    st.header("⚙️ Processing Settings")

    # Processing mode selection
    processing_mode = st.selectbox("Processing Mode", ["Partial Video", "Entire Video"], help="Choose whether to process part or entire video")

    if processing_mode == "Partial Video":
        max_frames = st.slider(
            "Max Frames to Process",
            min_value=1000,
            max_value=100000,
            value=10000,
            step=1000,
            help="Maximum number of frames to process",
        )

        start_frame = st.slider(
            "Start Frame",
            min_value=0,
            max_value=50000,  # INCREASED
            value=0,
            step=100,
            help="Starting frame for processing",
        )
    else:
        # Entire video mode
        max_frames = None  # Process all frames
        start_frame = 0
        st.info("🎬 **Entire Video Mode:** All frames will be processed")

        # Show estimated processing time
        st.warning("⚠️ **Note:** Processing entire video may take significant time depending on video length")

    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.15, 0.05, help="Confidence threshold for detection")

        probation_frames = st.slider("Probation Frames", 10, 60, 30, 5, help="Number of frames required to confirm pizza")

        dispatch_timer = st.slider("Dispatch Timer (seconds)", 30, 180, 90, 10, help="Wait time for dispatch after pizza leaves zone")

    return processing_mode, max_frames, start_frame, confidence_threshold, probation_frames, dispatch_timer


def display_results_with_camera_info(final_sales, camera_id, model_name, max_frames, start_frame, conf_threshold, probation_frames, dispatch_timer, processor):
    """Display comprehensive processing results with camera information"""

    from config.camera_config import REAL_CAMERA_CONFIGS

    st.success(f"🎉 {camera_id} Robust Processing Completed Successfully!")

    # Main metrics display
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(f"🍕 {camera_id} Sales", final_sales, help=f"Confirmed sales from {camera_id} staging zone")

    with col2:
        st.metric("💰 Revenue Estimate", f"${final_sales * 15:.2f}", help="Estimated revenue (15$/pizza)")

    with col3:
        st.metric("🎯 Processing Method", "Robust Zone", help="Using probation-based tracking")

    with col4:
        st.metric("📊 Model Used", model_name, help="Model version used for detection")

    # Camera-specific information
    st.subheader(f"📹 {camera_id} Configuration Details")

    camera_config = REAL_CAMERA_CONFIGS[camera_id]
    zone_points = camera_config["zone_polygon"]

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"""
        **{camera_id} Zone Configuration:**
        - **Zone Points:** {len(zone_points)} coordinates
        - **Resolution:** {camera_config["frame_resolution_wh"]}
        - **Zone Type:** Real staging area (scaled from PolygonZone)
        - **Coordinate Range:** X: {zone_points[:, 0].min()}-{zone_points[:, 0].max()}, Y: {zone_points[:, 1].min()}-{zone_points[:, 1].max()}
        """)

    with col2:
        st.info(f"""
        **Processing Parameters:**
        - **Frames Processed:** {max_frames:,} (from frame {start_frame:,})
        - **Confidence Threshold:** {conf_threshold}
        - **Probation Period:** {probation_frames} frames
        - **Dispatch Timer:** {dispatch_timer} seconds
        """)

    # Technical details with real zone info
    with st.expander(f"🔍 {camera_id} Technical Implementation"):
        st.markdown(f"""
        ### 🧠 Robust Zone-Based Detection for {camera_id}
        
        **Real Zone Configuration:**
        - Zone coordinates extracted from PolygonZone tool
        - Scaled from 1280x720 to 1920x1080 resolution
        - Represents actual pizza staging area in {camera_id}
        
        **Probation System:**
        - Each detection must survive {probation_frames} frames to be confirmed
        - Only confirmed pizzas are counted towards sales
        - Grid-based spatial tracking with 120-pixel grid size
        
        **Sales Logic for {camera_id}:**
        1. Pizza enters real staging zone → Probation period
        2. After {probation_frames} frames → Confirmed pizza (Spatial ID assigned)
        3. Pizza leaves zone → Pending dispatch ({dispatch_timer}s timer)
        4. Timer expires → Sale confirmed for {camera_id} 💰
        
        **Zone Polygon Points:**
        ```
        {zone_points}
        ```
        """)


def create_download_report(final_sales, camera_id, model_name, video_name):
    """Create downloadable sales report"""
    report_text = f"""
    ==========================================
    🍕 ROBUST PIZZA SALES REPORT
    ==========================================
    
    📹 Camera: {camera_id}
    📊 Total Sales: {final_sales}
    💰 Revenue: ${final_sales * 15:.2f}
    
    🔧 Processing Configuration:
    - Model: {model_name}
    - Video: {video_name}
    - Processing Method: Robust Zone-Based Tracking
    
    📈 Performance Metrics:
    - Confirmation Rate: High (Probation-based)
    - Processing Method: Robust Zone-Based Tracking
    
    ==========================================
    """

    st.download_button(label="📥 Download Sales Report", data=report_text, file_name=f"pizza_sales_report_{camera_id}_{video_name.split('.')[0]}.txt", mime="text/plain")
