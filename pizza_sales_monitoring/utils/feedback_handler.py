import json
import os
import uuid
from datetime import datetime

import cv2
import streamlit as st


class FeedbackHandler:
    def __init__(self):
        self.feedback_dir = "feedback_data"
        os.makedirs(self.feedback_dir, exist_ok=True)
        os.makedirs(f"{self.feedback_dir}/images", exist_ok=True)
        os.makedirs(f"{self.feedback_dir}/annotations", exist_ok=True)

    def create_feedback_interface(self):
        """Create feedback interface in sidebar"""
        st.sidebar.markdown("---")
        st.sidebar.header("🔄 User Feedback System")

        # Feedback type selection
        feedback_type = st.sidebar.selectbox("Issue Type", ["false_positive", "missed_detection", "wrong_count", "zone_issue"], format_func=lambda x: {"false_positive": "❌ False Detection", "missed_detection": "⚠️ Missed Pizza", "wrong_count": "🔢 Wrong Count", "zone_issue": "🎯 Zone Problem"}[x])

        # User comment
        user_comment = st.sidebar.text_area("Describe the issue:", placeholder="E.g., Pizza was counted twice at 2:30, or missed pizza at bottom-right corner...", height=100, help="Provide detailed feedback to help improve the model")

        # Current timestamp for feedback
        if "current_timestamp" in st.session_state:
            st.sidebar.info(f"⏰ Current time: {st.session_state.current_timestamp:.1f}s")

        # Submit feedback button
        if st.sidebar.button("📸 Submit Feedback", type="secondary"):
            if user_comment.strip():
                return feedback_type, user_comment.strip()
            else:
                st.sidebar.error("Please describe the issue!")

        return None, None

    def save_feedback(self, feedback_type, user_comment, camera_id, timestamp=None, frame_data=None):
        """Save user feedback to JSON"""
        feedback_id = str(uuid.uuid4())[:8]

        feedback_data = {"feedback_id": feedback_id, "timestamp": datetime.now().isoformat(), "camera_id": camera_id, "video_timestamp": timestamp, "feedback_type": feedback_type, "user_comment": user_comment, "status": "pending", "frame_info": frame_data}

        feedback_file = f"{self.feedback_dir}/annotations/{feedback_id}.json"
        with open(feedback_file, "w") as f:
            json.dump(feedback_data, f, indent=2)

        return feedback_id, feedback_file

    def display_feedback_summary(self):
        """Display feedback summary"""
        st.subheader("📋 User Feedback Management")

        feedback_dir = f"{self.feedback_dir}/annotations"
        if os.path.exists(feedback_dir):
            feedback_files = [f for f in os.listdir(feedback_dir) if f.endswith(".json")]

            if feedback_files:
                st.info(f"📊 Total feedback collected: {len(feedback_files)}")

                # Show recent feedback
                with st.expander("📝 Recent Feedback"):
                    for feedback_file in sorted(feedback_files)[-5:]:  # Show last 5
                        with open(os.path.join(feedback_dir, feedback_file)) as f:
                            feedback = json.load(f)

                        st.markdown(f"""
                        **ID:** {feedback["feedback_id"]} | **Type:** {feedback["feedback_type"]}  
                        **Camera:** {feedback["camera_id"]} | **Time:** {feedback.get("video_timestamp", "N/A")}s  
                        **Comment:** {feedback["user_comment"]}
                        """)

                # Export feedback button
                if st.button("📥 Export Feedback for Model Improvement"):
                    self.export_feedback_for_training()
                    st.success("✅ Feedback exported to feedback_export.json")
            else:
                st.info("No feedback collected yet. Users can submit feedback during video processing.")

    def export_feedback_for_training(self):
        """Export feedback data for model retraining"""
        feedback_dir = f"{self.feedback_dir}/annotations"
        feedback_files = [f for f in os.listdir(feedback_dir) if f.endswith(".json")]

        all_feedback = []
        for feedback_file in feedback_files:
            with open(os.path.join(feedback_dir, feedback_file)) as f:
                feedback = json.load(f)
                all_feedback.append(feedback)

        # Export for retraining
        with open("feedback_export.json", "w") as f:
            json.dump(all_feedback, f, indent=2)

        return len(all_feedback)
