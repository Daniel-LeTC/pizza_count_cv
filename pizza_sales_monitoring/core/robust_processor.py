import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


class RobustZoneProcessor:
    def __init__(self, camera_configs, model, model_version="stable"):
        """
        Initialize RobustZoneProcessor

        Args:
            camera_configs: Dictionary of camera configurations
            model: YOLO model instance (already loaded)
            model_version: "stable" or "v2_stable"
        """
        self.camera_configs = camera_configs
        self.model = model
        self.model_version = model_version

        # Probation System - Core innovation
        self.potential_pizzas = {}  # {grid_key: {frames_seen: int, first_seen: ts, bbox: bbox}}
        self.min_frames_for_confirmation = 30  # Must be seen for 30 frames (~1.25s) to be confirmed

        # Core System
        self.grid_size = 120  # Spatial grid for tracking
        self.spatial_memory = {}  # Confirmed pizzas only
        self.pizza_states = {}
        self.sales_counts = {cam: {"total_sales": 0, "current_in_zone": 0, "pending_dispatch": 0} for cam in camera_configs}

        # Timing
        self.dispatch_threshold = 90  # 90 seconds dispatch timer
        self.spatial_id_counter = 1
        self.frame_count = 0
        self.fps = 30

        print(f"🍕 RobustZoneProcessor initialized with model: {model_version}")
        print(f"📊 Probation period: {self.min_frames_for_confirmation} frames")
        print(f"⏱️  Dispatch threshold: {self.dispatch_threshold} seconds")

    def get_grid_key(self, center_x, center_y):
        """Convert coordinates to grid key for spatial tracking"""
        return f"{int(center_x // self.grid_size)}_{int(center_y // self.grid_size)}"

    def is_in_zone(self, center_point, zone_polygon):
        """Check if point is inside the staging zone"""
        return cv2.pointPolygonTest(zone_polygon, center_point, False) >= 0

    def robust_zone_tracking(self, detections, zone_polygon, current_timestamp):
        """
        Core robust tracking with probation system
        Only pizzas that survive probation period become confirmed
        """
        # Cleanup expired entries
        self._cleanup_expired_entries()

        if not hasattr(detections, "xyxy") or len(detections.xyxy) == 0:
            return [], []  # confirmed, potential

        bboxes = detections.xyxy
        confidences = detections.confidence

        confirmed_detections = []
        potential_detections = []
        unmatched_detections = list(range(len(bboxes)))

        # Step 1: Match to existing CONFIRMED pizzas
        for grid_key, memory in list(self.spatial_memory.items()):
            best_match_idx = -1
            best_dist = float("inf")

            for i in unmatched_detections:
                x1, y1, x2, y2 = bboxes[i]
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

                if not self.is_in_zone((center_x, center_y), zone_polygon):
                    continue

                dist = np.sqrt((center_x - memory["center"][0]) ** 2 + (center_y - memory["center"][1]) ** 2)

                if dist < 150 and dist < best_dist:
                    best_dist = dist
                    best_match_idx = i

            if best_match_idx != -1:
                # Update confirmed pizza
                x1, y1, x2, y2 = bboxes[best_match_idx]
                new_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                new_grid_key = self.get_grid_key(new_center[0], new_center[1])

                # Move memory to new grid if needed
                if new_grid_key != grid_key:
                    del self.spatial_memory[grid_key]

                memory["bbox"] = bboxes[best_match_idx]
                memory["center"] = new_center
                memory["last_seen"] = self.frame_count
                self.spatial_memory[new_grid_key] = memory

                confirmed_detections.append(memory)
                unmatched_detections.remove(best_match_idx)

        # Step 2: Process remaining detections as POTENTIAL
        for i in unmatched_detections:
            bbox = bboxes[i]
            conf = confidences[i]

            if conf < 0.2:  # Stricter confidence for new objects
                continue

            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

            if not self.is_in_zone(center, zone_polygon):
                continue

            grid_key = self.get_grid_key(center[0], center[1])

            # Check if matches existing potential
            matched_potential = False
            for old_grid, pot in list(self.potential_pizzas.items()):
                dist = np.sqrt((center[0] - pot["center"][0]) ** 2 + (center[1] - pot["center"][1]) ** 2)

                if dist < 100:  # Match potential
                    pot["frames_seen"] += 1
                    pot["center"] = center
                    pot["last_seen_frame"] = self.frame_count

                    if pot["frames_seen"] >= self.min_frames_for_confirmation:
                        # GRADUATION! Promote to confirmed
                        new_id = self.spatial_id_counter
                        self.spatial_id_counter += 1

                        confirmed_memory = {"spatial_id": new_id, "bbox": bbox, "center": center, "last_seen": self.frame_count}
                        self.spatial_memory[grid_key] = confirmed_memory
                        confirmed_detections.append(confirmed_memory)

                        print(f"✅ CONFIRMED! Pizza Spatial ID {new_id} at grid {grid_key} after {pot['frames_seen']} frames.")

                        # Remove from potential list
                        del self.potential_pizzas[old_grid]
                    else:
                        # Still on probation
                        potential_detections.append(pot)

                    matched_potential = True
                    break

            if not matched_potential:
                # First time seeing this potential pizza
                self.potential_pizzas[grid_key] = {"frames_seen": 1, "first_seen": current_timestamp, "center": center, "bbox": bbox, "last_seen_frame": self.frame_count}
                potential_detections.append(self.potential_pizzas[grid_key])

        return confirmed_detections, potential_detections

    def _cleanup_expired_entries(self):
        """Clean up old potential pizzas and confirmed memories"""
        # Clean up old potential pizzas
        expired_potentials = [k for k, v in self.potential_pizzas.items() if self.frame_count - v["last_seen_frame"] > self.min_frames_for_confirmation * 2]
        for k in expired_potentials:
            del self.potential_pizzas[k]

        # Clean up old confirmed spatial memories
        expired_confirmed = [k for k, v in self.spatial_memory.items() if self.frame_count - v["last_seen"] > 180]  # 6 seconds
        for k in expired_confirmed:
            del self.spatial_memory[k]

    def update_sales_logic(self, cam_id, confirmed_detections, current_timestamp):
        """Update sales tracking logic using only CONFIRMED detections"""
        current_confirmed_ids = set()

        # Process confirmed detections only
        for detection in confirmed_detections:
            spatial_id = detection["spatial_id"]
            current_confirmed_ids.add(spatial_id)

            # Pizza in zone
            if spatial_id not in self.pizza_states:
                self.pizza_states[spatial_id] = {"status": "in_zone", "enter_time": current_timestamp, "exit_time": None, "cam_id": cam_id}
                print(f"🍕 Pizza Spatial {spatial_id} entered staging zone at {current_timestamp:.1f}s")

            # Pizza returned to zone
            elif self.pizza_states[spatial_id]["status"] == "pending_dispatch":
                self.pizza_states[spatial_id]["status"] = "in_zone"
                self.pizza_states[spatial_id]["exit_time"] = None
                self.sales_counts[cam_id]["pending_dispatch"] -= 1
                print(f"↩️  Pizza Spatial {spatial_id} returned to zone - dispatch cancelled")

        # Check for pizzas that left the zone
        for spatial_id, state in list(self.pizza_states.items()):
            if state["status"] == "in_zone" and spatial_id not in current_confirmed_ids:
                # Pizza left zone
                self.pizza_states[spatial_id]["status"] = "pending_dispatch"
                self.pizza_states[spatial_id]["exit_time"] = current_timestamp
                self.sales_counts[cam_id]["pending_dispatch"] += 1
                print(f"🚀 Pizza Spatial {spatial_id} left zone - pending dispatch ({self.dispatch_threshold}s timer)")

        # Check for sales (dispatch timer expired)
        for spatial_id, state in list(self.pizza_states.items()):
            if state["status"] == "pending_dispatch" and current_timestamp - state["exit_time"] >= self.dispatch_threshold:
                state["status"] = "dispatched"
                self.sales_counts[cam_id]["total_sales"] += 1
                self.sales_counts[cam_id]["pending_dispatch"] -= 1
                print(f"💰 SALE! Pizza Spatial {spatial_id} dispatched. Total Sales: {self.sales_counts[cam_id]['total_sales']}")

        self.sales_counts[cam_id]["current_in_zone"] = len(current_confirmed_ids)
        return len(current_confirmed_ids), self.sales_counts[cam_id]["total_sales"], self.sales_counts[cam_id]["pending_dispatch"]

    def draw_robust_annotations(self, frame, confirmed, potentials, zone_polygon, in_zone_count, sales_count, pending_count):
        """Visualize confirmed (green) and potential (yellow) pizzas"""
        annotated_frame = frame.copy()

        # Draw zone polygon
        cv2.polylines(annotated_frame, [zone_polygon.astype(np.int32)], True, (0, 255, 0), 2)

        # Draw CONFIRMED pizzas (Green)
        for det in confirmed:
            x1, y1, x2, y2 = det["bbox"].astype(int)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(annotated_frame, f"Confirmed {det['spatial_id']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw POTENTIAL pizzas (Yellow)
        for det in potentials:
            x1, y1, x2, y2 = det["bbox"].astype(int)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"Potential {det['frames_seen']}/{self.min_frames_for_confirmation}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Status text
        cv2.putText(annotated_frame, f"SALES: {sales_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(annotated_frame, f"In Zone (Confirmed): {in_zone_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Pending Dispatch: {pending_count}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 165, 0), 2)
        cv2.putText(annotated_frame, f"Potential Pizzas: {len(potentials)}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        return annotated_frame

    def process_frame(self, cam_id, frame, current_timestamp):
        """Process single frame with robust logic"""
        self.frame_count += 1

        # YOLO detection
        results = self.model(frame, conf=0.15, verbose=False)[0]

        # Convert to simple format
        detections_data = type("obj", (object,), {"xyxy": results.boxes.xyxy.cpu().numpy() if hasattr(results, "boxes") and len(results.boxes) > 0 else np.empty((0, 4)), "confidence": results.boxes.conf.cpu().numpy() if hasattr(results, "boxes") and len(results.boxes) > 0 else np.empty(0)})()

        zone_polygon = self.camera_configs[cam_id]["zone_polygon"]

        # Robust tracking
        confirmed_detections, potential_detections = self.robust_zone_tracking(detections_data, zone_polygon, current_timestamp)

        # Sales logic (on confirmed detections only)
        pizzas_in_zone, total_sales, pending_dispatch = self.update_sales_logic(cam_id, confirmed_detections, current_timestamp)

        # Annotations
        annotated_frame = self.draw_robust_annotations(frame, confirmed_detections, potential_detections, zone_polygon, pizzas_in_zone, total_sales, pending_dispatch)

        return annotated_frame

    def process_partial_video(self, cam_id, max_frames=3000, start_frame=0):
        """Process video with robust, probation-based logic"""
        config = self.camera_configs[cam_id]

        print(f"🚀 Starting ROBUST PROBATION-BASED TRACKING for {cam_id}")
        print(f"📊 Model version: {self.model_version}")

        # Reset state
        self.potential_pizzas = {}
        self.spatial_memory = {}
        self.pizza_states = {}
        self.sales_counts[cam_id] = {"total_sales": 0, "current_in_zone": 0, "pending_dispatch": 0}
        self.frame_count = start_frame
        self.spatial_id_counter = 1

        # Open video
        cap = cv2.VideoCapture(config["source_video_path"])
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video {config['source_video_path']}")
            return None

        # Video properties
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Jump to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Video writer
        output_path = config["target_video_path"].replace(".mp4", f"_robust_{start_frame}to{start_frame + max_frames}.mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))

        frame_idx = 0
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            current_timestamp = (start_frame + frame_idx) / self.fps
            processed_frame = self.process_frame(cam_id, frame, current_timestamp)
            out.write(processed_frame)

            frame_idx += 1

            if frame_idx % 300 == 0:
                print(f"📊 Frame {start_frame + frame_idx}: Sales {self.sales_counts[cam_id]['total_sales']}, Confirmed {self.sales_counts[cam_id]['current_in_zone']}, Potentials {len(self.potential_pizzas)}")

        # Final dispatch check
        final_timestamp = (start_frame + frame_idx) / self.fps
        for spatial_id, state in list(self.pizza_states.items()):
            if state["status"] == "pending_dispatch" and final_timestamp - state["exit_time"] >= self.dispatch_threshold:
                self.sales_counts[cam_id]["total_sales"] += 1
                print(f"🎯 FINAL SALE! Pizza Spatial {spatial_id} dispatched.")

        # Cleanup
        cap.release()
        out.release()

        final_sales = self.sales_counts[cam_id]["total_sales"]
        print(f"✅ ROBUST TRACKING COMPLETED! Final Sales: {final_sales}")
        return final_sales

    def generate_sales_report(self):
        """Generate final sales report"""
        print("=" * 60)
        print("🍕 ROBUST PIZZA SALES REPORT")
        print("=" * 60)

        total_sales_all_stores = 0
        for cam_id, sales_data in self.sales_counts.items():
            total_sales = sales_data.get("total_sales", 0)
            total_sales_all_stores += total_sales
            print(f"📹 {cam_id}: Total Sales {total_sales}")

        print(f"🏆 TOTAL SALES ACROSS ALL STORES: {total_sales_all_stores}")
        print("=" * 60)
