import cv2
import math
import numpy as np
import sys
import os
import json
import uuid
from datetime import datetime

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    print("Error: mediapipe is not installed correctly or missing 'tasks' API.")
    sys.exit(1)

# ==========================================
# 1. Automated Math Engine (3D & 2D)
# ==========================================
class ScoutEngine:
    def __init__(self):
        self.pixel_to_cm_ratio = None
        self.estimated_height_cm = 0.0
        
        # Vertical Jump State
        self.baseline_y_px = None
        self.max_vertical_cm = 0.0
        
        # Broad Jump State (Tracking trailing heel X)
        self.baseline_x_px = None
        self.max_broad_cm = 0.0
        
        # Jitter Detection Window
        self.hip_y_history = []
        
    def calibrate_from_3d_world(self, world_landmarks, pixel_landmarks, img_height, img_width):
        """
        The "Magic" Feature: Calculates the athlete's true physical height 
        using MediaPipe's intrinsic 3D World Landmarks (which are in real-world meters).
        """
        if self.pixel_to_cm_ratio is not None:
            return  # Already calibrated

        # 1. Estimate 3D Height in Meters (Nose to Mid-Heel)
        nose_3d = world_landmarks[0]
        left_heel_3d = world_landmarks[29]
        right_heel_3d = world_landmarks[30]
        
        mid_heel_x = (left_heel_3d.x + right_heel_3d.x) / 2
        mid_heel_y = (left_heel_3d.y + right_heel_3d.y) / 2
        mid_heel_z = (left_heel_3d.z + right_heel_3d.z) / 2
        
        # Euclidean distance in 3D space
        dist_meters = math.sqrt(
            (nose_3d.x - mid_heel_x)**2 + 
            (nose_3d.y - mid_heel_y)**2 + 
            (nose_3d.z - mid_heel_z)**2
        )
        
        # Add ~12cm for the top of the cranium above the nose
        self.estimated_height_cm = (dist_meters * 100) + 12.0
        
        # 2. Map this physical height to the current 2D pixels taking up the screen
        nose_2d = pixel_landmarks[0]
        left_heel_2d = pixel_landmarks[29]
        right_heel_2d = pixel_landmarks[30]
        
        nose_y_px = nose_2d.y * img_height
        mid_heel_y_px = ((left_heel_2d.y + right_heel_2d.y) / 2) * img_height
        
        body_length_pixels = abs(mid_heel_y_px - nose_y_px)
        
        if body_length_pixels > 0:
            self.pixel_to_cm_ratio = self.estimated_height_cm / body_length_pixels
            print(f"[AI Scout] Auto-Calibrated Height: {self.estimated_height_cm:.1f} cm")
            print(f"[AI Scout] Pixel Ratio: 1 px = {self.pixel_to_cm_ratio:.4f} cm")

    def analyze_frame(self, current_hip_y_px, current_heel_x_px):
        if self.pixel_to_cm_ratio is None: return 0.0, 0.0, False
        
        is_camera_shaking = False
        
        # 1. Camera Shake Detection (Variance of the last 10 hip positions)
        self.hip_y_history.append(current_hip_y_px)
        if len(self.hip_y_history) > 10:
            self.hip_y_history.pop(0)
            
        if len(self.hip_y_history) == 10:
            variance = np.var(self.hip_y_history)
            if variance > 100.0 and self.baseline_y_px is not None:
                # If variance is high but they haven't clearly moved far from baseline, the scout's hand is shaking
                if abs(current_hip_y_px - self.baseline_y_px) < 50:
                    is_camera_shaking = True

        # 2. Vertical Jump Logic (Tracking upward Y displacement)
        if self.baseline_y_px is None:
            self.baseline_y_px = current_hip_y_px
            
        vert_displacement_px = self.baseline_y_px - current_hip_y_px
        if vert_displacement_px > 0:
            current_vert_cm = vert_displacement_px * self.pixel_to_cm_ratio
            if current_vert_cm > self.max_vertical_cm:
                self.max_vertical_cm = current_vert_cm
                
        # 3. Broad Jump Logic (Tracking forward X displacement)
        if self.baseline_x_px is None:
            self.baseline_x_px = current_heel_x_px
            
        # Broad jump could be left-to-right or right-to-left. Use absolute displacement.
        broad_displacement_px = abs(current_heel_x_px - self.baseline_x_px)
        
        if broad_displacement_px > 0:
            current_broad_cm = broad_displacement_px * self.pixel_to_cm_ratio
            if current_broad_cm > self.max_broad_cm:
                self.max_broad_cm = current_broad_cm

        return current_vert_cm if vert_displacement_px > 0 else 0.0, current_broad_cm if broad_displacement_px > 0 else 0.0, is_camera_shaking


# ==========================================
# 2. Rendering & Persistence Engine
# ==========================================
class AIScoutApp:
    def __init__(self, input_video_path: str, output_video_path: str):
        self.input_path = input_video_path
        self.output_path = output_video_path
        self.engine = ScoutEngine()
        
    def generate_report(self):
        report = {
            "athlete_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "estimated_height_cm": round(self.engine.estimated_height_cm, 1),
                "max_vertical_jump_cm": round(self.engine.max_vertical_cm, 1),
                "max_broad_jump_cm": round(self.engine.max_broad_cm, 1)
            }
        }
        
        with open('talent_report.json', 'w') as f:
            json.dump(report, f, indent=4)
        print("\n[Database] talent_report.json generated successfully!")

    def draw_dashboard(self, frame, realtime_v, realtime_b, shake_warning):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (450, 250), (0, 0, 0), -1)
        # Apply semi-transparent dashboard
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Colors: Saffron / Navy (from PRD)
        cv2.putText(frame, "SAI EDGE SCOUT HUD", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
        cv2.putText(frame, f"Est. Height: {self.engine.estimated_height_cm:.1f} cm", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(frame, f"VERT JUMP: {self.engine.max_vertical_cm:.1f} cm", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"BROAD JUMP: {self.engine.max_broad_cm:.1f} cm", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        
        if shake_warning:
            cv2.rectangle(frame, (20, 210), (430, 240), (0, 0, 255), -1)
            cv2.putText(frame, "HOLD CAMERA STEADY", (60, 233), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

    def draw_skeleton(self, frame, landmarks, width, height):
        connections = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), 
                       (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), 
                       (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23), 
                       (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), 
                       (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)]
        
        points = {}
        for idx, lm in enumerate(landmarks):
            if lm.visibility > 0.5:
                px_x, px_y = int(lm.x * width), int(lm.y * height)
                points[idx] = (px_x, px_y)
                cv2.circle(frame, (px_x, px_y), 4, (0, 255, 255), -1)
                
        for p1, p2 in connections:
            if p1 in points and p2 in points:
                cv2.line(frame, points[p1], points[p2], (255, 255, 255), 2)

    def run(self):
        if not os.path.exists(self.input_path):
            print(f"Error: The video '{self.input_path}' does not exist.")
            sys.exit(1)

        cap = cv2.VideoCapture(self.input_path)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out_writer = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        print("\n[Booting] Core Engine Initializing...")
        base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False)
        detector = vision.PoseLandmarker.create_from_options(options)
            
        print("[Tracking] Engine active. Analyzing movement physics...")
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = detector.detect(mp_image)
            
            annotated_frame = frame.copy()
            
            # If AI detects an athlete
            if result.pose_landmarks and result.pose_world_landmarks:
                pixel_lm = result.pose_landmarks[0]
                world_lm = result.pose_world_landmarks[0]
                
                # Auto-Calibrate immediately on the first clear frame
                if frame_idx <= 5: 
                    self.engine.calibrate_from_3d_world(world_lm, pixel_lm, height, width)
                
                # Fetch essential tracking nodes
                left_hip, right_hip = pixel_lm[23], pixel_lm[24]
                left_heel, right_heel = pixel_lm[29], pixel_lm[30]
                
                mid_hip_y_px = ((left_hip.y + right_hip.y) / 2) * height
                # Trailing heel logic for Broad Jump (whichever heel is further back horizontally)
                # Assuming athlete jumps from left to right or right to left across screen
                # For safety, we just track the average of the heels in X
                mid_heel_x_px = ((left_heel.x + right_heel.x) / 2) * width
                
                # Process physics
                v_jump, b_jump, is_shaking = self.engine.analyze_frame(mid_hip_y_px, mid_heel_x_px)
                
                # Draw Visuals
                self.draw_skeleton(annotated_frame, pixel_lm, width, height)
                self.draw_dashboard(annotated_frame, v_jump, b_jump, is_shaking)
                
            out_writer.write(annotated_frame)
            
        cap.release()
        out_writer.release()
        cv2.destroyAllWindows()
        self.generate_report()


if __name__ == "__main__":
    INPUT_VIDEO = r"C:\Users\Roqaiah Anjum E\Downloads\demo.mp4"
    OUTPUT_VIDEO = "scouted_athlete_result.mp4"
    
    app = AIScoutApp(INPUT_VIDEO, OUTPUT_VIDEO)
    app.run()
