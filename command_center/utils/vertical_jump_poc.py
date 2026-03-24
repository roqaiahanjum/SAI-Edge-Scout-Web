import cv2
import math
import numpy as np
import sys
import os

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    print("Error: mediapipe is not installed correctly or missing 'tasks' API.")
    sys.exit(1)

# ==========================================
# 1. Mathematical Logic Engine (Easy to port)
# ==========================================
class JumpCalculator:
    def __init__(self, physical_height_cm: float):
        self.physical_height_cm = physical_height_cm
        self.pixel_to_cm_ratio = None
        
        self.baseline_y = None
        self.max_jump_cm = 0.0
        self.current_jump_cm = 0.0
        
    def calibrate(self, nose_y: float, heel_y: float):
        body_length_pixels = abs(heel_y - nose_y)
        if body_length_pixels > 0:
            self.pixel_to_cm_ratio = self.physical_height_cm / body_length_pixels
            print(f"[Engine] Calibrated: 1 pixel = {self.pixel_to_cm_ratio:.4f} cm")
        else:
            raise ValueError("Calibration failed: Distance from nose to heel is zero.")

    def process_frame(self, current_hip_y: float) -> float:
        if self.pixel_to_cm_ratio is None: return 0.0
        if self.baseline_y is None:
            self.baseline_y = current_hip_y
            print(f"[Engine] Baseline locked at Y: {self.baseline_y:.1f}px")
            
        pixel_displacement = self.baseline_y - current_hip_y
        
        if pixel_displacement > 0:
            self.current_jump_cm = pixel_displacement * self.pixel_to_cm_ratio
            if self.current_jump_cm > self.max_jump_cm:
                self.max_jump_cm = self.current_jump_cm
        else:
            self.current_jump_cm = 0.0
            
        return self.current_jump_cm


# ==========================================
# 2. Video Processing & UI Rendering Engine 
# ==========================================
class VideoProcessor:
    def __init__(self, input_video_path: str, output_video_path: str, athlete_height_cm: float):
        self.input_path = input_video_path
        self.output_path = output_video_path
        self.calculator = JumpCalculator(physical_height_cm=athlete_height_cm)
        
    def draw_skeleton(self, frame, landmarks, width, height):
        # Standard MediaPipe full body connections mapping
        connections = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), 
                       (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), 
                       (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23), 
                       (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), 
                       (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)]
        
        points = {}
        for idx, lm in enumerate(landmarks):
            # Only draw landmarks that the AI is confident about
            if lm.visibility > 0.5:
                # Convert normalized coordinates (0.0 - 1.0) back to actual pixels
                px_x = int(lm.x * width)
                px_y = int(lm.y * height)
                points[idx] = (px_x, px_y)
                # Draw the joint node
                cv2.circle(frame, (px_x, px_y), 4, (0, 255, 0), -1)
                
        # Draw the bones connecting the joints
        for p1, p2 in connections:
            if p1 in points and p2 in points:
                cv2.line(frame, points[p1], points[p2], (255, 255, 255), 2)
                
    def run(self):
        if not os.path.exists(self.input_path):
            print(f"Error: The file '{self.input_path}' does not exist.")
            sys.exit(1)

        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            print(f"Error: Could not open {self.input_path}")
            sys.exit(1)
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        print("\n[Video] Loading MediaPipe Tasks AI Engine...")
        
        # Load the modern MediaPipe Tasks Vision model we just downloaded
        base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False)
        
        try:
            detector = vision.PoseLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Failed to load pose_landmarker.task: {e}")
            print("Downloading the tiny model weights for Windows compatibility...")
            sys.exit(1)
            
        print("[Video] Starting processing loop...")
        
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Use the new Image object required by the Tasks API
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)
            
            annotated_frame = frame.copy()
            
            if detection_result.pose_landmarks:
                # Get the first person's skeleton mapping
                landmarks = detection_result.pose_landmarks[0]
                
                # Indexes for MediaPipe Pose
                # 0: nose, 23: left hip, 24: right hip, 29: left heel, 30: right heel
                nose = landmarks[0]
                left_hip = landmarks[23]
                right_hip = landmarks[24]
                left_heel = landmarks[29]
                right_heel = landmarks[30]
                
                nose_y_px = nose.y * height
                mid_heel_y_px = ((left_heel.y + right_heel.y) / 2) * height
                mid_hip_y_px = ((left_hip.y + right_hip.y) / 2) * height
                
                if frame_idx == 1:
                    self.calculator.calibrate(nose_y=nose_y_px, heel_y=mid_heel_y_px)
                    
                current_jump = self.calculator.process_frame(mid_hip_y_px)
                
                # Draw full skeleton overlay mapping
                self.draw_skeleton(annotated_frame, landmarks, width, height)

                # Explicitly draw the Center of Mass tracking node (Blue)
                hip_center_x = int((left_hip.x + right_hip.x) / 2 * width)
                hip_center_y = int(mid_hip_y_px)
                cv2.circle(annotated_frame, (hip_center_x, hip_center_y), 8, (255, 0, 0), -1)
                
                # Drawing baseline

                baseline_px = int(self.calculator.baseline_y) if self.calculator.baseline_y else int(mid_hip_y_px)
                cv2.line(annotated_frame, (0, baseline_px), (width, baseline_px), (0, 0, 255), 2)
                cv2.putText(annotated_frame, "STARTING GROUND", (10, baseline_px - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                # Draw UI Text Readouts
                cv2.putText(annotated_frame, f"Real-Time Jump: {current_jump:.1f} cm", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
                cv2.putText(annotated_frame, f"MAX JUMP (Engine Result): {self.calculator.max_jump_cm:.1f} cm", (20, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)
                            
            out_writer.write(annotated_frame)
            
        cap.release()
        out_writer.release()
        cv2.destroyAllWindows()
        print(f"[Video] Processing Complete. Saved to: {self.output_path}")
        print(f"[Result] Absolute Maximum Vertical Jump: {self.calculator.max_jump_cm:.1f} cm")


# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == "__main__":
    
    INPUT_VIDEO = r"C:\Users\Roqaiah Anjum E\Downloads\demo.mp4"
    OUTPUT_VIDEO = "tracked_jump_result.mp4"
    
    REAL_WORLD_HEIGHT_CM = 180.0
    print(f"--- ENGINE VERIFICATION TEST STARTING ---")
    print(f"Simulating physical ground truth measurement process on {INPUT_VIDEO}...")
    processor = VideoProcessor(
        input_video_path=INPUT_VIDEO,
        output_video_path=OUTPUT_VIDEO,
        athlete_height_cm=REAL_WORLD_HEIGHT_CM
    )
    processor.run()
