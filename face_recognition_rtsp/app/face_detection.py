from deepface import DeepFace
import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        self.detector = DeepFace.build_model("SFace")
        self.detector_name = "SFace"

    def detect_and_recognize_faces(self, frame):
        try:
            faces = DeepFace.find(img_path=frame, db_path="database", model_name=self.detector_name, detector_backend="mtcnn", enforce_detection=False)
            if len(faces) > 0 and len(faces[0]) > 0:
                for index, instance in faces[0].iterrows():
                    if 'facial_area' in instance:  # ตรวจสอบว่ามี facial_area
                        x, y, w, h = int(instance['facial_area']['x']), int(instance['facial_area']['y']), int(instance['facial_area']['w']), int(instance['facial_area']['h'])
                        identity = instance['identity'].split('\\')[-1].split('.')[0]  # ดึงชื่อบุคคล
                        confidence = instance['distance']
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{identity} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        print("No facial_area found in DeepFace results.")
            else:
                print("No faces found by DeepFace.")
        except ValueError as e:
            print(f"ValueError during face detection: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        return frame