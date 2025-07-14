import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def detect_face_shape(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Image not found"

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return "No face detected"

        landmarks = results.multi_face_landmarks[0].landmark

        # Get key landmark points by index
        jaw = np.array([(landmarks[i].x, landmarks[i].y) for i in range(0, 17)])      # Jawline
        forehead = np.array([(landmarks[i].x, landmarks[i].y) for i in range(10, 11)]) # Approximate top
        cheeks = np.array([(landmarks[234].x, landmarks[454].x)])  # Sides of face

        # Width (jawline) and height (top to chin)
        face_width = np.linalg.norm(jaw[0] - jaw[-1])
        face_height = np.linalg.norm(np.array([landmarks[10].x, landmarks[10].y]) - np.array([landmarks[152].x, landmarks[152].y]))

        # Ratio for shape decision
        ratio = face_width / face_height

        if ratio > 1.5:
            return "Round"
        elif ratio > 1.3:
            return "Square"
        elif ratio > 1.1:
            return "Oval"
        else:
            return "Heart"
