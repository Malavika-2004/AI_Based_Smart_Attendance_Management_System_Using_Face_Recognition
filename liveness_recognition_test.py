import cv2
import time
import os
import numpy as np
from insightface.app import FaceAnalysis

print("==========================================")
print(" LIVENESS + ARCFACE RECOGNITION TEST")
print("==========================================")

# =========================================================
# PATHS
# =========================================================

DATASET_PATH = r"C:\Users\malav\OneDrive\Desktop\dataset"

EMBEDDINGS_FILE = os.path.join(
    DATASET_PATH,
    "arcface_embeddings.npy"
)

NAMES_FILE = os.path.join(
    DATASET_PATH,
    "arcface_names.npy"
)

# =========================================================
# LOAD ARCFACE EMBEDDINGS
# =========================================================

print("\nLoading ArcFace embeddings...")

try:
    stored_embeddings = np.load(
        EMBEDDINGS_FILE
    )

    stored_names = np.load(
        NAMES_FILE,
        allow_pickle=True
    )

except Exception as e:

    print("ERROR: Could not load embeddings.")
    print(e)
    exit()

print(
    "Embeddings loaded:",
    stored_embeddings.shape
)

print(
    "Students:",
    len(np.unique(stored_names))
)

# =========================================================
# LOAD INSIGHTFACE
# =========================================================

print("\nLoading InsightFace...")

app = FaceAnalysis(
    name="buffalo_l"
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("InsightFace loaded.")

# =========================================================
# LOAD OPENCV FACE DETECTOR
# =========================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():

    print("ERROR: Could not load Haar detector.")
    exit()

# =========================================================
# OPEN WEBCAM
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Webcam could not be opened.")
    exit()

print("\nWebcam opened.")

# =========================================================
# LIVENESS VARIABLES
# =========================================================

left_seen = False
right_seen = False

liveness_passed = False

start_time = time.time()

MOVEMENT_THRESHOLD = 30

# =========================================================
# RECOGNITION THRESHOLD
# =========================================================

THRESHOLD = 0.48

# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Could not read webcam.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # =====================================================
    # LIVENESS DETECTION
    # =====================================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    status = "Move LEFT and RIGHT"

    if len(faces) > 0:

        # Select largest face
        face = max(
            faces,
            key=lambda f: f[2] * f[3]
        )

        x, y, w, h = face

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Face center
        center_x = x + w // 2
        center_y = y + h // 2

        # Draw center point
        cv2.circle(
            frame,
            (center_x, center_y),
            6,
            (255, 0, 0),
            -1
        )

        # Screen center
        screen_center = frame.shape[1] // 2

        # Draw center line
        cv2.line(
            frame,
            (screen_center, 0),
            (screen_center, frame.shape[0]),
            (255, 255, 0),
            1
        )

        # ---------------------------------------------
        # DETECT LEFT MOVEMENT
        # ---------------------------------------------

        if center_x < screen_center - MOVEMENT_THRESHOLD:

            left_seen = True

        # ---------------------------------------------
        # DETECT RIGHT MOVEMENT
        # ---------------------------------------------

        elif center_x > screen_center + MOVEMENT_THRESHOLD:

            right_seen = True

        # ---------------------------------------------
        # CHECK LIVENESS
        # ---------------------------------------------

        if left_seen and right_seen:

            liveness_passed = True

            status = "LIVENESS PASSED"

        elif left_seen:

            status = "LEFT DETECTED - MOVE RIGHT"

        elif right_seen:

            status = "RIGHT DETECTED - MOVE LEFT"

        else:

            status = "MOVE LEFT AND RIGHT"

    else:

        status = "NO FACE DETECTED"

    # =====================================================
    # AFTER LIVENESS PASSED
    # =====================================================

    if liveness_passed:

        status = "LIVENESS PASSED - RECOGNIZING..."

        # =================================================
        # INSIGHTFACE RECOGNITION
        # =================================================

        insight_faces = app.get(frame)

        if len(insight_faces) > 0:

            # Select largest face
            recognized_face = max(
                insight_faces,
                key=lambda f: (
                    f.bbox[2] - f.bbox[0]
                ) * (
                    f.bbox[3] - f.bbox[1]
                )
            )

            # Get embedding
            query_embedding = recognized_face.embedding

            # Normalize query embedding
            query_embedding = (
                query_embedding
                / np.linalg.norm(query_embedding)
            )

            # Normalize stored embeddings
            normalized_embeddings = (
                stored_embeddings
                / np.linalg.norm(
                    stored_embeddings,
                    axis=1,
                    keepdims=True
                )
            )

            # =================================================
            # COSINE SIMILARITY
            # =================================================

            scores = np.dot(
                normalized_embeddings,
                query_embedding
            )

            best_index = np.argmax(scores)

            best_score = scores[best_index]

            best_name = stored_names[best_index]

            # =================================================
            # RECOGNITION RESULT
            # =================================================

            if best_score >= THRESHOLD:

                name = str(best_name)

                recognition_text = (
                    f"{name} ({best_score:.2f})"
                )

                print(
                    f"\nRecognized: {name}"
                )

                print(
                    f"Similarity: {best_score:.2f}"
                )

                status = recognition_text

            else:

                status = (
                    f"Unknown ({best_score:.2f})"
                )

        else:

            status = "LIVENESS PASSED - NO FACE"

    # =====================================================
    # DISPLAY STATUS
    # =====================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (700, 60),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        status,
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )

    # =====================================================
    # DISPLAY INSTRUCTIONS
    # =====================================================

    if not liveness_passed:

        cv2.putText(
            frame,
            "Move your head LEFT then RIGHT",
            (25, frame.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "Liveness verified - Face recognition active",
            (25, frame.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    # =====================================================
    # SHOW CAMERA
    # =====================================================

    cv2.imshow(
        "Liveness + ArcFace Recognition",
        frame
    )

    # =====================================================
    # SUCCESS
    # =====================================================

    if liveness_passed and len(insight_faces) > 0:

        # Give recognition time to display
        if 'best_score' in locals():

            if best_score >= THRESHOLD:

                cv2.waitKey(3000)

                print()
                print("==========================================")
                print(" LIVENESS + RECOGNITION TEST SUCCESSFUL")
                print("==========================================")
                print(
                    f"Student: {best_name}"
                )
                print(
                    f"Similarity: {best_score:.2f}"
                )
                print()

                break

    # =====================================================
    # TIMEOUT
    # =====================================================

    if time.time() - start_time > 30:

        print()
        print("==========================================")
        print(" LIVENESS TEST TIMED OUT")
        print("==========================================")

        break

    # =====================================================
    # QUIT
    # =====================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        print("\nTest stopped by user.")
        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()

print("\nTest finished.")