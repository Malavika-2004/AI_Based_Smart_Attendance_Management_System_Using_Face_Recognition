import os
import cv2
import csv
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from insightface.app import FaceAnalysis

# =========================================================
# SETTINGS
# =========================================================

DATASET_PATH = r"C:\Users\malav\OneDrive\Desktop\dataset"
DB_PATH = r"C:\Users\malav\OneDrive\Desktop\attendance.db"

THRESHOLD = 0.48

ATTENDANCE_FILE = os.path.join(DATASET_PATH, "attendance.csv")

# =========================================================
# CLASS TIMINGS
# =========================================================
# Change these if your real timetable is different.

CLASS_HOURS = {
    "1st Hour": {"start": "12:35", "end": "13:25"},
    "2nd Hour": {"start": "13:25", "end": "14:15"},
    "3rd Hour": {"start": "14:15", "end": "15:05"},
    "4th Hour": {"start": "15:20", "end": "16:05"},
}

# =========================================================
# LIVENESS SETTINGS
# =========================================================

MOVEMENT_THRESHOLD = 30

left_seen = False
right_seen = False
liveness_passed = False

# =========================================================
# LOAD ARCFACE EMBEDDINGS
# =========================================================

print()
print("==========================================")
print("Loading ArcFace embeddings...")
print("==========================================")

try:
    embeddings = np.load(
        os.path.join(DATASET_PATH, "arcface_embeddings.npy")
    )

    names = np.load(
        os.path.join(DATASET_PATH, "arcface_names.npy"),
        allow_pickle=True
    )

except Exception as e:
    print("ERROR: Could not load ArcFace files.")
    print(e)
    raise SystemExit

print("Embeddings loaded:", embeddings.shape)
print("Students:", len(set(names)))

# =========================================================
# GROUP REFERENCES BY STUDENT
# =========================================================

student_embeddings = {}

for embedding, name in zip(embeddings, names):
    name = str(name)

    if name not in student_embeddings:
        student_embeddings[name] = []

    student_embeddings[name].append(embedding)

# =========================================================
# DATABASE SETUP
# =========================================================

def create_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Keep the existing students table. This CREATE statement only
    # applies when the table does not already exist.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    """)

    # Create the attendance table with the project fields if needed.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            student_id TEXT,
            date TEXT NOT NULL,
            hour TEXT NOT NULL,
            time TEXT,
            status TEXT NOT NULL,
            confidence REAL,
            session TEXT,
            UNIQUE(student_name, date, hour)
        )
    """)

    # Upgrade an older attendance table if any of the new columns
    # are missing.
    cursor.execute("PRAGMA table_info(attendance)")
    attendance_columns = [row[1] for row in cursor.fetchall()]

    for column, datatype in [
        ("student_id", "TEXT"),
        ("confidence", "REAL"),
        ("session", "TEXT")
    ]:
        if column not in attendance_columns:
            cursor.execute(
                f"ALTER TABLE attendance ADD COLUMN {column} {datatype}"
            )

    # Do not replace existing student records. Add ArcFace names
    # only when a matching name is not already present.
    cursor.execute("SELECT name FROM students")
    existing_names = {str(row[0]) for row in cursor.fetchall()}

    for student_name in student_embeddings.keys():
        if student_name not in existing_names:
            cursor.execute(
                "INSERT INTO students (name) VALUES (?)",
                (student_name,)
            )

    conn.commit()
    conn.close()

    print("SQLite database ready:", DB_PATH)
    print("Students in database:", len(student_embeddings))


def db_attendance_exists(name, date, hour):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT 1
        FROM attendance
        WHERE student_name = ?
          AND date = ?
          AND hour = ?
        LIMIT 1
    """, (name, date, hour))

    result = cursor.fetchone()
    conn.close()

    return result is not None


def get_student_id(name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT student_id FROM students WHERE name = ? LIMIT 1",
        (name,)
    )
    row = cursor.fetchone()
    conn.close()

    if row and row[0] is not None:
        return str(row[0])

    return None


def db_attendance_exists(name, date, hour):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT 1
        FROM attendance
        WHERE student_name = ?
          AND date = ?
          AND hour = ?
        LIMIT 1
    """, (name, date, hour))

    result = cursor.fetchone()
    conn.close()

    return result is not None


def save_attendance_to_db(
    name,
    date,
    hour,
    record_time,
    status,
    confidence=None
):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    student_id = get_student_id(name)

    # session stores the class/session name required by the project.
    cursor.execute("""
        INSERT OR IGNORE INTO attendance
        (student_name, student_id, date, hour, time,
         status, confidence, session)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        name,
        student_id,
        date,
        hour,
        record_time,
        status,
        float(confidence) if confidence is not None else None,
        hour
    ))

    inserted = cursor.rowcount > 0

    conn.commit()
    conn.close()

    return inserted


def get_database_attendance():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            student_id,
            student_name,
            date,
            hour,
            time,
            status,
            confidence,
            session
        FROM attendance
        ORDER BY date, hour, student_name
    """)

    rows = cursor.fetchall()
    conn.close()

    return rows


create_database()

# =========================================================
# LOAD INSIGHTFACE
# =========================================================

print()
print("==========================================")
print("Loading InsightFace...")
print("==========================================")

try:
    app = FaceAnalysis(name="buffalo_l")

    app.prepare(
        ctx_id=-1,
        det_size=(640, 640)
    )

except Exception as e:
    print("ERROR: Could not load InsightFace.")
    print(e)
    raise SystemExit

print("InsightFace loaded.")

# =========================================================
# LOAD OPENCV LIVENESS DETECTOR
# =========================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("ERROR: Could not load OpenCV face detector.")
    raise SystemExit

print("Liveness detector loaded.")

# =========================================================
# CREATE CSV BACKUP FILE
# =========================================================

if not os.path.exists(ATTENDANCE_FILE):
    with open(
        ATTENDANCE_FILE,
        "w",
        newline=""
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "Name",
            "Date",
            "Hour",
            "Time",
            "Status"
        ])

    print("attendance.csv created.")

# =========================================================
# CURRENT CLASS HOUR
# =========================================================

def get_current_hour():
    now = datetime.now().time()

    for hour_name, class_hour in CLASS_HOURS.items():
        start = datetime.strptime(
            class_hour["start"],
            "%H:%M"
        ).time()

        end = datetime.strptime(
            class_hour["end"],
            "%H:%M"
        ).time()

        if start <= now < end:
            return {
                "hour": hour_name,
                "start": start,
                "end": end
            }

    return None

# =========================================================
# CSV DUPLICATE CHECK
# =========================================================

def csv_attendance_exists(name, date, hour):
    if not os.path.exists(ATTENDANCE_FILE):
        return False

    with open(
        ATTENDANCE_FILE,
        "r",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if (
                row.get("Name") == name
                and row.get("Date") == date
                and row.get("Hour") == hour
            ):
                return True

    return False


def already_marked(name, date, hour):
    # Database is the primary duplicate check.
    if db_attendance_exists(name, date, hour):
        return True

    # CSV is also checked because it is being retained as a backup.
    return csv_attendance_exists(name, date, hour)

# =========================================================
# SAVE TO CSV BACKUP
# =========================================================

def save_attendance_to_csv(
    name,
    date,
    hour,
    record_time,
    status
):
    # Do not create a second CSV record.
    if csv_attendance_exists(name, date, hour):
        return

    with open(
        ATTENDANCE_FILE,
        "a",
        newline=""
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            name,
            date,
            hour,
            record_time,
            status
        ])

# =========================================================
# MARK PRESENT OR LATE
# =========================================================

def mark_attendance(name, class_hour, confidence=None):

    now = datetime.now()

    today = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    hour_name = class_hour["hour"]

    # -----------------------------------------------------
    # PREVENT DUPLICATE
    # -----------------------------------------------------

    if already_marked(
        name,
        today,
        hour_name
    ):
        print(
            f"{name} already marked for {hour_name}."
        )
        return False

    # -----------------------------------------------------
    # CALCULATE PRESENT DEADLINE
    # -----------------------------------------------------

    start_datetime = datetime.combine(
        now.date(),
        class_hour["start"]
    )

    present_deadline = (
        start_datetime +
        timedelta(minutes=15)
    )

    # -----------------------------------------------------
    # DETERMINE STATUS
    # -----------------------------------------------------

    if now <= present_deadline:
        status = "Present"
    else:
        status = "Late"

    # -----------------------------------------------------
    # SAVE TO DATABASE
    # -----------------------------------------------------

    inserted = save_attendance_to_db(
        name,
        today,
        hour_name,
        current_time,
        status,
        confidence
    )

    if not inserted:
        print(
            f"{name} already exists in database for {hour_name}."
        )
        return False

    # -----------------------------------------------------
    # SAVE CSV BACKUP
    # -----------------------------------------------------

    save_attendance_to_csv(
        name,
        today,
        hour_name,
        current_time,
        status
    )

    # -----------------------------------------------------
    # PRINT RESULT
    # -----------------------------------------------------

    print()
    print("==========================================")
    print("          ATTENDANCE MARKED")
    print("==========================================")
    print("Name:", name)
    print("Student ID:", get_student_id(name))
    print("Hour:", hour_name)
    print("Time:", current_time)
    print("Status:", status)
    if confidence is not None:
        print("Confidence:", f"{confidence:.2f}")
    print("Saved to:", DB_PATH)
    print("CSV backup: attendance.csv")
    print("==========================================")
    print()

    return True

# =========================================================
# MARK ABSENT STUDENTS
# =========================================================

def mark_absent_students(class_hour):

    now = datetime.now()

    today = now.strftime("%Y-%m-%d")
    hour_name = class_hour["hour"]

    print()
    print("==========================================")
    print("Class hour finished:", hour_name)
    print("Checking absent students...")
    print("==========================================")

    absent_count = 0

    for student_name in student_embeddings.keys():

        # Already Present/Late/Absent
        if already_marked(
            student_name,
            today,
            hour_name
        ):
            continue

        # Save Absent in database
        inserted = save_attendance_to_db(
            student_name,
            today,
            hour_name,
            "",
            "Absent"
        )

        if not inserted:
            continue

        # Save CSV backup
        save_attendance_to_csv(
            student_name,
            today,
            hour_name,
            "",
            "Absent"
        )

        print("Absent:", student_name)
        absent_count += 1

    print("Absent students added:", absent_count)

# =========================================================
# ARCFACE RECOGNITION
# =========================================================

def recognize_embedding(test_embedding):

    best_name = "Unknown"
    best_score = -1

    for (
        student_name,
        reference_embeddings
    ) in student_embeddings.items():

        scores = []

        for reference_embedding in reference_embeddings:

            denominator = (
                np.linalg.norm(test_embedding)
                *
                np.linalg.norm(reference_embedding)
            )

            if denominator == 0:
                continue

            score = np.dot(
                test_embedding,
                reference_embedding
            ) / denominator

            scores.append(score)

        if len(scores) == 0:
            continue

        student_score = max(scores)

        if student_score > best_score:
            best_score = student_score
            best_name = student_name

    if best_score >= THRESHOLD:
        return best_name, best_score

    return "Unknown", best_score

# =========================================================
# FINISHED HOURS
# =========================================================

finished_hours = set()

# =========================================================
# OPEN WEBCAM
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit

print()
print("==========================================")
print("       AI ATTENDANCE SYSTEM")
print("==========================================")
print()
print("Webcam started.")
print()
print("Step 1: Move your head LEFT.")
print("Step 2: Move your head RIGHT.")
print("Step 3: Liveness will pass.")
print("Step 4: ArcFace will recognize you.")
print("Step 5: Attendance will be recorded.")
print()
print("Press Q to quit.")
print()

# =========================================================
# MAIN LOOP
# =========================================================

while True:

    # =====================================================
    # CURRENT CLASS HOUR
    # =====================================================

    current_hour = get_current_hour()

    # =====================================================
    # CHECK FINISHED HOURS
    # =====================================================

    now = datetime.now().time()

    for hour_name, class_hour_data in CLASS_HOURS.items():

        end_time = datetime.strptime(
            class_hour_data["end"],
            "%H:%M"
        ).time()

        if (
            now >= end_time
            and hour_name not in finished_hours
        ):

            finished_class_hour = {
                "hour": hour_name,
                "start": datetime.strptime(
                    class_hour_data["start"],
                    "%H:%M"
                ).time(),
                "end": end_time
            }

            mark_absent_students(
                finished_class_hour
            )

            finished_hours.add(hour_name)

    # =====================================================
    # READ WEBCAM
    # =====================================================

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # =====================================================
    # CLEAN FRAME FOR ARCFACE
    # =====================================================

    clean_frame = frame.copy()

    # =====================================================
    # LIVENESS DETECTION
    # =====================================================

    gray = cv2.cvtColor(
        clean_frame,
        cv2.COLOR_BGR2GRAY
    )

    liveness_faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    liveness_status = "MOVE LEFT AND RIGHT"

    # =====================================================
    # FACE DETECTED
    # =====================================================

    if len(liveness_faces) > 0:

        # Select largest face
        live_face = max(
            liveness_faces,
            key=lambda f: f[2] * f[3]
        )

        x, y, w, h = live_face

        # Face center
        center_x = x + w // 2
        center_y = y + h // 2

        # Screen center
        screen_center = (
            clean_frame.shape[1] // 2
        )

        # Draw liveness box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Draw face center
        cv2.circle(
            frame,
            (center_x, center_y),
            6,
            (255, 0, 0),
            -1
        )

        # Draw screen center
        cv2.line(
            frame,
            (screen_center, 0),
            (
                screen_center,
                frame.shape[0]
            ),
            (255, 255, 0),
            1
        )

        # -------------------------------------------------
        # LEFT MOVEMENT
        # -------------------------------------------------

        if center_x < screen_center - MOVEMENT_THRESHOLD:
            left_seen = True

        # -------------------------------------------------
        # RIGHT MOVEMENT
        # -------------------------------------------------

        elif center_x > screen_center + MOVEMENT_THRESHOLD:
            right_seen = True

        # -------------------------------------------------
        # CHECK LIVENESS
        # -------------------------------------------------

        if left_seen and right_seen:
            liveness_passed = True
            liveness_status = "LIVENESS PASSED"

        elif left_seen:
            liveness_status = "LEFT DETECTED - MOVE RIGHT"

        elif right_seen:
            liveness_status = "RIGHT DETECTED - MOVE LEFT"

        else:
            liveness_status = "MOVE LEFT AND RIGHT"

    else:
        liveness_status = "NO FACE DETECTED"

    # =====================================================
    # ARCFACE RECOGNITION
    # ONLY AFTER LIVENESS
    # =====================================================

    recognized_faces = []

    if liveness_passed:

        # Use CLEAN FRAME.
        # Do not use frame containing drawings.
        recognized_faces = app.get(clean_frame)

        print(
            "ArcFace faces detected:",
            len(recognized_faces)
        )

        # =================================================
        # RECOGNIZE EACH FACE
        # =================================================

        for face in recognized_faces:

            # Face box
            x1, y1, x2, y2 = (
                face.bbox.astype(int)
            )

            # Get embedding
            embedding = face.embedding

            # Recognize
            name, score = recognize_embedding(
                embedding
            )

            print(
                f"Recognition: {name} "
                f"| Score: {score:.2f}"
            )

            # =================================================
            # ATTENDANCE
            # =================================================

            attendance_marked = False

            if (
                name != "Unknown"
                and current_hour is not None
            ):

                attendance_marked = mark_attendance(
                    name,
                    current_hour,
                    score
                )

            # =================================================
            # DRAW ARCFACE BOX
            # =================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Display name + score
            recognition_text = (
                f"{name} ({score:.2f})"
            )

            cv2.putText(
                frame,
                recognition_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # =================================================
            # DISPLAY ATTENDANCE STATUS
            # =================================================

            if attendance_marked:

                cv2.putText(
                    frame,
                    "ATTENDANCE MARKED",
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2
                )

            elif name == "Unknown":

                cv2.putText(
                    frame,
                    "UNKNOWN STUDENT",
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2
                )

            elif current_hour is None:

                cv2.putText(
                    frame,
                    "NO ACTIVE CLASS",
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "ALREADY MARKED",
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 0),
                    2
                )

    # =====================================================
    # DISPLAY CURRENT HOUR
    # =====================================================

    if current_hour is not None:
        hour_text = current_hour["hour"]
    else:
        hour_text = "No active class"

    cv2.putText(
        frame,
        hour_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # =====================================================
    # DISPLAY LIVENESS STATUS
    # =====================================================

    cv2.putText(
        frame,
        liveness_status,
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # =====================================================
    # DISPLAY INSTRUCTION
    # =====================================================

    if not liveness_passed:
        instruction = (
            "Liveness: Move LEFT then RIGHT"
        )
    else:
        instruction = (
            "Liveness verified - Recognition active"
        )

    cv2.putText(
        frame,
        instruction,
        (20, frame.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # =====================================================
    # SHOW WEBCAM
    # =====================================================

    cv2.imshow(
        "AI Attendance - Liveness + ArcFace",
        frame
    )

    # =====================================================
    # QUIT
    # =====================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# =========================================================
# CLEANUP
# =========================================================

cap.release()
cv2.destroyAllWindows()

print()
print("==========================================")
print("WEBCAM STOPPED")
print("==========================================")

# =========================================================
# DATABASE SUMMARY
# =========================================================

try:
    rows = get_database_attendance()

    print()
    print("Database attendance records:", len(rows))

    if rows:
        print("------------------------------------------")

        for row in rows[-10:]:
            print(row)

        print("------------------------------------------")

except Exception as e:
    print("Could not read database summary:", e)
