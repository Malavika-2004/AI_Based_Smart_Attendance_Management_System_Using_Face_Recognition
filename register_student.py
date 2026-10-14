# Corrected registration script: embeddings are generated directly from
# the detected webcam face, avoiding re-detection of tight cropped images.
import os, cv2, sqlite3, numpy as np, time
from insightface.app import FaceAnalysis

DATASET_PATH = r"C:\Users\malav\OneDrive\Desktop\dataset"
DB_PATH = r"C:\Users\malav\OneDrive\Desktop\attendance.db"
EMBEDDINGS_FILE = os.path.join(DATASET_PATH, "arcface_embeddings.npy")
NAMES_FILE = os.path.join(DATASET_PATH, "arcface_names.npy")
NUM_IMAGES = 15
CAPTURE_DELAY = 0.35

def setup_database():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, student_id TEXT,
        department TEXT, year TEXT, section TEXT, contact TEXT)""")
    cur.execute("PRAGMA table_info(students)")
    cols = {r[1] for r in cur.fetchall()}
    for col, typ in {"student_id":"TEXT","department":"TEXT","year":"TEXT","section":"TEXT","contact":"TEXT"}.items():
        if col not in cols:
            cur.execute(f"ALTER TABLE students ADD COLUMN {col} {typ}")
    conn.commit()
    return conn

def get_student_details(conn):
    print("\n" + "="*60 + "\n        STUDENT REGISTRATION\n" + "="*60)
    student = {
        "student_id": input("Student ID: ").strip(),
        "name": input("Student Name: ").strip(),
        "department": input("Department: ").strip(),
        "year": input("Year: ").strip(),
        "section": input("Section: ").strip(),
        "contact": input("Contact Number: ").strip()
    }
    if not all(student.values()):
        print("\nERROR: All fields are required.")
        return None
    cur = conn.cursor()
    cur.execute("SELECT name FROM students WHERE student_id = ?", (student["student_id"],))
    row = cur.fetchone()
    if row:
        print(f"\nERROR: Student ID '{student['student_id']}' already exists for '{row[0]}'.")
        return None
    return student

def load_face_model():
    print("\nLoading ArcFace / InsightFace model...")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("ArcFace model loaded successfully.")
    return app

def capture_student(app, student_name):
    folder = os.path.join(DATASET_PATH, student_name)
    os.makedirs(folder, exist_ok=True)
    print("\n" + "="*60 + "\nWEBCAM CAPTURE\n" + "="*60)
    print(f"Student : {student_name}\nImages  : {NUM_IMAGES}")
    print("\nLook directly at the camera. Move your head slightly between captures.")
    print("Press Q to cancel.\n")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return [], False
    embeddings, captured, last_capture = [], 0, 0
    try:
        while captured < NUM_IMAGES:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Could not read webcam frame.")
                break
            display = frame.copy()
            faces = app.get(frame)
            if faces:
                face = max(faces, key=lambda f: max(0, f.bbox[2]-f.bbox[0]) * max(0, f.bbox[3]-f.bbox[1]))
                x1,y1,x2,y2 = face.bbox.astype(int)
                h,w = frame.shape[:2]
                x1,y1 = max(0,x1), max(0,y1)
                x2,y2 = min(w,x2), min(h,y2)
                cv2.rectangle(display,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(display,"Face detected - hold still",(20,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,0),2)
                now=time.time()
                if now-last_capture >= CAPTURE_DELAY:
                    emb = face.embedding
                    if emb is not None:
                        emb=np.asarray(emb,dtype=np.float32)
                        if emb.size and np.isfinite(emb).all():
                            embeddings.append(emb)
                            bw,bh=x2-x1,y2-y1
                            px,py=int(bw*.25),int(bh*.25)
                            cx1,cy1=max(0,x1-px),max(0,y1-py)
                            cx2,cy2=min(w,x2+px),min(h,y2+py)
                            crop=frame[cy1:cy2,cx1:cx2]
                            if crop.size:
                                n=captured+1
                                cv2.imwrite(os.path.join(folder,f"{student_name}_{n:02d}.jpg"),crop)
                            captured += 1
                            last_capture=now
                            print(f"Captured {captured}/{NUM_IMAGES}")
            else:
                cv2.putText(display,"No face detected",(20,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2)
            cv2.putText(display,f"Captured: {captured}/{NUM_IMAGES}",(20,70),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2)
            cv2.imshow("Student Registration",display)
            if (cv2.waitKey(1)&0xFF)==ord('q'):
                print("\nCapture cancelled.")
                return embeddings, False
    finally:
        cap.release(); cv2.destroyAllWindows()
    if not embeddings:
        print("\nERROR: No valid face embeddings generated.")
        return embeddings, False
    print(f"\nSuccessfully generated {len(embeddings)} face embeddings.")
    return embeddings, True

def save_embeddings(new_embeddings, student_name):
    new=np.asarray(new_embeddings,dtype=np.float32)
    if new.ndim==1: new=new.reshape(1,-1)
    if os.path.exists(EMBEDDINGS_FILE):
        old=np.asarray(np.load(EMBEDDINGS_FILE,allow_pickle=True),dtype=np.float32)
        if old.size==0: old=np.empty((0,new.shape[1]),dtype=np.float32)
        if old.ndim==1: old=old.reshape(1,-1)
        if old.shape[1]!=new.shape[1]:
            print(f"\nERROR: Embedding dimensions do not match. Existing: {old.shape[1]}, New: {new.shape[1]}")
            return False
    else: old=np.empty((0,new.shape[1]),dtype=np.float32)
    names=np.asarray(np.load(NAMES_FILE,allow_pickle=True),dtype=object) if os.path.exists(NAMES_FILE) else np.array([],dtype=object)
    if any(str(n).strip()==student_name for n in names):
        print(f"\nERROR: '{student_name}' already exists in arcface_names.npy.")
        return False
    combined=np.vstack([old,new])
    combined_names=np.concatenate([names,np.array([student_name]*len(new),dtype=object)])
    if os.path.exists(EMBEDDINGS_FILE): np.save(EMBEDDINGS_FILE+".bak",old)
    if os.path.exists(NAMES_FILE): np.save(NAMES_FILE+".bak",names)
    np.save(EMBEDDINGS_FILE,combined); np.save(NAMES_FILE,combined_names)
    print("\nArcFace files updated successfully.")
    print(f"Total embeddings: {len(combined)}\nTotal names     : {len(combined_names)}")
    return True

def save_student(conn, s):
    conn.execute("INSERT INTO students (name,student_id,department,year,section,contact) VALUES (?,?,?,?,?,?)",
                 (s["name"],s["student_id"],s["department"],s["year"],s["section"],s["contact"]))
    conn.commit()
    print("\nStudent details saved to SQLite database.")

def main():
    os.makedirs(DATASET_PATH,exist_ok=True)
    conn=setup_database()
    try:
        student=get_student_details(conn)
        if student is None: return
        print("\nStudent details:")
        for k,v in student.items(): print(f"{k.title():12}: {v}")
        if input("\nContinue with face registration? (y/n): ").strip().lower()!='y':
            print("Registration cancelled."); return
        app=load_face_model()
        embeddings,ok=capture_student(app,student["name"])
        if not ok or len(embeddings)<NUM_IMAGES:
            print("\nRegistration stopped. No database changes were made."); return
        if not save_embeddings(embeddings,student["name"]):
            print("\nRegistration stopped. Student was NOT added to the database."); return
        try: save_student(conn,student)
        except sqlite3.IntegrityError as e:
            print(f"\nDatabase error: {e}"); return
        print("\n"+"="*60+"\n       REGISTRATION SUCCESSFUL\n"+"="*60)
        print(f"Student ID : {student['student_id']}\nName       : {student['name']}\nDepartment : {student['department']}\nYear       : {student['year']}\nSection    : {student['section']}\nEmbeddings : {len(embeddings)}")
        print("="*60)
        print("\nNext step: run the webcam recognition program and test this student.")
    finally: conn.close()

if __name__=="__main__": main()
