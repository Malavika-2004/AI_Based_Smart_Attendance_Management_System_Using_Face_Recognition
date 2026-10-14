import sqlite3
from pathlib import Path
from collections import Counter, defaultdict
import streamlit as st

# ============================================================
# CONFIGURATION
# ============================================================
DB_PATH = r"C:\Users\malav\OneDrive\Desktop\attendance.db"
VALID_HOURS = ["1st Hour", "2nd Hour", "3rd Hour", "4th Hour"]

st.set_page_config(
    page_title="AI-Based Smart Attendance Management System",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# DATABASE
# ============================================================
def load_data():
    """Read the existing database without modifying it."""
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        conn.row_factory = sqlite3.Row

        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        if "attendance" not in tables:
            raise RuntimeError("The database does not contain an 'attendance' table.")
        if "students" not in tables:
            raise RuntimeError("The database does not contain a 'students' table.")

        attendance_rows = conn.execute("""
            SELECT student_id, student_name, date, hour, time,
                   status, confidence, session
            FROM attendance
            ORDER BY date DESC, hour, student_name
        """).fetchall()

        student_rows = conn.execute("""
            SELECT name, student_id, department, year, section, contact
            FROM students
            ORDER BY name
        """).fetchall()

    attendance = [dict(row) for row in attendance_rows]
    students = [dict(row) for row in student_rows]

    # Exclude temporary/test hours; only use the four real timetable hours.
    attendance = [
        row for row in attendance
        if str(row.get("hour") or "").strip() in VALID_HOURS
    ]

    for row in attendance:
        row["student_name"] = str(row.get("student_name") or "Unknown").strip()
        row["student_id"] = str(row.get("student_id") or "").strip()
        row["date"] = str(row.get("date") or "").strip()
        row["hour"] = str(row.get("hour") or "").strip()
        row["time"] = str(row.get("time") or "").strip()
        row["status"] = str(row.get("status") or "").strip().title()

    # Fill a missing attendance ID only if that registered name has one unique ID.
    name_to_ids = defaultdict(set)
    for student in students:
        name = str(student.get("name") or "").strip().casefold()
        student_id = str(student.get("student_id") or "").strip()
        if name and student_id:
            name_to_ids[name].add(student_id)

    for row in attendance:
        if not row["student_id"]:
            possible_ids = name_to_ids.get(row["student_name"].casefold(), set())
            row["student_id"] = (
                next(iter(possible_ids))
                if len(possible_ids) == 1
                else "Not assigned"
            )

    for row in students:
        for key in ["name", "student_id", "department", "year", "section", "contact"]:
            row[key] = str(row.get(key) or "")

    return attendance, students


# ============================================================
# LOAD AND DISPLAY ERRORS CLEARLY
# ============================================================
st.title("AI-Based Smart Attendance Management System")
st.caption("Face Recognition • Liveness Detection • SQLite • Streamlit")

st.info("Dashboard started. Loading attendance data...")

try:
    attendance, students = load_data()
except Exception as exc:
    st.error("The dashboard could not load its data.")
    st.exception(exc)
    st.stop()

st.success(
    f"Database loaded successfully: {len(students)} registered students, "
    f"{len(attendance)} attendance records for regular class hours."
)

# ============================================================
# FILTERS
# ============================================================
st.sidebar.header("Attendance Filters")

dates = sorted(
    {row["date"] for row in attendance if row["date"]},
    reverse=True
)
available_hours = [
    hour for hour in VALID_HOURS
    if any(row["hour"] == hour for row in attendance)
]

selected_date = st.sidebar.selectbox("Date", ["All"] + dates)
selected_hour = st.sidebar.selectbox("Class Hour", ["All"] + available_hours)

filtered = attendance.copy()

if selected_date != "All":
    filtered = [row for row in filtered if row["date"] == selected_date]

if selected_hour != "All":
    filtered = [row for row in filtered if row["hour"] == selected_hour]

# ============================================================
# KPI CARDS
# ============================================================
status_counts = Counter(
    row["status"] if row["status"] else "Unknown"
    for row in filtered
)

present_count = status_counts.get("Present", 0)
late_count = status_counts.get("Late", 0)
absent_count = status_counts.get("Absent", 0)
known_status_count = present_count + late_count + absent_count

attendance_percentage = (
    (present_count + late_count) / known_status_count * 100
    if known_status_count else 0
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Registered Students", len(students))
col2.metric("Present", present_count)
col3.metric("Late", late_count)
col4.metric("Absent", absent_count)
col5.metric("Attendance %", f"{attendance_percentage:.1f}%")

st.divider()

# ============================================================
# ATTENDANCE STATUS CHART
# ============================================================
st.subheader("Attendance Status")

if status_counts:
    status_table = [
        {"Status": status, "Count": count}
        for status, count in sorted(status_counts.items())
    ]
    st.bar_chart(status_table, x="Status", y="Count")
    st.dataframe(status_table, use_container_width=True, hide_index=True)
else:
    st.info("No attendance records match the selected filters.")

# ============================================================
# DAILY ATTENDANCE TREND
# ============================================================
st.subheader("Daily Attendance Trend")

daily_counts = defaultdict(lambda: {"Present": 0, "Total": 0})

for row in attendance:
    if row["date"] and row["status"] in {"Present", "Late", "Absent"}:
        daily_counts[row["date"]]["Total"] += 1
        if row["status"] in {"Present", "Late"}:
            daily_counts[row["date"]]["Present"] += 1

daily_table = []
for date, values in sorted(daily_counts.items()):
    percentage = (
        values["Present"] / values["Total"] * 100
        if values["Total"] else 0
    )
    daily_table.append({
        "Date": date,
        "Attendance %": round(percentage, 2),
        "Present or Late": values["Present"],
        "Total Records": values["Total"]
    })

if daily_table:
    st.line_chart(daily_table, x="Date", y="Attendance %")
    st.dataframe(daily_table, use_container_width=True, hide_index=True)
else:
    st.info("No dated attendance records available.")


# ============================================================
# STUDENT-WISE ATTENDANCE
# ============================================================


st.subheader("Student-Wise Attendance")

if attendance and students:

    # Use registered student IDs as the source of truth.
    registered_id_lookup = {}
    ambiguous_names = set()

    for student in students:
        name_key = str(student.get("name") or "").strip().casefold()
        student_id = str(student.get("student_id") or "").strip()

        if not name_key or not student_id:
            continue

        if (
            name_key in registered_id_lookup
            and registered_id_lookup[name_key] != student_id
        ):
            ambiguous_names.add(name_key)
        else:
            registered_id_lookup[name_key] = student_id

    # Do not guess an ID when multiple registered students share a name.
    for name_key in ambiguous_names:
        registered_id_lookup.pop(name_key, None)

    # Group attendance records using the registered ID when available.
    student_records = {}

    for row in attendance:
        student_name = str(row.get("student_name") or "Unknown").strip()
        name_key = student_name.casefold()

        registered_id = registered_id_lookup.get(name_key, "")
        existing_id = str(row.get("student_id") or "").strip()

        display_id = registered_id or existing_id or "Not assigned"
        key = (display_id, student_name)

        if key not in student_records:
            student_records[key] = []

        student_records[key].append(row)

    # Build the student-wise summary.
    summary = []

    for (student_id, student_name), records in student_records.items():
        present_count = 0
        late_count = 0
        absent_count = 0

        for record in records:
            status = str(record.get("status") or "").strip().casefold()

            if status == "present":
                present_count += 1
            elif status == "late":
                late_count += 1
            elif status == "absent":
                absent_count += 1

        total_sessions = len(records)
        attendance_percentage = (
            (present_count + late_count) / total_sessions * 100
            if total_sessions > 0
            else 0
        )

        summary.append({
            "Student ID": student_id,
            "Student Name": student_name,
            "Total Sessions": total_sessions,
            "Present": present_count,
            "Late": late_count,
            "Absent": absent_count,
            "Attendance %": round(attendance_percentage, 2),
        })

    summary.sort(
        key=lambda item: (
            item["Attendance %"],
            item["Student Name"].casefold()
        )
    )

    st.dataframe(
        summary,
        width="stretch",
        hide_index=True
    )

else:
    st.info("No attendance records or registered students were found.")
    summary = []


# ============================================================
# LOW ATTENDANCE ALERTS
# ============================================================
st.subheader("Low Attendance Students (Below 75%)")

low_attendance = [
    row for row in summary
    if row["Attendance %"] < 75
]

if low_attendance:
    st.warning(f"{len(low_attendance)} student(s) are below 75% attendance.")
    st.dataframe(low_attendance, use_container_width=True, hide_index=True)
else:
    st.success("No students are currently below 75% attendance, or no summary data exists.")

# ============================================================
# REGISTERED STUDENTS
# ============================================================
st.subheader("Registered Students")

if students:
    st.dataframe(students, use_container_width=True, hide_index=True)
else:
    st.info("No registered students were found.")

# ============================================================
# DETAILED ATTENDANCE RECORDS
# ============================================================
st.subheader("Detailed Attendance Records")

display_rows = [
    {
        "Student ID": row["student_id"],
        "Student Name": row["student_name"],
        "Date": row["date"],
        "Hour": row["hour"],
        "Time": row["time"],
        "Status": row["status"],
        "Confidence": row.get("confidence"),
        "Session": row.get("session")
    }
    for row in filtered
]

if display_rows:
    st.dataframe(display_rows, use_container_width=True, hide_index=True)

    headers = list(display_rows[0].keys())

    def csv_escape(value):
        if value is None:
            value = ""
        return '"' + str(value).replace('"', '""') + '"'

    csv_content = ",".join(csv_escape(header) for header in headers) + "\n"
    csv_content += "\n".join(
        ",".join(csv_escape(row.get(header, "")) for header in headers)
        for row in display_rows
    )

    st.download_button(
        label="Download Attendance CSV",
        data=csv_content.encode("utf-8-sig"),
        file_name="attendance_report.csv",
        mime="text/csv"
    )
else:
    st.info("No detailed records match the selected filters.")

# ============================================================
# ATTENDANCE NOTIFICATIONS
# ============================================================
st.divider()
st.header("Attendance Notifications")

if not attendance:
    st.info("No regular-hour attendance records are available.")
else:
    target_date = (
        selected_date
        if selected_date != "All"
        else (dates[0] if dates else None)
    )

    date_records = [
        row for row in attendance
        if target_date is not None and row["date"] == target_date
    ]

    if selected_hour != "All":
        target_hour = selected_hour
    else:
        target_hour = next(
            (
                hour for hour in reversed(VALID_HOURS)
                if any(row["hour"] == hour for row in date_records)
            ),
            None
        )

    session_records = [
        row for row in date_records
        if target_hour is None or row["hour"] == target_hour
    ]

    st.caption(
        f"Notifications for {target_date or 'no date selected'}"
        + (f" | {target_hour}" if target_hour else "")
    )

    present_names = sorted({
        row["student_name"]
        for row in session_records
        if row["status"] == "Present"
    })
    late_names = sorted({
        row["student_name"]
        for row in session_records
        if row["status"] == "Late"
    })
    absent_rows = [
        row for row in session_records
        if row["status"] == "Absent"
    ]

    if present_names:
        st.success("Present: " + ", ".join(present_names))

    if late_names:
        st.warning("Late: " + ", ".join(late_names))

    if absent_rows:
        st.warning(
            f"{len(absent_rows)} record(s) are marked Absent "
            "for this session."
        )
        st.dataframe(
            [
                {
                    "Student ID": row["student_id"],
                    "Student Name": row["student_name"],
                    "Status": row["status"]
                }
                for row in absent_rows
            ],
            use_container_width=True,
            hide_index=True
        )

    if not session_records:
        st.info("No records found for this session.")

    st.subheader("Low Attendance Alerts")
    if low_attendance:
        st.dataframe(
            [
                {
                    "Student ID": row["Student ID"],
                    "Student Name": row["Student Name"],
                    "Present": row["Present"],
                    "Late": row["Late"],
                    "Total Sessions": row["Total Sessions"],
                    "Attendance %": row["Attendance %"]
                }
                for row in low_attendance
            ],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("No low-attendance alerts to display.")

# ============================================================
# REFRESH
# ============================================================
if st.button("Refresh Dashboard"):
    st.rerun()

st.divider()
st.caption("AI-Based Smart Attendance Management System")
