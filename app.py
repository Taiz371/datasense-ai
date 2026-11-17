import os
import time
from datetime import date
from io import BytesIO

import pandas as pd
import plotly.express as px
import psycopg2
import requests
import streamlit as st
from dotenv import load_dotenv
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError

from sql_engine import PostgresSQLEngine

load_dotenv()

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="DataSense AI | Business Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: Inter, sans-serif; }
    .hero {
        padding: 25px 28px; border-radius: 18px; margin-bottom: 20px;
        background: linear-gradient(120deg,#111827,#172554 65%,#312e81);
        border: 1px solid rgba(255,255,255,.12);
    }
    .hero h1 { margin: 0; color: #e0e7ff; font-size: 30px; }
    .hero p { margin: 7px 0 0; color: #cbd5e1; }
    div[data-testid="stMetric"] {
        background: rgba(100,116,139,.08); padding: 13px;
        border: 1px solid rgba(148,163,184,.18); border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
DEFAULT_STATE = {
    "query_history": [],
    "current_result": None,
    "user_question": "",
    "should_run": False,
    "last_audio": None,
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


def set_query(value: str):
    st.session_state.user_question = value
    st.session_state.should_run = True


def trigger_query():
    st.session_state.should_run = True


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ DataSense Settings")
    st.caption("Connect your database and AI providers.")

    with st.expander("PostgreSQL connection", expanded=True):
        db_user = st.text_input("Database user", os.getenv("DB_USER", "postgres"))
        db_password = st.text_input(
            "Database password", os.getenv("DB_PASSWORD", ""), type="password"
        )
        db_host = st.text_input("Host", os.getenv("DB_HOST", "localhost"))
        db_port = st.text_input("Port", os.getenv("DB_PORT", "5432"))
        db_name = st.text_input("Database", os.getenv("DB_NAME", "company_db"))

    with st.expander("AI provider keys", expanded=True):
        google_api_key = st.text_input(
            "Gemini API key", os.getenv("GOOGLE_API_KEY", ""), type="password"
        )
        groq_api_key = st.text_input(
            "Groq API key (voice/fallback)",
            os.getenv("GROQ_API_KEY", ""),
            type="password",
        )

    st.caption("Credentials are used by this app at runtime. Do not commit .env files.")

if not db_password:
    st.warning("Enter your PostgreSQL password in the sidebar to continue.")
    st.stop()

# URL.create safely handles special characters in passwords.
db_url_object = URL.create(
    "postgresql+psycopg2",
    username=db_user,
    password=db_password,
    host=db_host,
    port=int(db_port),
    database=db_name,
)
db_url = db_url_object.render_as_string(hide_password=False)


@st.cache_resource(show_spinner=False)
def get_sqlalchemy_engine(url: str):
    return create_engine(url, pool_pre_ping=True, pool_recycle=1800)


@st.cache_resource(show_spinner=False)
def get_ai_engine(url: str, gemini_key: str, groq_key: str):
    return PostgresSQLEngine(
        db_url=url,
        google_api_key=gemini_key,
        groq_api_key=groq_key,
    )


try:
    db_engine = get_sqlalchemy_engine(db_url)
    with db_engine.begin() as connection:
        connection.execute(text("SELECT 1"))
        # Auto-align serial sequences with max existing IDs on startup
        try:
            connection.execute(text("""
                SELECT setval(pg_get_serial_sequence('departments', 'dept_id'), COALESCE(max(dept_id), 1)) FROM departments;
                SELECT setval(pg_get_serial_sequence('employees', 'emp_id'), COALESCE(max(emp_id), 1)) FROM employees;
            """))
        except Exception:
            pass
    db_connected = True
except Exception as exc:
    db_connected = False
    st.error(f"PostgreSQL connection failed: {exc}")
    st.stop()

ai_engine = None
if google_api_key:
    try:
        ai_engine = get_ai_engine(db_url, google_api_key, groq_api_key)
    except Exception as exc:
        st.sidebar.error(f"AI engine could not start: {exc}")
else:
    st.sidebar.info("Add a Gemini key to enable natural-language SQL.")

# ---------------------------------------------------------
# DATABASE UTILITIES
# ---------------------------------------------------------
def read_sql(query: str, params: dict | None = None) -> pd.DataFrame:
    """Run a parameterized read query and return a DataFrame."""
    with db_engine.connect() as connection:
        return pd.read_sql_query(text(query), connection, params=params or {})


def get_public_tables() -> list[str]:
    from sqlalchemy import inspect
    return inspect(db_engine).get_table_names(schema="public")


def get_table_data(table_name: str, limit: int = 500) -> pd.DataFrame:
    """Read a selected public table; quote its identifier and bind the row limit."""
    from sqlalchemy import inspect
    allowed = set(inspect(db_engine).get_table_names(schema="public"))
    if table_name not in allowed:
        raise ValueError("That table is not available in the public schema.")
    quoted = db_engine.dialect.identifier_preparer.quote(table_name)
    return read_sql(f"SELECT * FROM public.{quoted} LIMIT :row_limit",
                    {"row_limit": int(limit)})


def transcribe_audio(audio_bytes: bytes, api_key: str) -> str:
    if not api_key:
        return "Add a Groq API key to use voice input."
    endpoint = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key.strip()}"}
    for model in ("whisper-large-v3-turbo", "whisper-large-v3"):
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                files={"file": ("recording.wav", audio_bytes, "audio/wav")},
                data={"model": model, "temperature": "0"},
                timeout=30,
            )
            if response.ok:
                return response.json().get("text", "").strip()
        except requests.RequestException as exc:
            return f"Voice request failed: {exc}"
    return f"Transcription failed: {response.status_code} {response.text[:300]}"


def export_excel(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    return output.getvalue()


def record_history(prompt: str, result: dict, duration: float):
    st.session_state.query_history.insert(
        0,
        {
            "question": prompt,
            "mode": result.get("type", "unknown"),
            "provider": result.get("provider_used", "unknown"),
            "seconds": duration,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    st.session_state.query_history = st.session_state.query_history[:100]


# ---------------------------------------------------------
# HEADER AND CONNECTION STATUS
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>⚡ DataSense AI</h1>
      <p>Autonomous PostgreSQL copilot · Dynamic analytics · Data management</p>
    </div>
    """,
    unsafe_allow_html=True,
)
status_col1, status_col2, status_col3 = st.columns(3)
status_col1.success("PostgreSQL connected")
status_col2.success("Database: " + db_name)
status_col3.success("AI ready" if ai_engine else "AI key required")

# ---------------------------------------------------------
# NAVIGATION
# ---------------------------------------------------------
tab_ai, tab_dashboard, tab_explorer, tab_employees, tab_departments, tab_history = st.tabs(
    [
        "🤖 AI Copilot",
        "📊 Dashboard",
        "🗂️ Data Explorer",
        "👥 Employee Manager",
        "🏢 Departments",
        "📜 Query History",
    ]
)

# ---------------------------------------------------------
# AI COPILOT
# ---------------------------------------------------------
with tab_ai:
    st.subheader("Ask your database in natural language")
    st.caption(
        "Examples: salary expense by department, employees hired this year, "
        "top five salaries, or explain SQL joins."
    )

    presets = [
        "Show the top 5 highest-paid employees",
        "Which department has the highest salary expense?",
        "Show average salary by department",
        "List employees hired in the last 12 months",
        "Explain INNER JOIN, LEFT JOIN and FULL OUTER JOIN",
        "Show employee count by department",
    ]
    preset_cols = st.columns(3)
    for index, preset in enumerate(presets):
        preset_cols[index % 3].button(
            preset,
            key=f"preset_{index}",
            use_container_width=True,
            on_click=set_query,
            args=(preset,),
        )

    voice_col, prompt_col = st.columns([1, 3])
    with voice_col:
        st.markdown("**🎙️ Voice query**")
        audio = st.audio_input("Record a question", label_visibility="collapsed")
        if audio is not None:
            audio_bytes = audio.getvalue()
            if audio_bytes != st.session_state.last_audio:
                st.session_state.last_audio = audio_bytes
                with st.spinner("Transcribing audio..."):
                    transcript = transcribe_audio(audio_bytes, groq_api_key)
                if transcript and not transcript.startswith(
                    ("Voice request failed:", "Transcription failed:", "Add a Groq")
                ):
                    st.session_state.user_question = transcript
                    st.rerun()
                else:
                    st.error(transcript)

    with prompt_col:
        st.text_input(
            "Your question",
            key="user_question",
            placeholder="Ask a question about your PostgreSQL database...",
            on_change=trigger_query,
        )
        run_clicked = st.button(
            "🚀 Run AI analysis",
            type="primary",
            use_container_width=True,
        )

    if run_clicked or st.session_state.should_run:
        st.session_state.should_run = False
        prompt = st.session_state.user_question.strip()
        if not prompt:
            st.warning("Enter a question first.")
        elif ai_engine is None:
            st.error("Add a valid Gemini API key to activate the AI copilot.")
        else:
            started = time.time()
            try:
                with st.spinner("Generating and executing the analysis..."):
                    result = ai_engine.execute_with_auto_repair(prompt)
                duration = round(time.time() - started, 2)
                result["duration"] = duration
                result["prompt"] = prompt
                st.session_state.current_result = result
                record_history(prompt, result, duration)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

    result = st.session_state.current_result
    if result:
        st.divider()
        st.caption(
            f"Completed in {result.get('duration', 0)}s · "
            f"Provider: {result.get('provider_used', 'Unknown')}"
        )
        if result.get("type") == "explanation":
            st.markdown(result.get("content", "No explanation returned."))
        else:
            df = result.get("df")
            if isinstance(df, pd.DataFrame):
                m1, m2 = st.columns(2)
                m1.metric("Rows returned", len(df))
                m2.metric("Columns returned", len(df.columns))
                st.dataframe(df, use_container_width=True, hide_index=True)
                download_col1, download_col2 = st.columns(2)
                download_col1.download_button(
                    "Download CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    "ai_query_result.csv",
                    "text/csv",
                    use_container_width=True,
                )
                download_col2.download_button(
                    "Download Excel",
                    export_excel(df),
                    "ai_query_result.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

                if len(df.columns) >= 2:
                    st.markdown("#### Quick visualization")
                    x_col = st.selectbox("Category / X-axis", df.columns, key="ai_x")
                    y_options = [
                        col for col in df.columns
                        if pd.api.types.is_numeric_dtype(df[col])
                    ]
                    if y_options:
                        y_col = st.selectbox("Numeric / Y-axis", y_options, key="ai_y")
                        chart_type = st.selectbox(
                            "Chart type",
                            ["Bar", "Line", "Scatter", "Pie", "Donut"],
                            key="ai_chart",
                        )
                        if chart_type == "Bar":
                            fig = px.bar(df, x=x_col, y=y_col, template="plotly_dark")
                        elif chart_type == "Line":
                            fig = px.line(
                                df, x=x_col, y=y_col, markers=True,
                                template="plotly_dark"
                            )
                        elif chart_type == "Scatter":
                            fig = px.scatter(
                                df, x=x_col, y=y_col, template="plotly_dark"
                            )
                        else:
                            fig = px.pie(
                                df, names=x_col, values=y_col,
                                hole=0.45 if chart_type == "Donut" else 0,
                                template="plotly_dark",
                            )
                        st.plotly_chart(fig, use_container_width=True)

            if result.get("final_sql"):
                with st.expander("View generated SQL and repair trace"):
                    st.code(result["final_sql"], language="sql")
                    for attempt in result.get("attempts_log", []):
                        st.write(
                            f"Attempt {attempt.get('attempt')} · "
                            f"{attempt.get('provider')}"
                        )
                        if attempt.get("error"):
                            st.error(attempt["error"])
                        st.code(attempt.get("sql", ""), language="sql")

# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------
with tab_dashboard:
    st.subheader("Company overview")
    try:
        summary = read_sql("""
            SELECT
                (SELECT COUNT(*) FROM employees) AS employees,
                (SELECT COUNT(*) FROM departments) AS departments,
                (SELECT COALESCE(SUM(salary), 0) FROM employees) AS payroll,
                (SELECT COALESCE(AVG(salary), 0) FROM employees) AS avg_salary
        """)
        values = summary.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total employees", f"{int(values['employees']):,}")
        c2.metric("Departments", f"{int(values['departments']):,}")
        c3.metric("Monthly payroll*", f"₹{float(values['payroll']):,.0f}")
        c4.metric("Average salary", f"₹{float(values['avg_salary']):,.0f}")
        st.caption("*Payroll is the sum of stored salary values; the app does not assume a pay period.")

        dept_df = read_sql("""
            SELECT d.dept_name,
                   COUNT(e.emp_id) AS employee_count,
                   COALESCE(SUM(e.salary), 0) AS salary_total,
                   COALESCE(AVG(e.salary), 0) AS average_salary
            FROM departments d
            LEFT JOIN employees e ON e.dept_id = d.dept_id
            GROUP BY d.dept_id, d.dept_name
            ORDER BY salary_total DESC
        """)
        left, right = st.columns(2)
        with left:
            st.markdown("#### Employees by department")
            st.plotly_chart(
                px.bar(
                    dept_df, x="dept_name", y="employee_count",
                    color="dept_name", template="plotly_dark"
                ),
                use_container_width=True,
            )
        with right:
            st.markdown("#### Salary distribution by department")
            if dept_df["salary_total"].sum() > 0:
                st.plotly_chart(
                    px.pie(
                        dept_df, names="dept_name", values="salary_total",
                        hole=0.45, template="plotly_dark"
                    ),
                    use_container_width=True,
                )
            else:
                st.info("No salary records are available.")

        st.markdown("#### Department performance")
        st.dataframe(dept_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download department report",
            dept_df.to_csv(index=False).encode("utf-8"),
            "department_report.csv",
            "text/csv",
        )
    except Exception as exc:
        st.warning(
            "Dashboard tables are not ready. Create the expected employees and "
            f"departments tables first. Details: {exc}"
        )

# ---------------------------------------------------------
# DYNAMIC DATA EXPLORER
# ---------------------------------------------------------
with tab_explorer:
    st.subheader("Explore database tables")
    try:
        tables = get_public_tables()
        if not tables:
            st.info("No tables found in the public schema.")
        else:
            selected_table = st.selectbox("Select a table", tables)
            row_limit = st.select_slider(
                "Maximum rows",
                options=[50, 100, 250, 500, 1000, 5000],
                value=500,
            )
            if st.button("Load table", key="load_table"):
                st.session_state["explorer_table"] = selected_table
                st.session_state["explorer_limit"] = row_limit

            if st.session_state.get("explorer_table") in tables:
                table_df = get_table_data(
                    st.session_state["explorer_table"],
                    st.session_state.get("explorer_limit", 500),
                )
                st.caption(
                    f"{len(table_df):,} rows loaded from "
                    f"{st.session_state['explorer_table']}"
                )
                search = st.text_input("Search visible rows")
                filtered_df = table_df
                if search:
                    mask = table_df.astype(str).apply(
                        lambda col: col.str.contains(
                            search, case=False, na=False, regex=False
                        )
                    ).any(axis=1)
                    filtered_df = table_df[mask]
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Export table as CSV",
                    filtered_df.to_csv(index=False).encode("utf-8"),
                    f"{selected_table}.csv",
                    "text/csv",
                )
    except Exception as exc:
        st.error(f"Could not inspect database tables: {exc}")

# ---------------------------------------------------------
# EMPLOYEE CRUD
# ---------------------------------------------------------
with tab_employees:
    st.subheader("Employee management")
    try:
        departments_df = read_sql(
            "SELECT dept_id, dept_name FROM departments ORDER BY dept_name"
        )
        employee_df = read_sql("""
            SELECT e.emp_id, e.name, e.role, e.salary, e.hire_date,
                   e.dept_id, d.dept_name
            FROM employees e
            LEFT JOIN departments d ON e.dept_id = d.dept_id
            ORDER BY e.emp_id
        """)

        search_text = st.text_input("Search by employee name or role")
        department_options = ["All departments"] + departments_df["dept_name"].tolist()
        chosen_department = st.selectbox("Filter department", department_options)
        filtered = employee_df.copy()
        if search_text:
            filtered = filtered[
                filtered["name"].str.contains(search_text, case=False, na=False)
                | filtered["role"].str.contains(search_text, case=False, na=False)
            ]
        if chosen_department != "All departments":
            filtered = filtered[filtered["dept_name"] == chosen_department]

        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.download_button(
            "Export filtered employees",
            filtered.to_csv(index=False).encode("utf-8"),
            "employees_filtered.csv",
            "text/csv",
        )

        add_tab, edit_tab, delete_tab = st.tabs(
            ["➕ Add employee", "✏️ Update employee", "🗑️ Delete employee"]
        )

        with add_tab:
            with st.form("add_employee_form", clear_on_submit=True):
                new_name = st.text_input("Full name", placeholder="Aarav Sharma")
                new_role = st.text_input("Job role", placeholder="Python Developer")
                new_salary = st.number_input(
                    "Salary", min_value=0.0, step=1000.0, value=50000.0
                )
                new_hire_date = st.date_input("Hire date", value=date.today())
                dept_map = dict(
                    zip(departments_df["dept_name"], departments_df["dept_id"])
                )
                new_department = st.selectbox(
                    "Department", list(dept_map.keys())
                ) if dept_map else None
                submitted = st.form_submit_button(
                    "Add employee", type="primary"
                )
                if submitted:
                    if not new_name.strip() or not new_role.strip() or not new_department:
                        st.error("Name, role and department are required.")
                    else:
                        with db_engine.begin() as connection:
                            # Auto-sync sequence to current max emp_id
                            try:
                                connection.execute(text("SELECT setval(pg_get_serial_sequence('employees', 'emp_id'), COALESCE(max(emp_id), 1)) FROM employees"))
                            except Exception:
                                pass
                            connection.execute(
                                text("""
                                    INSERT INTO employees
                                        (name, role, salary, hire_date, dept_id)
                                    VALUES
                                        (:name, :role, :salary, :hire_date, :dept_id)
                                """),
                                {
                                    "name": new_name.strip(),
                                    "role": new_role.strip(),
                                    "salary": new_salary,
                                    "hire_date": new_hire_date,
                                    "dept_id": dept_map[new_department],
                                },
                            )
                        st.success("Employee added.")
                        st.rerun()

        with edit_tab:
            if not employee_df.empty:
                employee_ids = employee_df["emp_id"].tolist()
                selected_emp = st.selectbox(
                    "Select employee ID to update", employee_ids
                )
                current = employee_df[
                    employee_df["emp_id"] == selected_emp
                ].iloc[0]
                with st.form("update_employee_form"):
                    edit_name = st.text_input("Name", value=current["name"])
                    edit_role = st.text_input("Role", value=current["role"])
                    edit_salary = st.number_input(
                        "Salary",
                        min_value=0.0,
                        value=float(current["salary"]),
                        step=1000.0,
                    )
                    edit_date = st.date_input(
                        "Hire date",
                        value=pd.to_datetime(current["hire_date"]).date(),
                    )
                    edit_dept_map = dict(
                        zip(departments_df["dept_name"], departments_df["dept_id"])
                    )
                    dept_names = list(edit_dept_map.keys())
                    current_dept = current["dept_name"]
                    dept_index = (
                        dept_names.index(current_dept)
                        if current_dept in dept_names else 0
                    )
                    edit_department = st.selectbox(
                        "Department", dept_names, index=dept_index
                    ) if dept_names else None
                    update_submitted = st.form_submit_button(
                        "Save changes", type="primary"
                    )
                    if update_submitted:
                        if not edit_name.strip() or not edit_role.strip() or not edit_department:
                            st.error("Name, role and department are required.")
                        else:
                            with db_engine.begin() as connection:
                                connection.execute(
                                    text("""
                                        UPDATE employees
                                        SET name=:name, role=:role, salary=:salary,
                                            hire_date=:hire_date, dept_id=:dept_id
                                        WHERE emp_id=:emp_id
                                    """),
                                    {
                                        "name": edit_name.strip(),
                                        "role": edit_role.strip(),
                                        "salary": edit_salary,
                                        "hire_date": edit_date,
                                        "dept_id": edit_dept_map[edit_department],
                                        "emp_id": int(selected_emp),
                                    },
                                )
                            st.success("Employee updated.")
                            st.rerun()
            else:
                st.info("There are no employees to update.")

        with delete_tab:
            if not employee_df.empty:
                delete_id = st.selectbox(
                    "Employee to delete",
                    employee_df["emp_id"].tolist(),
                    format_func=lambda emp_id: (
                        f"{emp_id} — "
                        f"{employee_df.loc[employee_df.emp_id == emp_id, 'name'].iloc[0]}"
                    ),
                )
                confirm_delete = st.checkbox(
                    "I understand this permanently deletes the selected employee."
                )
                if st.button(
                    "Delete employee",
                    type="primary",
                    disabled=not confirm_delete,
                ):
                    with db_engine.begin() as connection:
                        connection.execute(
                            text("DELETE FROM employees WHERE emp_id=:emp_id"),
                            {"emp_id": int(delete_id)},
                        )
                    st.success("Employee deleted.")
                    st.rerun()
            else:
                st.info("There are no employees to delete.")
    except Exception as exc:
        st.error(f"Employee manager error: {exc}")

# ---------------------------------------------------------
# DEPARTMENT MANAGEMENT (Safe & Sequence-Aligned)
# ---------------------------------------------------------
with tab_departments:
    st.subheader("Department management")
    try:
        dept_view = read_sql("""
            SELECT d.dept_id, d.dept_name, d.location,
                   COUNT(e.emp_id) AS employee_count
            FROM departments d
            LEFT JOIN employees e ON e.dept_id = d.dept_id
            GROUP BY d.dept_id, d.dept_name, d.location
            ORDER BY d.dept_id
        """)
        st.dataframe(dept_view, use_container_width=True, hide_index=True)

        with st.form("add_department_form", clear_on_submit=True):
            dept_name = st.text_input("New department name")
            dept_location = st.text_input(
                "Office location", placeholder="Kolkata"
            )
            add_dept = st.form_submit_button("Create department")
            if add_dept:
                clean_name = dept_name.strip()
                clean_loc = dept_location.strip()
                if not clean_name or not clean_loc:
                    st.error("Department name and location are required.")
                else:
                    with db_engine.begin() as connection:
                        # 1. Align the primary key sequence to avoid duplicate key errors
                        try:
                            connection.execute(text("SELECT setval(pg_get_serial_sequence('departments', 'dept_id'), COALESCE(max(dept_id), 1)) FROM departments"))
                        except Exception:
                            pass

                        # 2. Resilient insert that checks for existing department name (case-insensitive)
                        result = connection.execute(
                            text("""
                                INSERT INTO departments (dept_name, location)
                                SELECT :name, :location
                                WHERE NOT EXISTS (
                                    SELECT 1 FROM departments WHERE LOWER(dept_name) = LOWER(:name)
                                )
                            """),
                            {
                                "name": clean_name,
                                "location": clean_loc,
                            },
                        )
                    if result.rowcount > 0:
                        st.success(f"Department '{clean_name}' created successfully.")
                    else:
                        st.info(f"Department '{clean_name}' already exists.")
                    st.rerun()

        st.download_button(
            "Download departments CSV",
            dept_view.to_csv(index=False).encode("utf-8"),
            "departments.csv",
            "text/csv",
        )
    except Exception as exc:
        st.error(f"Department manager error: {exc}")

# ---------------------------------------------------------
# QUERY HISTORY
# ---------------------------------------------------------
with tab_history:
    st.subheader("Session query history")
    history = st.session_state.query_history
    if history:
        history_df = pd.DataFrame(history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download query history",
            history_df.to_csv(index=False).encode("utf-8"),
            "query_history.csv",
            "text/csv",
        )
        if st.button("Clear session history"):
            st.session_state.query_history = []
            st.rerun()
    else:
        st.info("No AI queries have been run in this session.")

st.divider()
st.caption(
    "DataSense AI · PostgreSQL + SQLAlchemy + Streamlit + Gemini + Groq. "
    "Database changes are committed immediately; use a backup before production use."
)