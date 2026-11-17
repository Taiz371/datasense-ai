import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2 import sql
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "company_db")

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


# 1. Create database if not exists
def create_database_if_not_exists():
    conn = psycopg2.connect(
        dbname="postgres",
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT 1 FROM pg_database WHERE datname = %s",
        (DB_NAME,)
    )

    if not cursor.fetchone():
        cursor.execute(
            sql.SQL("CREATE DATABASE {}").format(
                sql.Identifier(DB_NAME)
            )
        )
        print(f"Database {DB_NAME} created.")
    else:
        print(f"Database {DB_NAME} already exists.")

    cursor.close()
    conn.close()


# 2. Create SQLAlchemy engine
def get_engine():
    return create_engine(DATABASE_URL)


# 3. Create tables & add missing constraints safely
def create_tables():
    schema_query = text("""
        CREATE TABLE IF NOT EXISTS departments (
            dept_id SERIAL PRIMARY KEY,
            dept_name VARCHAR(100) NOT NULL UNIQUE,
            location VARCHAR(100) NOT NULL
        );

        CREATE TABLE IF NOT EXISTS employees (
            emp_id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            role VARCHAR(100) NOT NULL,
            salary NUMERIC(10, 2) NOT NULL,
            hire_date DATE NOT NULL,
            dept_id INT REFERENCES departments(dept_id)
                ON DELETE SET NULL
        );
    """)

    # Non-destructive migration to add UNIQUE constraint to existing tables
    migration_query = text("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint 
                WHERE conname = 'uq_dept_name' OR conname = 'departments_dept_name_key'
            ) THEN
                BEGIN
                    ALTER TABLE departments ADD CONSTRAINT uq_dept_name UNIQUE (dept_name);
                EXCEPTION WHEN OTHERS THEN
                    NULL;
                END;
            END IF;
        END $$;
    """)

    with get_engine().begin() as conn:
        conn.execute(schema_query)
        conn.execute(migration_query)

    print("Tables and constraints created successfully.")


# 4. Insert Indian sample data
def seed_tables():
    with get_engine().begin() as conn:
        conn.execute(text("""
            INSERT INTO departments
                (dept_id, dept_name, location)
            VALUES
                (1, 'Engineering', 'Bengaluru'),
                (2, 'Sales', 'Mumbai'),
                (3, 'Marketing', 'Delhi'),
                (4, 'Human Resources', 'Pune'),
                (5, 'Finance', 'Kolkata')
            ON CONFLICT (dept_id) DO NOTHING;
        """))

        conn.execute(text("""
            INSERT INTO employees
                (emp_id, name, role, salary, hire_date, dept_id)
            VALUES
                (101, 'Aarav Sharma', 'Senior Software Engineer',
                    125000, '2021-03-15', 1),

                (102, 'Priya Verma', 'Sales Executive',
                    78000, '2022-06-01', 2),

                (103, 'Rahul Patel', 'Data Scientist',
                    115000, '2023-01-10', 1),

                (104, 'Ananya Singh', 'Marketing Manager',
                    95000, '2020-11-20', 3),

                (105, 'Rohan Das', 'QA Engineer',
                    88000, '2022-09-12', 1),

                (106, 'Sneha Gupta', 'HR Manager',
                    85000, '2021-07-18', 4),

                (107, 'Vikram Mehta', 'Financial Analyst',
                    92000, '2022-02-14', 5),

                (108, 'Kavya Iyer', 'Python Developer',
                    105000, '2023-05-22', 1),

                (109, 'Arjun Nair', 'Business Analyst',
                    98000, '2021-12-05', 2),

                (110, 'Meera Roy', 'Content Strategist',
                    72000, '2024-01-15', 3)

            ON CONFLICT (emp_id) DO NOTHING;
        """))

    print("Indian sample data inserted successfully.")


# 5. Display all employees
def show_all_employees():
    with get_engine().connect() as conn:
        result = conn.execute(text("""
            SELECT e.emp_id, e.name, e.role,
                   e.salary, e.hire_date, d.dept_name
            FROM employees e
            LEFT JOIN departments d
                ON e.dept_id = d.dept_id
            ORDER BY e.emp_id;
        """))
        for row in result:
            print(row)


# 6. Display all departments
def show_all_departments():
    with get_engine().connect() as conn:
        result = conn.execute(text("""
            SELECT * FROM departments
            ORDER BY dept_id;
        """))
        for row in result:
            print(row)


# 7. Search employee by name
def search_employee(name):
    with get_engine().connect() as conn:
        result = conn.execute(
            text("""
                SELECT * FROM employees
                WHERE name ILIKE :name;
            """),
            {"name": f"%{name}%"}
        )
        for row in result:
            print(row)


# 8. Filter employees by department
def employees_by_department(dept_name):
    with get_engine().connect() as conn:
        result = conn.execute(
            text("""
                SELECT e.emp_id, e.name, e.role, e.salary
                FROM employees e
                JOIN departments d
                    ON e.dept_id = d.dept_id
                WHERE d.dept_name ILIKE :dept;
            """),
            {"dept": f"%{dept_name}%"}
        )
        for row in result:
            print(row)


# 9. Main bootstrap entrypoint
def main():
    create_database_if_not_exists()
    create_tables()
    seed_tables()
    print("\n===== ALL EMPLOYEES =====")
    show_all_employees()
    print("\n===== ALL DEPARTMENTS =====")
    show_all_departments()


if __name__ == "__main__":
    main()