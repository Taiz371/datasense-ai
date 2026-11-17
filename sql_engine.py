import logging
import re
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import requests
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PostgresSQLEngine")

FEW_SHOT_EXAMPLES = [
    {
        "question": "Which department has the highest total salary expense?",
        "query": """
            SELECT d.dept_name,
                   COUNT(e.emp_id) AS total_employees,
                   SUM(e.salary) AS total_salary
            FROM departments AS d
            JOIN employees AS e ON e.dept_id = d.dept_id
            GROUP BY d.dept_id, d.dept_name
            ORDER BY total_salary DESC
            LIMIT 1
        """,
    },
    {
        "question": "List employees earning above their department average salary.",
        "query": """
            SELECT e.name, e.salary, d.dept_name
            FROM employees AS e
            JOIN departments AS d ON d.dept_id = e.dept_id
            WHERE e.salary > (
                SELECT AVG(e2.salary)
                FROM employees AS e2
                WHERE e2.dept_id = e.dept_id
            )
            ORDER BY e.salary DESC
        """,
    },
    {
        "question": "Show the top five highest-paid employees in India.",
        "query": """
            SELECT name, role, salary, hire_date
            FROM employees
            ORDER BY salary DESC
            LIMIT 5
        """,
    },
    {
        "question": "Count employees in each department, including departments with no employees.",
        "query": """
            SELECT d.dept_name, COUNT(e.emp_id) AS employee_count
            FROM departments AS d
            LEFT JOIN employees AS e ON e.dept_id = d.dept_id
            GROUP BY d.dept_id, d.dept_name
            ORDER BY employee_count DESC
        """,
    },
    {
        "question": "Rank employees by salary within each department.",
        "query": """
            SELECT d.dept_name, e.name, e.salary,
                   DENSE_RANK() OVER (
                       PARTITION BY e.dept_id ORDER BY e.salary DESC
                   ) AS salary_rank
            FROM employees AS e
            JOIN departments AS d ON d.dept_id = e.dept_id
        """,
    },
]


class PostgresSQLEngine:
    def __init__(
        self,
        db_url: str,
        google_api_key: str,
        groq_api_key: str,
        statement_timeout_ms: int = 8000,
        max_rows: int = 5000,
        max_retries: int = 3,
        connect_timeout: int = 10,
    ):
        if not db_url:
            raise ValueError("A PostgreSQL database URL is required.")
        if not google_api_key and not groq_api_key:
            raise ValueError("Configure at least one AI provider API key.")

        self.engine: Engine = create_engine(
            db_url,
            pool_pre_ping=True,
            pool_recycle=1800,
            pool_size=5,
            max_overflow=10,
            connect_args={"connect_timeout": connect_timeout},
        )
        self.statement_timeout_ms = max(1000, int(statement_timeout_ms))
        self.max_rows = max(1, int(max_rows))
        self.max_retries = max(1, int(max_retries))
        self.google_api_key = (google_api_key or "").strip()
        self.groq_api_key = (groq_api_key or "").strip()
        self._cached_schema = ""
        self._schema_metadata: Dict[str, Any] = {}

        self.gemini_candidates = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
        ]
        self.groq_candidates = self._discover_groq_models()

    def _discover_groq_models(self) -> List[str]:
        fallback = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ]
        if not self.groq_api_key:
            return []

        try:
            response = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {self.groq_api_key}"},
                timeout=8,
            )
            response.raise_for_status()
            models = response.json().get("data", [])

            excluded = (
                "whisper", "orpheus", "guard", "vision",
                "embedding", "audio", "tts",
            )
            usable = [
                item.get("id", "")
                for item in models
                if item.get("id")
                and not any(word in item["id"].lower() for word in excluded)
            ]

            preferred = [
                model for model in usable
                if "llama-3.3-70b-versatile" in model
                or "llama-3.1-8b-instant" in model
            ]
            return preferred or usable[:3] or fallback

        except Exception as exc:
            logger.warning("Could not discover Groq models: %s", exc)
            return fallback

    def _invoke_llm(self, messages: List[Any]) -> Tuple[str, str]:
        errors: List[str] = []

        if self.google_api_key:
            for model in self.gemini_candidates:
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=model,
                        temperature=0,
                        google_api_key=self.google_api_key,
                        max_retries=1,
                        timeout=45,
                    )
                    response = llm.invoke(messages)
                    return self._content_to_text(response.content), f"Google Gemini ({model})"
                except Exception as exc:
                    errors.append(f"Gemini [{model}]: {exc}")
                    logger.warning("Gemini model %s failed: %s", model, exc)

        if self.groq_api_key:
            for model in self.groq_candidates:
                try:
                    llm = ChatGroq(
                        model_name=model,
                        temperature=0,
                        groq_api_key=self.groq_api_key,
                        max_retries=1,
                        timeout=45,
                    )
                    response = llm.invoke(messages)
                    return self._content_to_text(response.content), f"Groq ({model})"
                except Exception as exc:
                    errors.append(f"Groq [{model}]: {exc}")
                    logger.warning("Groq model %s failed: %s", model, exc)

        details = "\n".join(f"- {error}" for error in errors)
        raise RuntimeError(
            "All configured AI providers failed."
            + (f"\n{details}" if details else " Check your API keys.")
        )

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "\n".join(parts).strip()
        return str(content).strip()

    def get_schema(self, force_refresh: bool = False) -> str:
        if self._cached_schema and not force_refresh:
            return self._cached_schema

        inspector = inspect(self.engine)
        schema_lines: List[str] = []
        metadata: Dict[str, Any] = {}

        for table_name in inspector.get_table_names(schema="public"):
            columns = inspector.get_columns(table_name, schema="public")
            primary_keys = inspector.get_pk_constraint(
                table_name, schema="public"
            ).get("constrained_columns", [])
            foreign_keys = inspector.get_foreign_keys(
                table_name, schema="public"
            )

            column_descriptions = []
            for column in columns:
                nullable = "NULL" if column.get("nullable", True) else "NOT NULL"
                column_descriptions.append(
                    f"{column['name']} ({column['type']}, {nullable})"
                )

            line = f"TABLE public.{table_name}: " + ", ".join(column_descriptions)
            if primary_keys:
                line += f" | PRIMARY KEY: {', '.join(primary_keys)}"

            fk_descriptions = []
            for fk in foreign_keys:
                description = (
                    f"{fk.get('constrained_columns')} -> "
                    f"{fk.get('referred_table')}.{fk.get('referred_columns')}"
                )
                fk_descriptions.append(description)

            if fk_descriptions:
                line += " | FOREIGN KEYS: " + "; ".join(fk_descriptions)

            schema_lines.append(line)
            metadata[table_name] = {
                "columns": [column["name"] for column in columns],
                "primary_keys": primary_keys,
                "foreign_keys": fk_descriptions,
            }

        self._cached_schema = (
            "\n".join(schema_lines) if schema_lines else
            "No tables found in the public schema."
        )
        self._schema_metadata = metadata
        return self._cached_schema

    def is_explanatory_intent(self, question: str) -> bool:
        q = question.strip().lower()
        triggers = (
            "explain", "what is", "what are", "how does", "how do",
            "difference between", "describe", "define", "teach me",
            "tutorial", "tell me about", "why does", "why do",
        )
        return any(q.startswith(trigger) or f" {trigger} " in f" {q} " for trigger in triggers)

    def explain_concept(self, question: str) -> Tuple[str, str]:
        schema = self.get_schema()
        system_message = (
            "You are a senior PostgreSQL database architect and practical SQL tutor. "
            "Answer the user's conceptual question in clear Markdown. "
            "When useful, use examples based on the actual schema below. "
            "Explain concepts clearly with bullet points and syntax-highlighted runnable SQL.\n\n"
            f"DATABASE SCHEMA:\n{schema}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "{question}"),
        ])
        messages = prompt.format_messages(question=question)
        answer, provider = self._invoke_llm(messages)
        return answer, provider

    @staticmethod
    def _clean_sql(raw_response: Any) -> str:
        if isinstance(raw_response, list):
            raw_response = "\n".join(
                str(item.get("text", "")) if isinstance(item, dict)
                else str(item)
                for item in raw_response
            )
        cleaned = str(raw_response).strip()
        cleaned = re.sub(
            r"^```(?:postgresql|postgres|sql)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        return cleaned

    @staticmethod
    def _remove_sql_comments(query: str) -> str:
        result = []
        i = 0
        quote: Optional[str] = None
        while i < len(query):
            char = query[i]
            if quote:
                result.append(char)
                if char == quote:
                    if i + 1 < len(query) and query[i + 1] == quote:
                        result.append(query[i + 1])
                        i += 1
                    else:
                        quote = None
            elif char in ("'", '"'):
                quote = char
                result.append(char)
            elif query.startswith("--", i):
                newline = query.find("\n", i)
                if newline == -1:
                    break
                result.append("\n")
                i = newline
            elif query.startswith("/*", i):
                end = query.find("*/", i + 2)
                if end == -1:
                    raise ValueError("Unclosed SQL block comment.")
                result.append(" ")
                i = end + 1
            else:
                result.append(char)
            i += 1
        return "".join(result)

    def _sanitize_ast(self, sql_query: str) -> str:
        cleaned = self._remove_sql_comments(sql_query).strip()
        cleaned = cleaned.rstrip(";").strip()

        if not cleaned:
            raise ValueError("The AI returned an empty SQL query.")

        statements = [s.strip() for s in cleaned.split(";") if s.strip()]
        if len(statements) > 1:
            cleaned = statements[0]

        first_keyword = re.match(r"([A-Za-z]+)", cleaned)
        if not first_keyword:
            raise ValueError("Could not identify the SQL statement.")

        if first_keyword.group(1).upper() not in {"SELECT", "WITH"}:
            raise ValueError("Only SELECT queries and read-only CTEs are allowed.")

        forbidden = (
            r"\bINSERT\b", r"\bUPDATE\b", r"\bDELETE\b", r"\bMERGE\b",
            r"\bDROP\b", r"\bALTER\b", r"\bCREATE\b", r"\bTRUNCATE\b",
            r"\bGRANT\b", r"\bREVOKE\b", r"\bCALL\b", r"\bDO\b",
            r"\bCOPY\b", r"\bVACUUM\b", r"\bANALYZE\b",
            r"\bSELECT\s+INTO\b",
        )
        for pattern in forbidden:
            if re.search(pattern, cleaned, flags=re.IGNORECASE):
                raise ValueError(
                    f"Read-only guardrail rejected SQL pattern: {pattern}"
                )

        return cleaned

    def generate_sql(self, question: str) -> Tuple[str, str]:
        schema = self.get_schema()

        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{question}"),
            ("ai", "{query}"),
        ])
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=FEW_SHOT_EXAMPLES,
        )

        system_instruction = """
You are an expert PostgreSQL analytics engineer.

Generate exactly ONE PostgreSQL read-only SELECT statement using the schema provided.
Never generate INSERT, UPDATE, DELETE, MERGE, DDL, or multiple statements.
Do not output Markdown backticks or commentary.

DATABASE SCHEMA:
{schema}
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            few_shot_prompt,
            ("human", "{question}"),
        ])
        messages = prompt.format_messages(schema=schema, question=question)
        raw, provider = self._invoke_llm(messages)
        return self._clean_sql(raw), provider

    def repair_sql(
        self,
        question: str,
        failed_sql: str,
        error_msg: str,
    ) -> Tuple[str, str]:
        schema = self.get_schema()
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a PostgreSQL query repair specialist. Return exactly "
                "one read-only SELECT statement without Markdown fences or commentary.",
            ),
            (
                "human",
                "Schema:\n{schema}\n\nQuestion:\n{question}\n\n"
                "Failed SQL:\n{failed_sql}\n\nDatabase error:\n{error}\n\n"
                "Return a corrected read-only PostgreSQL query.",
            ),
        ])
        messages = prompt.format_messages(
            schema=schema,
            question=question,
            failed_sql=failed_sql,
            error=error_msg[:4000],
        )
        raw, provider = self._invoke_llm(messages)
        return self._clean_sql(raw), provider

    def execute_read_query(self, sql_query: str) -> pd.DataFrame:
        safe_sql = self._sanitize_ast(sql_query)

        with self.engine.connect() as connection:
            transaction = connection.begin()
            try:
                connection.execute(text("SET TRANSACTION READ ONLY"))
                connection.execute(
                    text("SELECT set_config('statement_timeout', :timeout, true)"),
                    {"timeout": f"{self.statement_timeout_ms}ms"},
                )
                connection.execute(
                    text("SELECT set_config('search_path', 'public', true)")
                )

                df = pd.read_sql_query(text(safe_sql), connection)

                if len(df) > self.max_rows:
                    df = df.head(self.max_rows)

                transaction.rollback()
                return df
            except Exception:
                transaction.rollback()
                raise

    def execute_with_auto_repair(
        self,
        question: str,
        max_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("Please enter a question.")

        if self.is_explanatory_intent(question):
            content, provider = self.explain_concept(question)
            return {
                "type": "explanation",
                "content": content,
                "df": None,
                "final_sql": None,
                "attempts_log": [],
                "provider_used": provider,
            }

        retries = max(1, int(max_retries or self.max_retries))
        attempts_log: List[Dict[str, Any]] = []
        sql_query, provider_used = self.generate_sql(question)

        for attempt in range(1, retries + 1):
            try:
                safe_sql = self._sanitize_ast(sql_query)
                df = self.execute_read_query(safe_sql)

                attempts_log.append({
                    "attempt": attempt,
                    "sql": safe_sql,
                    "provider": provider_used,
                    "error": None,
                })

                return {
                    "type": "query",
                    "df": df,
                    "final_sql": safe_sql,
                    "attempts_log": attempts_log,
                    "provider_used": provider_used,
                    "content": None,
                }

            except Exception as exc:
                error_message = str(exc)
                attempts_log.append({
                    "attempt": attempt,
                    "sql": sql_query,
                    "provider": provider_used,
                    "error": error_message,
                })
                logger.warning(
                    "SQL attempt %s/%s failed: %s",
                    attempt, retries, error_message,
                )

                if attempt < retries:
                    sql_query, provider_used = self.repair_sql(
                        question, sql_query, error_message
                    )

        final_error = attempts_log[-1]["error"] if attempts_log else "Unknown error"
        raise RuntimeError(
            f"Query failed after {retries} attempt(s). Final error: {final_error}"
        )

    def close(self) -> None:
        self.engine.dispose()