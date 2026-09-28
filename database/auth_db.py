"""PostgreSQL-backed user authentication (Use Case 1: User Login)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

import psycopg2
import bcrypt

from config import DB_CONFIG, setup_logging

logger = setup_logging()


class AuthManager:
    """Manage the users table and credential verification."""

    def __init__(self, config: dict[str, str] | None = None) -> None:
        self.config = config or DB_CONFIG

    @contextmanager
    def connection(self) -> Iterator[psycopg2.extensions.connection]:
        conn = psycopg2.connect(**self.config)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def is_available(self) -> bool:
        try:
            with self.connection():
                return True
        except Exception as exc:
            logger.warning("PostgreSQL unavailable for auth: %s", exc)
            return False

    def create_users_table(self) -> bool:
        """Create the users table if it does not exist, with a default admin."""
        try:
            with self.connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS users (
                            id SERIAL PRIMARY KEY,
                            username VARCHAR(100) UNIQUE NOT NULL,
                            password_hash TEXT NOT NULL,
                            role VARCHAR(50) DEFAULT 'user',
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        );
                        """
                    )
                    cursor.execute("SELECT COUNT(*) FROM users;")
                    if cursor.fetchone()[0] == 0:
                        # Truncate password to 72 bytes for bcrypt compatibility
                        password = "admin123"
                        password_bytes = password.encode('utf-8')
                        if len(password_bytes) > 72:
                            password_bytes = password_bytes[:72]
                        cursor.execute(
                            "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s);",
                            ("admin", bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode('utf-8'), "administrator"),
                        )
                        logger.info("Created default admin user (admin / admin123). Change this password.")
            return True
        except Exception as exc:
            logger.warning("Could not create users table: %s", exc)
            return False

    def register_user(self, username: str, password: str, role: str = "user") -> tuple[bool, str]:
        """Create a new user account. Returns (success, message)."""
        if not username or not password:
            return False, "Username and password are required."
        if len(password) < 6:
            return False, "Password must be at least 6 characters."
        if not self.create_users_table():
            return False, "Database is unavailable."
        try:
            with self.connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1 FROM users WHERE username = %s;", (username,))
                    if cursor.fetchone():
                        return False, "Username already exists."
                    # Truncate password to 72 bytes for bcrypt compatibility
                    password_bytes = password.encode('utf-8')
                    if len(password_bytes) > 72:
                        password_bytes = password_bytes[:72]
                    cursor.execute(
                        "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s);",
                        (username, bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode('utf-8'), role),
                    )
            return True, "Account created successfully."
        except Exception as exc:
            logger.warning("Could not register user: %s", exc)
            return False, "Registration failed."

    def verify_user(self, username: str, password: str) -> Optional[dict]:
        """Validate credentials. Returns user dict on success, else None."""
        if not self.create_users_table():
            return None
        try:
            with self.connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT id, username, password_hash, role FROM users WHERE username = %s;",
                        (username,),
                    )
                    row = cursor.fetchone()
                    if not row:
                        return None
                    user_id, db_username, password_hash, role = row
                    # Truncate password to 72 bytes for bcrypt compatibility
                    password_bytes = password.encode('utf-8')
                    if len(password_bytes) > 72:
                        password_bytes = password_bytes[:72]
                    if bcrypt.checkpw(password_bytes, password_hash.encode('utf-8')):
                        return {"id": user_id, "username": db_username, "role": role}
                    return None
        except Exception as exc:
            logger.warning("Could not verify user: %s", exc)
            return None
