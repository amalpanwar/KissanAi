from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
from pathlib import Path
from typing import Any


DDL = [
    """
    CREATE TABLE IF NOT EXISTS research_documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_file TEXT NOT NULL,
        title TEXT,
        publication_year INTEGER,
        domain TEXT,
        district TEXT,
        text_content TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS advisories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        farmer_id TEXT,
        district TEXT NOT NULL,
        season TEXT NOT NULL,
        crop_name TEXT NOT NULL,
        recommendation_text TEXT NOT NULL,
        budget_min_inr REAL,
        budget_max_inr REAL,
        expected_yield_qtl_per_acre REAL,
        expected_revenue_inr_per_acre REAL,
        confidence REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS outcomes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        advisory_id INTEGER,
        farmer_id TEXT,
        actual_crop TEXT,
        actual_yield_qtl_per_acre REAL,
        actual_revenue_inr_per_acre REAL,
        weather_summary TEXT,
        notes TEXT,
        recorded_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(advisory_id) REFERENCES advisories(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS crop_economics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        district TEXT NOT NULL,
        season TEXT NOT NULL,
        crop_name TEXT NOT NULL,
        cost_min_inr_per_acre REAL,
        cost_max_inr_per_acre REAL,
        market_price_inr_per_qtl REAL,
        avg_yield_qtl_per_acre REAL,
        source TEXT,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS market_prices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        district TEXT NOT NULL,
        commodity TEXT NOT NULL,
        modal_price REAL,
        arrival_date TEXT,
        price_unit TEXT,
        source TEXT,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pesticide_recommendations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        crop_name TEXT NOT NULL,
        disease_name_en TEXT,
        disease_name_hi TEXT,
        pesticide_name TEXT,
        ai_g TEXT,
        formulation TEXT,
        dilution TEXT,
        dose_text TEXT,
        waiting_period_days TEXT,
        ai_unit TEXT,
        formulation_unit TEXT,
        dilution_unit TEXT,
        waiting_period_unit TEXT,
        unit_source TEXT,
        source_file TEXT,
        quality_status TEXT,
        quality_flags TEXT,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        email TEXT UNIQUE,
        display_name TEXT,
        role TEXT NOT NULL DEFAULT 'user',
        is_active INTEGER NOT NULL DEFAULT 1,
        is_verified INTEGER NOT NULL DEFAULT 0,
        verification_token TEXT,
        verification_sent_at TEXT,
        verified_at TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        last_login_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS query_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        session_id TEXT,
        user_query TEXT NOT NULL,
        composed_query TEXT,
        topic TEXT,
        answer_text TEXT NOT NULL,
        references_json TEXT,
        district TEXT,
        season TEXT,
        crop_name TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS answer_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query_log_id INTEGER NOT NULL,
        user_id INTEGER,
        rating TEXT NOT NULL,
        correction_text TEXT,
        validation_status TEXT NOT NULL DEFAULT 'unverified',
        validation_method TEXT,
        validation_notes TEXT,
        evidence_json TEXT,
        guardrail_flags TEXT,
        is_training_eligible INTEGER NOT NULL DEFAULT 0,
        reviewed_by_user_id INTEGER,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(query_log_id) REFERENCES query_logs(id),
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(reviewed_by_user_id) REFERENCES users(id)
    )
    """,
]


def get_conn(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path) -> None:
    conn = get_conn(db_path)
    try:
        cur = conn.cursor()
        for stmt in DDL:
            cur.execute(stmt)
        _ensure_column(cur, "users", "email", "TEXT")
        _ensure_column(cur, "users", "is_verified", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(cur, "users", "verification_token", "TEXT")
        _ensure_column(cur, "users", "verification_sent_at", "TEXT")
        _ensure_column(cur, "users", "verified_at", "TEXT")
        _ensure_column(cur, "users", "reset_otp", "TEXT")
        _ensure_column(cur, "users", "reset_otp_expires_at", "TEXT")
        _ensure_column(cur, "users", "auth_provider", "TEXT")
        _ensure_column(cur, "users", "external_user_id", "TEXT")
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username)")
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_external_user_id ON users(external_user_id)")
        conn.commit()
    finally:
        conn.close()


def _ensure_column(cur: sqlite3.Cursor, table: str, column: str, ddl: str) -> None:
    cur.execute(f"PRAGMA table_info({table})")
    cols = {str(r[1]) for r in cur.fetchall()}
    if column not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def insert_research_document(db_path: str | Path, row: dict[str, Any]) -> None:
    conn = get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO research_documents (
                source_file, title, publication_year, domain, district, text_content
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("source_file"),
                row.get("title"),
                row.get("publication_year"),
                row.get("domain"),
                row.get("district"),
                row.get("text_content", ""),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _pbkdf2_hash(password: str, salt: bytes) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return digest.hex()


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    return f"{salt.hex()}${_pbkdf2_hash(password, salt)}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, digest = stored_hash.split("$", 1)
        expected = _pbkdf2_hash(password, bytes.fromhex(salt_hex))
        return hmac.compare_digest(expected, digest)
    except Exception:
        return False


def create_user(
    db_path: str | Path,
    username: str,
    password: str,
    display_name: str = "",
    email: str = "",
) -> tuple[bool, str | dict[str, Any]]:
    uname = (username or "").strip().lower()
    email_clean = (email or "").strip().lower()
    if len(uname) < 3:
        return False, "Username must be at least 3 characters."
    if len(password or "") < 8:
        return False, "Password must be at least 8 characters."
    if not re_match_email(email_clean):
        return False, "Please enter a valid email address."
    conn = get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, is_verified FROM users WHERE email = ?", (email_clean,))
        existing_email = cur.fetchone()
        if existing_email:
            if int(existing_email["is_verified"] or 0) == 1:
                return False, "This email is already registered. Please sign in or use Forgot Password."
            return False, "This email already has a pending account. Please verify it first or request a new verification email."
        cur.execute("SELECT id FROM users WHERE username = ?", (uname,))
        if cur.fetchone():
            return False, "This username is already taken."
        cur.execute("SELECT COUNT(*) AS n FROM users")
        user_count = int(cur.fetchone()["n"])
        role = "admin" if user_count == 0 else "user"
        verification_token = os.urandom(24).hex()
        cur.execute(
            """
            INSERT INTO users (username, password_hash, email, display_name, role, is_verified, verification_token)
            VALUES (?, ?, ?, ?, ?, 0, ?)
            """,
            (uname, hash_password(password), email_clean, (display_name or "").strip() or uname, role, verification_token),
        )
        conn.commit()
        return True, {"role": role, "verification_token": verification_token, "email": email_clean}
    except sqlite3.IntegrityError:
        return False, "Username or email already exists."
    finally:
        conn.close()


def authenticate_user(db_path: str | Path, username: str, password: str) -> dict[str, Any] | None:
    status, user = authenticate_user_status(db_path, username, password)
    if status != "ok":
        return None
    return user


def authenticate_user_status(
    db_path: str | Path, username: str, password: str
) -> tuple[str, dict[str, Any] | None]:
    identity = (username or "").strip().lower()
    conn = get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, username, email, display_name, role, is_active, is_verified, password_hash
            FROM users
            WHERE lower(username) = ? OR lower(email) = ?
            """,
            (identity, identity),
        )
        row = cur.fetchone()
        if not row or int(row["is_active"] or 0) != 1:
            return "invalid", None
        if not verify_password(password, str(row["password_hash"])):
            return "invalid", None
        if int(row["is_verified"] or 0) != 1:
            return "unverified", {
                "id": int(row["id"]),
                "username": str(row["username"]),
                "email": str(row["email"] or ""),
            }
        cur.execute("UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE id = ?", (row["id"],))
        conn.commit()
        return "ok", {
            "id": int(row["id"]),
            "username": str(row["username"]),
            "email": str(row["email"] or ""),
            "display_name": str(row["display_name"] or row["username"]),
            "role": str(row["role"] or "user"),
        }
    finally:
        conn.close()


def get_user_by_id(db_path: str | Path, user_id: int) -> dict[str, Any] | None:
    conn = get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, username, email, display_name, role, is_active, is_verified
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        if int(row["is_active"] or 0) != 1 or int(row["is_verified"] or 0) != 1:
            return None
        return {
            "id": int(row["id"]),
            "username": str(row["username"]),
            "email": str(row["email"] or ""),
            "display_name": str(row["display_name"] or row["username"]),
            "role": str(row["role"] or "user"),
        }
    finally:
        conn.close()


def verify_user_by_token(db_path: str | Path, token: str) -> tuple[bool, str]:
    token = (token or "").strip()
    if not token:
        return False, "Missing verification token."
    conn = get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, username FROM users
            WHERE verification_token = ? AND is_active = 1
            """,
            (token,),
        )
        row = cur.fetchone()
        if not row:
            return False, "Invalid or expired verification link."
        cur.execute(
            """
            UPDATE users
            SET is_verified = 1,
                verification_token = NULL,
                verified_at = CURRENT_TIMESTAMP,
                reset_otp = NULL,
                reset_otp_expires_at = NULL
            WHERE id = ?
            """,
            (row["id"],),
        )
        conn.commit()
        return True, str(row["username"])
    finally:
        conn.close()


def get_user_by_email(db_path: str | Path, email: str) -> dict[str, Any] | None:
    email_clean = (email or "").strip().lower()
    conn = get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, username, email, display_name, role, is_active, is_verified
            FROM users
            WHERE email = ?
            """,
            (email_clean,),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _slug_username(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "." for ch in (value or "").strip())
    cleaned = ".".join(part for part in cleaned.split(".") if part)
    return cleaned or "user"


def _next_unique_username(cur: sqlite3.Cursor, base: str, exclude_id: int | None = None) -> str:
    candidate = _slug_username(base)
    idx = 0
    while True:
        probe = candidate if idx == 0 else f"{candidate}.{idx}"
        if exclude_id is None:
            cur.execute("SELECT id FROM users WHERE username = ?", (probe,))
        else:
            cur.execute("SELECT id FROM users WHERE username = ? AND id != ?", (probe, exclude_id))
        if not cur.fetchone():
            return probe
        idx += 1


def upsert_external_user(
    db_path: str | Path,
    *,
    provider: str,
    external_user_id: str,
    email: str,
    display_name: str = "",
    username: str = "",
    is_verified: bool = True,
) -> dict[str, Any]:
    email_clean = (email or "").strip().lower()
    external_id = (external_user_id or "").strip()
    if not email_clean or not external_id:
        raise ValueError("email and external_user_id are required")
    conn = get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS n FROM users")
        user_count = int(cur.fetchone()["n"])
        role_default = "admin" if user_count == 0 else "user"
        cur.execute(
            """
            SELECT id, username, email, display_name, role
            FROM users
            WHERE external_user_id = ? OR email = ?
            LIMIT 1
            """,
            (external_id, email_clean),
        )
        row = cur.fetchone()
        preferred_username = username.strip() or email_clean.split("@", 1)[0]
        preferred_display = (display_name or "").strip() or preferred_username
        verified_int = 1 if is_verified else 0
        if row:
            user_id = int(row["id"])
            role = str(row["role"] or role_default)
            final_username = _next_unique_username(cur, preferred_username, exclude_id=user_id)
            cur.execute(
                """
                UPDATE users
                SET username = ?,
                    password_hash = COALESCE(NULLIF(password_hash, ''), '__supabase__'),
                    email = ?,
                    display_name = ?,
                    role = ?,
                    is_active = 1,
                    is_verified = ?,
                    auth_provider = ?,
                    external_user_id = ?,
                    verification_token = NULL,
                    verified_at = CASE WHEN ? = 1 THEN COALESCE(verified_at, CURRENT_TIMESTAMP) ELSE verified_at END
                WHERE id = ?
                """,
                (
                    final_username,
                    email_clean,
                    preferred_display,
                    role,
                    verified_int,
                    provider,
                    external_id,
                    verified_int,
                    user_id,
                ),
            )
        else:
            final_username = _next_unique_username(cur, preferred_username)
            cur.execute(
                """
                INSERT INTO users (
                    username, password_hash, email, display_name, role,
                    is_active, is_verified, auth_provider, external_user_id, verified_at
                ) VALUES (?, '__supabase__', ?, ?, ?, 1, ?, ?, ?, CASE WHEN ? = 1 THEN CURRENT_TIMESTAMP ELSE NULL END)
                """,
                (
                    final_username,
                    email_clean,
                    preferred_display,
                    role_default,
                    verified_int,
                    provider,
                    external_id,
                    verified_int,
                ),
            )
            user_id = int(cur.lastrowid)
            role = role_default
        conn.commit()
        return {
            "id": user_id,
            "username": final_username,
            "email": email_clean,
            "display_name": preferred_display,
            "role": role,
        }
    finally:
        conn.close()


def set_verification_token_for_email(db_path: str | Path, email: str) -> tuple[bool, str | dict[str, Any]]:
    email_clean = (email or "").strip().lower()
    conn = get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, username, is_verified FROM users WHERE email = ? AND is_active = 1", (email_clean,))
        row = cur.fetchone()
        if not row:
            return False, "No account found for this email."
        if int(row["is_verified"] or 0) == 1:
            return False, "This email is already verified. Please sign in."
        token = os.urandom(24).hex()
        cur.execute(
            """
            UPDATE users
            SET verification_token = ?, verification_sent_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (token, row["id"]),
        )
        conn.commit()
        return True, {"email": email_clean, "verification_token": token, "username": str(row["username"])}
    finally:
        conn.close()


def create_password_reset_otp(db_path: str | Path, email: str) -> tuple[bool, str | dict[str, Any]]:
    email_clean = (email or "").strip().lower()
    conn = get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, username, is_active, is_verified
            FROM users
            WHERE email = ?
            """,
            (email_clean,),
        )
        row = cur.fetchone()
        if not row or int(row["is_active"] or 0) != 1:
            return False, "No verified account found for this email."
        if int(row["is_verified"] or 0) != 1:
            return False, "Please verify this email first before resetting the password."
        otp = f"{int.from_bytes(os.urandom(3), 'big') % 1000000:06d}"
        cur.execute(
            """
            UPDATE users
            SET reset_otp = ?,
                reset_otp_expires_at = datetime('now', '+15 minutes')
            WHERE id = ?
            """,
            (otp, row["id"]),
        )
        conn.commit()
        return True, {"email": email_clean, "otp": otp, "username": str(row["username"])}
    finally:
        conn.close()


def reset_password_with_otp(
    db_path: str | Path,
    email: str,
    otp: str,
    new_password: str,
) -> tuple[bool, str]:
    email_clean = (email or "").strip().lower()
    otp_clean = (otp or "").strip()
    if len(new_password or "") < 8:
        return False, "New password must be at least 8 characters."
    conn = get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, reset_otp, reset_otp_expires_at
            FROM users
            WHERE email = ? AND is_active = 1 AND is_verified = 1
            """,
            (email_clean,),
        )
        row = cur.fetchone()
        if not row:
            return False, "No verified account found for this email."
        if not row["reset_otp"] or str(row["reset_otp"]) != otp_clean:
            return False, "Invalid OTP."
        cur.execute(
            "SELECT datetime('now') <= datetime(?) AS valid_until",
            (row["reset_otp_expires_at"],),
        )
        valid = cur.fetchone()
        if not valid or int(valid["valid_until"] or 0) != 1:
            return False, "OTP has expired. Please request a new one."
        cur.execute(
            """
            UPDATE users
            SET password_hash = ?,
                reset_otp = NULL,
                reset_otp_expires_at = NULL
            WHERE id = ?
            """,
            (hash_password(new_password), row["id"]),
        )
        conn.commit()
        return True, "Password updated successfully. Please sign in."
    finally:
        conn.close()


def re_match_email(value: str) -> bool:
    return bool(value) and ("@" in value) and ("." in value.rsplit("@", 1)[-1])


def create_query_log(db_path: str | Path, row: dict[str, Any]) -> int | None:
    conn = get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO query_logs (
                user_id, session_id, user_query, composed_query, topic, answer_text,
                references_json, district, season, crop_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("user_id"),
                row.get("session_id"),
                row.get("user_query", ""),
                row.get("composed_query", ""),
                row.get("topic"),
                row.get("answer_text", ""),
                json.dumps(row.get("references", []), ensure_ascii=False),
                row.get("district"),
                row.get("season"),
                row.get("crop_name"),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    except Exception:
        return None
    finally:
        conn.close()


def feedback_exists(db_path: str | Path, query_log_id: int, user_id: int | None = None) -> bool:
    conn = get_conn(db_path)
    try:
        cur = conn.cursor()
        if user_id is None:
            cur.execute("SELECT 1 FROM answer_feedback WHERE query_log_id = ? LIMIT 1", (query_log_id,))
        else:
            cur.execute(
                "SELECT 1 FROM answer_feedback WHERE query_log_id = ? AND user_id = ? LIMIT 1",
                (query_log_id, user_id),
            )
        return cur.fetchone() is not None
    finally:
        conn.close()


def save_feedback(db_path: str | Path, row: dict[str, Any]) -> int | None:
    conn = get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO answer_feedback (
                query_log_id, user_id, rating, correction_text, validation_status,
                validation_method, validation_notes, evidence_json, guardrail_flags,
                is_training_eligible
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("query_log_id"),
                row.get("user_id"),
                row.get("rating", "not_helpful"),
                row.get("correction_text"),
                row.get("validation_status", "unverified"),
                row.get("validation_method"),
                row.get("validation_notes"),
                json.dumps(row.get("evidence", []), ensure_ascii=False),
                json.dumps(row.get("guardrail_flags", []), ensure_ascii=False),
                int(row.get("is_training_eligible", 0)),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    except Exception:
        return None
    finally:
        conn.close()


def get_feedback_queue(db_path: str | Path, limit: int = 20) -> list[dict[str, Any]]:
    conn = get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                f.id,
                f.query_log_id,
                f.rating,
                f.correction_text,
                f.validation_status,
                f.validation_method,
                f.validation_notes,
                f.evidence_json,
                f.guardrail_flags,
                f.is_training_eligible,
                f.created_at,
                q.user_query,
                q.answer_text,
                q.topic,
                u.username
            FROM answer_feedback f
            LEFT JOIN query_logs q ON q.id = f.query_log_id
            LEFT JOIN users u ON u.id = f.user_id
            WHERE f.validation_status IN ('needs_review', 'source_matched')
            ORDER BY f.created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = []
        for r in cur.fetchall():
            item = dict(r)
            for key in ("evidence_json", "guardrail_flags"):
                try:
                    item[key] = json.loads(item.get(key) or "[]")
                except Exception:
                    item[key] = []
            rows.append(item)
        return rows
    finally:
        conn.close()


def review_feedback(
    db_path: str | Path,
    feedback_id: int,
    reviewer_user_id: int,
    status: str,
    training_eligible: bool,
) -> None:
    conn = get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE answer_feedback
            SET validation_status = ?,
                is_training_eligible = ?,
                reviewed_by_user_id = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (status, int(training_eligible), reviewer_user_id, feedback_id),
        )
        conn.commit()
    finally:
        conn.close()


def export_training_feedback(db_path: str | Path, output_path: str | Path) -> int:
    conn = get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                f.id,
                q.user_query,
                q.answer_text,
                q.topic,
                q.references_json,
                f.correction_text,
                f.validation_status,
                f.validation_method,
                f.validation_notes
            FROM answer_feedback f
            JOIN query_logs q ON q.id = f.query_log_id
            WHERE f.is_training_eligible = 1
            ORDER BY f.created_at DESC
            """
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            item = dict(r)
            try:
                item["references_json"] = json.loads(item.get("references_json") or "[]")
            except Exception:
                item["references_json"] = []
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    return count


def get_training_feedback_examples(db_path: str | Path, limit: int = 200) -> list[dict[str, Any]]:
    conn = get_conn(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                f.id,
                q.user_query,
                q.topic,
                q.district,
                q.crop_name,
                q.answer_text,
                f.correction_text,
                f.validation_status,
                f.is_training_eligible
            FROM answer_feedback f
            JOIN query_logs q ON q.id = f.query_log_id
            WHERE f.validation_status = 'accepted' OR f.is_training_eligible = 1
            ORDER BY f.updated_at DESC, f.created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
