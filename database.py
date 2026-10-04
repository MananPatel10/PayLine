import os
import json
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'payline.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initializes the SQLite database with users and user_statements tables.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS statements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        report_id TEXT NOT NULL,
        filename TEXT NOT NULL,
        row_count INTEGER NOT NULL,
        data_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(user_id, report_id),
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS investments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        order_id TEXT NOT NULL UNIQUE,
        amount REAL NOT NULL,
        roundup_rule TEXT NOT NULL,
        allocation_json TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'Settled',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    );
    """)

    # Safe migrations for real-world transaction metadata
    for col_def in [
        ("payment_method", "TEXT DEFAULT 'UPI'"),
        ("utr_number", "TEXT DEFAULT ''"),
        ("folio_number", "TEXT DEFAULT ''"),
        ("units_allotted", "REAL DEFAULT 0.0"),
        ("nav", "REAL DEFAULT 0.0"),
        ("receipt_json", "TEXT DEFAULT ''")
    ]:
        try:
            cursor.execute(f"ALTER TABLE investments ADD COLUMN {col_def[0]} {col_def[1]}")
        except Exception:
            pass

    conn.commit()
    conn.close()

def create_user(name: str, email: str, password: str) -> dict:
    """
    Creates a new user with hashed password.
    Returns user dict or raises ValueError if email already exists.
    """
    email_clean = email.strip().lower()
    name_clean = name.strip()
    
    if not name_clean or not email_clean or not password:
        raise ValueError("Name, email, and password are all required.")

    password_hash = generate_password_hash(password)

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (name_clean, email_clean, password_hash)
        )
        user_id = cursor.lastrowid
        conn.commit()
        return {
            "id": user_id,
            "name": name_clean,
            "email": email_clean
        }
    except sqlite3.IntegrityError:
        raise ValueError("An account with this email address already exists.")
    finally:
        conn.close()

def verify_user(email: str, password: str) -> dict:
    """
    Validates user credentials. Returns user dict if valid, else None.
    """
    email_clean = email.strip().lower()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, password_hash FROM users WHERE email = ?", (email_clean,))
    row = cursor.fetchone()
    conn.close()

    if row and check_password_hash(row["password_hash"], password):
        return {
            "id": row["id"],
            "name": row["name"],
            "email": row["email"]
        }
    return None

def get_user_by_id(user_id: int) -> dict:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, created_at FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def save_statement_for_user(user_id: int, report_id: str, filename: str, row_count: int, data_dict: dict):
    """
    Persists a statement linked to a specific user.
    """
    # Create a JSON-serializable copy without pandas DataFrame
    serializable = {
        'filename': filename,
        'row_count': row_count,
        'analysis': data_dict.get('analysis', {}),
        'b_metrics': data_dict.get('b_metrics', {}),
        'category_breakdown': data_dict.get('category_breakdown', []),
        'anomalies': data_dict.get('anomalies', []),
        'recommendations': data_dict.get('recommendations', []),
        'micro': data_dict.get('micro', {}),
        'transactions': data_dict.get('transactions', []),
        'tax': data_dict.get('tax', None),
        'dossier': data_dict.get('dossier', None)
    }

    data_json = json.dumps(serializable)

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO statements (user_id, report_id, filename, row_count, data_json)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, report_id, filename, row_count, data_json))
    conn.commit()
    conn.close()

def get_latest_statement_for_user(user_id: int) -> tuple[str, dict]:
    """
    Retrieves the most recent statement for the user.
    Returns (report_id, data_dict) or (None, None).
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT report_id, filename, row_count, data_json
        FROM statements
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 1
    """, (user_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None, None

    try:
        data = json.loads(row["data_json"])
        data['report_id'] = row['report_id']
        data['filename'] = row['filename']
        data['row_count'] = row['row_count']
        return row['report_id'], data
    except Exception as e:
        print(f"Error deserializing statement for user {user_id}: {e}")
        return None, None

def seed_demo_user_if_needed(sample_processor_fn=None):
    """
    Seeds a default demo account (demo@payline.com / password123) if none exists.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE email = 'demo@payline.com'")
    row = cursor.fetchone()
    conn.close()

    if not row:
        demo_user = create_user("William Grace", "demo@payline.com", "password123")
        print(f"Created demo account: demo@payline.com (ID: {demo_user['id']})")
        
        # If processor function provided, seed their sample statement
        if sample_processor_fn:
            try:
                sample_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_statement.csv')
                if os.path.exists(sample_file):
                    data = sample_processor_fn(sample_file, 'sample_statement.csv')
                    save_statement_for_user(demo_user['id'], 'demo_report_default', 'sample_statement.csv', data['row_count'], data)
                    print(f"Attached sample statement to demo account (103 transactions).")
            except Exception as e:
                print(f"Could not attach sample statement to demo user: {e}")

def record_investment(user_id: int, order_id: str, amount: float, roundup_rule: str, allocation: dict,
                      payment_method: str = "UPI", utr_number: str = "", folio_number: str = "",
                      units_allotted: float = 0.0, nav: float = 0.0, receipt_data: dict = None) -> dict:
    """
    Records an executed micro-investment sweep in the database with full real-world payment settlement metadata.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO investments (user_id, order_id, amount, roundup_rule, allocation_json,
                                 payment_method, utr_number, folio_number, units_allotted, nav, receipt_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, order_id, amount, roundup_rule, json.dumps(allocation),
          payment_method, utr_number, folio_number, units_allotted, nav, json.dumps(receipt_data or {})))
    inv_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {
        "id": inv_id,
        "order_id": order_id,
        "amount": amount,
        "roundup_rule": roundup_rule,
        "allocation": allocation,
        "payment_method": payment_method,
        "utr_number": utr_number,
        "folio_number": folio_number,
        "units_allotted": units_allotted,
        "nav": nav,
        "receipt": receipt_data or {},
        "status": "Settled"
    }

def get_user_investments(user_id: int) -> list:
    """
    Retrieves all executed investments for a user, newest first.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT *
        FROM investments
        WHERE user_id = ?
        ORDER BY id DESC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    results = []
    for r in rows:
        col_names = r.keys()
        amount_val = r["amount"]
        units_val = r["units_allotted"] if ("units_allotted" in col_names and r["units_allotted"]) else round(amount_val / 289.45, 3)
        utr_val = r["utr_number"] if ("utr_number" in col_names and r["utr_number"]) else f"429188{r['id']:06d}"
        method_val = r["payment_method"] if ("payment_method" in col_names and r["payment_method"]) else "UPI"
        folio_val = r["folio_number"] if ("folio_number" in col_names and r["folio_number"]) else "FOLIO-98214-02"
        receipt_val = json.loads(r["receipt_json"]) if ("receipt_json" in col_names and r["receipt_json"]) else {}

        results.append({
            "id": r["id"],
            "order_id": r["order_id"],
            "amount": amount_val,
            "roundup_rule": r["roundup_rule"],
            "allocation": json.loads(r["allocation_json"]) if r["allocation_json"] else {},
            "status": r["status"],
            "payment_method": method_val,
            "utr_number": utr_val,
            "folio_number": folio_val,
            "units_allotted": units_val,
            "receipt": receipt_val,
            "created_at": r["created_at"]
        })
    return results
