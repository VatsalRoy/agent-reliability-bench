"""The fake shop: database schema, seed data, and clean-copy helpers."""

import shutil
import sqlite3
import tempfile
from pathlib import Path

# A fixed "today" so every run sees the same calendar (reproducibility).
TODAY = "2026-10-20"

PRISTINE_PATH = Path(__file__).parent / "pristine.db"

SCHEMA = """
CREATE TABLE customers (
    id     INTEGER PRIMARY KEY,
    name   TEXT NOT NULL,
    email  TEXT NOT NULL,
    tier   TEXT NOT NULL CHECK (tier IN ('standard', 'gold'))
);

CREATE TABLE orders (
    id             INTEGER PRIMARY KEY,
    customer_id    INTEGER NOT NULL REFERENCES customers(id),
    item           TEXT NOT NULL,
    status         TEXT NOT NULL CHECK (status IN ('processing', 'shipped', 'delivered', 'cancelled')),
    total          REAL NOT NULL,
    order_date     TEXT NOT NULL,
    eta            TEXT NOT NULL,
    delivered_date TEXT
);

CREATE TABLE tickets (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id  INTEGER NOT NULL REFERENCES orders(id),
    status    TEXT NOT NULL CHECK (status IN ('open', 'on_hold', 'solved')),
    notes     TEXT NOT NULL
);

CREATE TABLE refunds (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id  INTEGER NOT NULL REFERENCES orders(id),
    amount    REAL NOT NULL,
    reason    TEXT NOT NULL,
    status    TEXT NOT NULL CHECK (status IN ('issued'))
);

CREATE TABLE replies (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    message  TEXT NOT NULL
);
"""

CUSTOMERS = [
    # id, name, email, tier
    (1, "Alice Chen", "alice@example.com", "gold"),
    (2, "Bob Martinez", "bob@example.com", "standard"),
    (3, "Carol Singh", "carol@example.com", "gold"),
    (4, "David Okafor", "david@example.com", "standard"),
    (5, "Eve Johnson", "eve@example.com", "standard"),
]

ORDERS = [
    # id, customer_id, item, status, total, order_date, eta, delivered_date
    (101, 1, "Wireless headphones", "delivered", 120.00, "2026-10-01", "2026-10-06", "2026-10-05"),
    (102, 2, "Coffee maker",        "delivered",  80.00, "2026-10-02", "2026-10-07", "2026-10-12"),
    (103, 3, "Standing desk",       "delivered", 650.00, "2026-10-03", "2026-10-10", "2026-10-09"),
    (104, 4, "Running shoes",       "shipped",    95.00, "2026-10-14", "2026-10-22", None),
    (105, 5, "Desk lamp",           "processing", 40.00, "2026-10-19", "2026-10-26", None),
    (106, 2, "Backpack",            "delivered",  60.00, "2026-08-20", "2026-08-25", "2026-08-24"),
    (107, 1, "Smart watch",         "delivered", 300.00, "2026-10-04", "2026-10-09", "2026-10-15"),
    (108, 4, "Yoga mat",            "delivered",  35.00, "2026-10-05", "2026-10-10", "2026-10-10"),
    (109, 3, "Blender",             "processing", 150.00, "2026-10-18", "2026-10-25", None),
    (110, 5, "Phone case",          "delivered",  25.00, "2026-10-08", "2026-10-13", "2026-10-13"),
]

TICKETS = [
    # order_id, status, notes  (one pre-existing ticket)
    (107, "open", "Customer asked about late delivery. Awaiting reply."),
]


def create_pristine(path: Path = PRISTINE_PATH) -> Path:
    """Build the pristine database from scratch (overwrites any existing file)."""
    path = Path(path)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", CUSTOMERS)
    conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?)", ORDERS)
    conn.executemany(
        "INSERT INTO tickets (order_id, status, notes) VALUES (?, ?, ?)", TICKETS
    )
    conn.commit()
    conn.close()
    return path


def fresh_copy(workdir: str | Path | None = None) -> Path:
    """Copy the pristine DB into a new temp folder and return the copy's path.

    Every agent attempt gets its own copy, so attempts never share state.
    """
    if not PRISTINE_PATH.exists():
        create_pristine()
    workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="arb_"))
    workdir.mkdir(parents=True, exist_ok=True)
    dest = workdir / "shop.db"
    shutil.copy(PRISTINE_PATH, dest)
    return dest


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def snapshot(conn: sqlite3.Connection) -> dict[str, list[dict]]:
    """Return every row of every table as plain dicts (used by the inspector)."""
    tables = ["customers", "orders", "tickets", "refunds", "replies"]
    return {
        t: [dict(r) for r in conn.execute(f"SELECT * FROM {t} ORDER BY id")]
        for t in tables
    }


if __name__ == "__main__":
    p = create_pristine()
    print(f"Created {p}")
    with connect(p) as c:
        for table, rows in snapshot(c).items():
            print(f"  {table}: {len(rows)} rows")