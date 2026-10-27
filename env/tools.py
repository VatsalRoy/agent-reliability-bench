"""The buttons the agent can press.

Design rule: tools enforce only basic data integrity (the order exists, the
amount is positive). They do NOT enforce the refund policy. Like a real back
office, the system lets you break the rules, and that's exactly what the
inspector will catch.
"""

import re
import sqlite3
from pathlib import Path

POLICY_PATH = Path(__file__).parent / "policy.md"

TICKET_STATUSES = ("open", "on_hold", "solved")


class ToolError(Exception):
    """A tool was called with bad input. The message is shown to the agent."""


class ShopTools:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.conn.row_factory = sqlite3.Row

    # ---------- dispatcher ----------
    def call(self, name: str, args: dict | None) -> dict:
        """Run a tool by name. Never raises: errors come back as {"error": ...}."""
        fn = getattr(self, name, None)
        if name.startswith("_") or name == "call" or not callable(fn):
            return {"error": f"Unknown tool '{name}'"}
        try:
            return fn(**(args or {}))
        except ToolError as e:
            return {"error": str(e)}
        except TypeError as e:  # wrong / missing arguments
            return {"error": f"Bad arguments for {name}: {e}"}

    # ---------- helpers ----------
    def _one(self, sql: str, params=()) -> dict | None:
        row = self.conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def _all(self, sql: str, params=()) -> list[dict]:
        return [dict(r) for r in self.conn.execute(sql, params)]

    @staticmethod
    def _int(value, field: str) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            raise ToolError(f"{field} must be an integer, got {value!r}")

    @staticmethod
    def _float(value, field: str) -> float:
        try:
            return round(float(value), 2)
        except (TypeError, ValueError):
            raise ToolError(f"{field} must be a number, got {value!r}")

    # ---------- read-only tools ----------
    def get_customer(self, customer_id) -> dict:
        cid = self._int(customer_id, "customer_id")
        cust = self._one("SELECT * FROM customers WHERE id = ?", (cid,))
        if not cust:
            raise ToolError(f"Customer {cid} not found")
        cust["orders"] = self._all(
            "SELECT id, item, status, total FROM orders WHERE customer_id = ? ORDER BY id",
            (cid,),
        )
        return cust

    def get_order(self, order_id) -> dict:
        oid = self._int(order_id, "order_id")
        order = self._one("SELECT * FROM orders WHERE id = ?", (oid,))
        if not order:
            raise ToolError(f"Order {oid} not found")
        order["tickets"] = self._all("SELECT * FROM tickets WHERE order_id = ?", (oid,))
        order["refunds"] = self._all("SELECT * FROM refunds WHERE order_id = ?", (oid,))
        return order

    def search_policy(self, query) -> dict:
        """Return the policy sections that best match the query words."""
        text = POLICY_PATH.read_text()
        sections = re.split(r"\n(?=## )", text)[1:]  # skip the top title
        words = {w for w in re.findall(r"[a-z]+", str(query).lower()) if len(w) > 3}
        scored = []
        for sec in sections:
            low = sec.lower()
            score = sum(low.count(w) for w in words)
            if score:
                scored.append((score, sec.strip()))
        scored.sort(key=lambda s: -s[0])
        if not scored:
            headings = [s.splitlines()[0].lstrip("# ") for s in sections]
            return {"matches": [], "available_sections": headings}
        return {"matches": [s for _, s in scored[:3]]}

    # ---------- tools that change the database ----------
    def create_ticket(self, order_id, notes) -> dict:
        oid = self._int(order_id, "order_id")
        if not self._one("SELECT id FROM orders WHERE id = ?", (oid,)):
            raise ToolError(f"Order {oid} not found")
        if not str(notes).strip():
            raise ToolError("notes must not be empty")
        cur = self.conn.execute(
            "INSERT INTO tickets (order_id, status, notes) VALUES (?, 'open', ?)",
            (oid, str(notes)),
        )
        self.conn.commit()
        return {"ticket_id": cur.lastrowid, "order_id": oid, "status": "open"}

    def update_ticket_status(self, ticket_id, status) -> dict:
        tid = self._int(ticket_id, "ticket_id")
        status = str(status).strip().lower()
        if status not in TICKET_STATUSES:
            raise ToolError(f"status must be one of {TICKET_STATUSES}, got {status!r}")
        if not self._one("SELECT id FROM tickets WHERE id = ?", (tid,)):
            raise ToolError(f"Ticket {tid} not found")
        self.conn.execute("UPDATE tickets SET status = ? WHERE id = ?", (status, tid))
        self.conn.commit()
        return {"ticket_id": tid, "status": status}

    def issue_refund(self, order_id, amount, reason) -> dict:
        oid = self._int(order_id, "order_id")
        amt = self._float(amount, "amount")
        order = self._one("SELECT * FROM orders WHERE id = ?", (oid,))
        if not order:
            raise ToolError(f"Order {oid} not found")
        if amt <= 0:
            raise ToolError("amount must be greater than 0")
        already = self._one(
            "SELECT COALESCE(SUM(amount), 0) AS s FROM refunds WHERE order_id = ?", (oid,)
        )["s"]
        if amt + already > order["total"] + 1e-9:
            raise ToolError(
                f"Refund of {amt:.2f} exceeds remaining refundable amount "
                f"{order['total'] - already:.2f}"
            )
        cur = self.conn.execute(
            "INSERT INTO refunds (order_id, amount, reason, status) VALUES (?, ?, ?, 'issued')",
            (oid, amt, str(reason)),
        )
        self.conn.commit()
        return {"refund_id": cur.lastrowid, "order_id": oid, "amount": amt, "status": "issued"}

    def cancel_order(self, order_id) -> dict:
        oid = self._int(order_id, "order_id")
        order = self._one("SELECT * FROM orders WHERE id = ?", (oid,))
        if not order:
            raise ToolError(f"Order {oid} not found")
        if order["status"] == "cancelled":
            raise ToolError(f"Order {oid} is already cancelled")
        self.conn.execute("UPDATE orders SET status = 'cancelled' WHERE id = ?", (oid,))
        self.conn.commit()
        return {"order_id": oid, "status": "cancelled"}

    def send_reply(self, message) -> dict:
        if not str(message).strip():
            raise ToolError("message must not be empty")
        cur = self.conn.execute("INSERT INTO replies (message) VALUES (?)", (str(message),))
        self.conn.commit()
        return {"reply_id": cur.lastrowid, "sent": True}


# ---------- JSON schemas the model sees ----------
def _fn(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SCHEMAS = [
    _fn("get_customer", "Look up a customer and list their orders.",
        {"customer_id": {"type": "integer", "description": "The customer's ID"}},
        ["customer_id"]),
    _fn("get_order", "Look up an order, including its tickets and refunds.",
        {"order_id": {"type": "integer", "description": "The order's ID"}},
        ["order_id"]),
    _fn("search_policy", "Search the customer support policy. Returns matching sections.",
        {"query": {"type": "string", "description": "Keywords, e.g. 'late delivery refund'"}},
        ["query"]),
    _fn("create_ticket", "Create a new support ticket (status 'open') for an order.",
        {"order_id": {"type": "integer"}, "notes": {"type": "string", "description": "What the ticket is about"}},
        ["order_id", "notes"]),
    _fn("update_ticket_status", "Change a ticket's status.",
        {"ticket_id": {"type": "integer"},
         "status": {"type": "string", "enum": list(TICKET_STATUSES)}},
        ["ticket_id", "status"]),
    _fn("issue_refund", "Issue a refund on an order.",
        {"order_id": {"type": "integer"},
         "amount": {"type": "number", "description": "Refund amount in dollars"},
         "reason": {"type": "string", "description": "e.g. damaged, late_delivery, change_of_mind"}},
        ["order_id", "amount", "reason"]),
    _fn("cancel_order", "Cancel an order.",
        {"order_id": {"type": "integer"}},
        ["order_id"]),
    _fn("send_reply", "Send a message to the customer.",
        {"message": {"type": "string"}},
        ["message"]),
]