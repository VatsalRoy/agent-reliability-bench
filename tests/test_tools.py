from env.db import snapshot
from env.tools import TOOL_SCHEMAS


# ----- read-only tools -----
def test_get_order(tools):
    r = tools.call("get_order", {"order_id": 107})
    assert r["item"] == "Smart watch"
    assert len(r["tickets"]) == 1


def test_get_order_accepts_string_id(tools):
    assert tools.call("get_order", {"order_id": "101"})["id"] == 101


def test_get_order_missing(tools):
    assert "error" in tools.call("get_order", {"order_id": 999})


def test_get_customer(tools):
    r = tools.call("get_customer", {"customer_id": 1})
    assert r["tier"] == "gold"
    assert [o["id"] for o in r["orders"]] == [101, 107]


def test_search_policy_finds_section(tools):
    r = tools.call("search_policy", {"query": "late delivery compensation"})
    assert "Late delivery" in r["matches"][0]


def test_search_policy_no_match_lists_sections(tools):
    r = tools.call("search_policy", {"query": "zzzz"})
    assert r["matches"] == [] and "Cancellations" in r["available_sections"]


def test_read_tools_do_not_change_db(tools, conn):
    before = snapshot(conn)
    tools.call("get_order", {"order_id": 101})
    tools.call("get_customer", {"customer_id": 2})
    tools.call("search_policy", {"query": "refund"})
    assert snapshot(conn) == before


# ----- write tools -----
def test_issue_refund(tools, conn):
    r = tools.call("issue_refund", {"order_id": 101, "amount": 120, "reason": "damaged"})
    assert r["status"] == "issued"
    assert snapshot(conn)["refunds"][0]["amount"] == 120.0


def test_refund_cannot_exceed_total(tools, conn):
    tools.call("issue_refund", {"order_id": 110, "amount": 20, "reason": "damaged"})
    r = tools.call("issue_refund", {"order_id": 110, "amount": 10, "reason": "damaged"})
    assert "error" in r
    assert len(snapshot(conn)["refunds"]) == 1


def test_refund_rejects_bad_amount(tools):
    assert "error" in tools.call("issue_refund", {"order_id": 101, "amount": -5, "reason": "x"})
    assert "error" in tools.call("issue_refund", {"order_id": 101, "amount": "lots", "reason": "x"})


def test_tools_do_not_enforce_policy(tools):
    """A refund on a shipped order breaks policy, but the tool allows it.
    Catching this is the inspector's job, not the tool's."""
    r = tools.call("issue_refund", {"order_id": 104, "amount": 95, "reason": "damaged"})
    assert r["status"] == "issued"


def test_ticket_lifecycle(tools, conn):
    t = tools.call("create_ticket", {"order_id": 103, "notes": "Refund > $500, needs manager"})
    tools.call("update_ticket_status", {"ticket_id": t["ticket_id"], "status": "on_hold"})
    tickets = {x["id"]: x for x in snapshot(conn)["tickets"]}
    assert tickets[t["ticket_id"]]["status"] == "on_hold"


def test_bad_ticket_status(tools):
    assert "error" in tools.call("update_ticket_status", {"ticket_id": 1, "status": "closed"})


def test_cancel_order(tools, conn):
    tools.call("cancel_order", {"order_id": 105})
    orders = {o["id"]: o for o in snapshot(conn)["orders"]}
    assert orders[105]["status"] == "cancelled"
    assert "error" in tools.call("cancel_order", {"order_id": 105})


def test_send_reply(tools, conn):
    tools.call("send_reply", {"message": "Your refund is on its way."})
    assert snapshot(conn)["replies"][0]["message"] == "Your refund is on its way."


# ----- the dispatcher never crashes -----
def test_unknown_tool(tools):
    assert "error" in tools.call("delete_everything", {})


def test_missing_argument(tools):
    assert "error" in tools.call("issue_refund", {"order_id": 101})


def test_private_methods_not_callable(tools):
    assert "error" in tools.call("_one", {"sql": "DELETE FROM orders"})


def test_every_schema_has_a_tool(tools):
    for s in TOOL_SCHEMAS:
        assert callable(getattr(tools, s["function"]["name"]))



# ----- regression tests for the Phase 4 fixes (see CHANGELOG) -----
def _headings(result):
    return [m.splitlines()[0] for m in result["matches"]]


def test_refund_search_shows_manager_approval_rule(tools):
    """Baseline v1 bug: this rule was cut off by a top-3 limit."""
    r = tools.call("search_policy", {"query": "damaged product refund"})
    assert "## Large refunds (manager approval)" in _headings(r)


def test_refund_search_shows_tickets_rule(tools):
    r = tools.call("search_policy", {"query": "late delivery refund"})
    assert "## Tickets" in _headings(r)
    assert _headings(r)[0] == "## Late delivery compensation"


def test_cancel_search_shows_tickets_rule(tools):
    r = tools.call("search_policy", {"query": "cancel order"})
    assert "## Tickets" in _headings(r) and "## Cancellations" in _headings(r)