from env.db import connect, fresh_copy, snapshot


def test_seed_counts(conn):
    snap = snapshot(conn)
    assert len(snap["customers"]) == 5
    assert len(snap["orders"]) == 10
    assert len(snap["tickets"]) == 1
    assert snap["refunds"] == []
    assert snap["replies"] == []


def test_copies_are_isolated(tmp_path):
    """Changing one copy must not affect another copy (no shared state)."""
    a = fresh_copy(tmp_path / "a")
    b = fresh_copy(tmp_path / "b")
    with connect(a) as ca:
        ca.execute("UPDATE orders SET status = 'cancelled' WHERE id = 105")
        ca.commit()
    with connect(b) as cb:
        status = cb.execute("SELECT status FROM orders WHERE id = 105").fetchone()[0]
    assert status == "processing"


def test_fresh_copy_is_identical_each_time(tmp_path):
    with connect(fresh_copy(tmp_path / "a")) as ca, connect(fresh_copy(tmp_path / "b")) as cb:
        assert snapshot(ca) == snapshot(cb)