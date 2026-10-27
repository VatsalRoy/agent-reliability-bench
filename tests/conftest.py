import pytest

from env.db import connect, create_pristine, fresh_copy
from env.tools import ShopTools


@pytest.fixture(scope="session", autouse=True)
def pristine():
    """Rebuild the pristine DB once per test session so it matches the seed data."""
    return create_pristine()


@pytest.fixture
def db_path(tmp_path):
    return fresh_copy(tmp_path)


@pytest.fixture
def conn(db_path):
    c = connect(db_path)
    yield c
    c.close()


@pytest.fixture
def tools(conn):
    return ShopTools(conn)