import pytest
from cbclib_v2.indexer import CBDIndexer, RefinerModel

@pytest.fixture
def indexer() -> CBDIndexer:
    return CBDIndexer(10)

@pytest.fixture
def model() -> RefinerModel:
    return RefinerModel()
