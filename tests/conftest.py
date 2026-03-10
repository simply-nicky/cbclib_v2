import pytest
from cbclib_v2.indexer import CBDIndexer, CBDModel

@pytest.fixture
def indexer() -> CBDIndexer:
    return CBDIndexer(10)

@pytest.fixture
def model() -> CBDModel:
    return CBDModel()
