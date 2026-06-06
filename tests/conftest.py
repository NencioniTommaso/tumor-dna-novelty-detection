"""
conftest.py
Shared pytest fixtures for the test suite.
"""

import pytest


@pytest.fixture
def sample_sequences():
    return ["ATGCATGC", "GCATGCAT", "AAAAAA", "TTTTTT"]


@pytest.fixture
def custom_alphabet():
    return ('A', 'C', 'G', 'T', 'M')
