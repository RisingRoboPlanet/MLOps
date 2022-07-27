import pandas as pd
import pytest
import src.basic_cleaning


@pytest.fixture
def data():
    df = pd.read_csv("data/raw/census.csv", skipinitialspace=True)
    df = src.basic_cleaning.__clean_dataset(df)
    return df


def test_null(data):
    assert data.shape == data.dropna().shape


def test_question_mark(data):
    assert '?' not in data.values


def test_removed_columns(data):
    assert "fnlgt" not in data.columns
    assert "education-num" not in data.columns
    assert "capital-gain" not in data.columns
    assert "capital-loss" not in data.columns