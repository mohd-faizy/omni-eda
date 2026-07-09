from __future__ import annotations

import pytest

from omni_eda.loaders import UnsupportedFileTypeError, load_data, load_directory


def test_load_dataframe_passthrough(basic_df):
    result = load_data(basic_df)
    assert result is basic_df


def test_load_csv(tmp_path, basic_df):
    path = tmp_path / "data.csv"
    basic_df.to_csv(path, index=False)
    loaded = load_data(path)
    assert len(loaded) == len(basic_df)
    assert set(loaded.columns) == set(basic_df.columns)


def test_load_json(tmp_path, basic_df):
    path = tmp_path / "data.json"
    basic_df.head(20).to_json(path, orient="records")
    loaded = load_data(path)
    assert len(loaded) == 20


def test_load_parquet(tmp_path, basic_df):
    pytest.importorskip("pyarrow")
    path = tmp_path / "data.parquet"
    basic_df.to_parquet(path)
    loaded = load_data(path)
    assert len(loaded) == len(basic_df)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "does_not_exist.csv")


def test_load_unsupported_extension(tmp_path):
    path = tmp_path / "data.xyz"
    path.write_text("nonsense")
    with pytest.raises(UnsupportedFileTypeError):
        load_data(path)


def test_load_directory(tmp_path, basic_df):
    basic_df.head(10).to_csv(tmp_path / "a.csv", index=False)
    basic_df.head(5).to_json(tmp_path / "b.json", orient="records")
    result = load_directory(tmp_path)
    assert "a" in result and "b" in result
    assert len(result["a"]) == 10
    assert len(result["b"]) == 5


def test_load_csv_chunked(tmp_path, basic_df):
    path = tmp_path / "data.csv"
    basic_df.to_csv(path, index=False)
    chunks = list(load_data(path, chunksize=50))
    assert sum(len(c) for c in chunks) == len(basic_df)
