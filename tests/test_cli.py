from __future__ import annotations

from omni_eda.cli import main


def test_cli_run_produces_html(tmp_path, basic_df):
    csv_path = tmp_path / "data.csv"
    basic_df.to_csv(csv_path, index=False)
    out_dir = tmp_path / "out"

    exit_code = main(["run", str(csv_path), "-o", str(out_dir), "-f", "html", "--quiet", "--no-pairplot"])
    assert exit_code == 0
    assert (out_dir / "report.html").exists()


def test_cli_run_with_target(tmp_path, target_df):
    csv_path = tmp_path / "data.csv"
    target_df.to_csv(csv_path, index=False)
    out_dir = tmp_path / "out"

    exit_code = main(["run", str(csv_path), "--target", "target", "-o", str(out_dir), "-f", "json", "--quiet", "--no-pairplot"])
    assert exit_code == 0
    assert (out_dir / "report.json").exists()


def test_cli_clean_command(tmp_path, messy_df):
    csv_path = tmp_path / "data.csv"
    messy_df.to_csv(csv_path, index=False)
    out_path = tmp_path / "cleaned.csv"

    exit_code = main(["clean", str(csv_path), "-o", str(out_path), "--quiet"])
    assert exit_code == 0
    assert out_path.exists()


def test_cli_missing_file_returns_nonzero(tmp_path):
    exit_code = main(["run", str(tmp_path / "nope.csv"), "-o", str(tmp_path / "out"), "--quiet"])
    assert exit_code == 1


def test_cli_version(capsys):
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    captured = capsys.readouterr()
    assert "omni_eda" in captured.out
