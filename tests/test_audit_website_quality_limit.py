from pathlib import Path

import pandas as pd

from src.audit_website_quality import audit_csv


def test_audit_csv_honors_limit_for_csv_input(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    pd.DataFrame(
        [
            {"company_name": "Alpha", "company_website_url": "https://alpha.example"},
            {"company_name": "Beta", "company_website_url": "https://beta.example"},
            {"company_name": "Gamma", "company_website_url": "https://gamma.example"},
        ]
    ).to_csv(input_path, index=False)

    output_path = tmp_path / "output.csv"
    result = audit_csv(input_path, output_path, live=False, limit=2)

    assert len(result) == 2
    assert result["company_name"].tolist() == ["Alpha", "Beta"]
