from pathlib import Path

from data_loading import pipeline
from utils.schema.config import DataLoadingConfig
from utils.schema.data_download import DatasetConfig


def test_process_dataset_calls_download_application_pcaps(monkeypatch, tmp_path):
    captured = {}

    def fake_download(
        dataset,
        download,
        extract_cfg,
        applications,
        filename_pattern,
        raw_dir,
        interim_dir,
    ):
        captured.update(
            dataset=dataset,
            applications=applications,
            raw_dir=raw_dir,
            interim_dir=interim_dir,
        )
        return []

    monkeypatch.setattr(pipeline, "download_application_pcaps", fake_download)

    cfg = DataLoadingConfig(
        datasets=[DatasetConfig(dataset_id="96jwzrp7fd", name="Qom", version=1)],
        applications=["copilot"],
    )
    cfg.paths.raw_dir = str(tmp_path / "raw")
    cfg.paths.interim_dir = str(tmp_path / "interim")
    cfg.paths.processed_dir = str(tmp_path / "processed")

    dataset = cfg.datasets[0]
    result = pipeline._process_dataset(dataset, cfg)

    assert captured["dataset"].dataset_id == "96jwzrp7fd"
    assert captured["applications"] == ["copilot"]
    assert captured["raw_dir"] == Path(cfg.paths.raw_dir) / "Qom"
    assert captured["interim_dir"] == Path(cfg.paths.interim_dir) / "Qom"
    assert result == []
