from utils.schema.config import DataLoadingConfig
from utils.schema.data_download import DownloadConfig


def test_default_applications_is_none():
    assert DataLoadingConfig().applications is None


def test_default_solid_fallback_is_true():
    assert DownloadConfig().solid_fallback is True
