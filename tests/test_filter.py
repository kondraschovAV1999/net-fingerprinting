import pandas as pd

from data_splitting.filter import cap_pcaps_per_app, filter_applications


def _pkt_df():
    rows = []
    for app in ("Telegram", "Skype"):
        for i in range(4):
            ts = 1000 + i
            fsrc = f"{ts}_{app}.pcap"
            for _ in range(2):
                rows.append(
                    {
                        "application": app,
                        "file_source": fsrc,
                        "capture_timestamp": ts,
                        "len": 100,
                    }
                )
    return pd.DataFrame(rows)


def test_filter_applications_case_insensitive():
    out = filter_applications(_pkt_df(), ["telegram"])
    assert set(out["application"]) == {"Telegram"}
    assert len(out) == 8  # 4 pcaps x 2 flows


def test_filter_applications_none_keeps_all():
    df = _pkt_df()
    assert len(filter_applications(df, None)) == len(df)
    assert len(filter_applications(df, [])) == len(df)


def test_cap_pcaps_per_app_keeps_oldest():
    out = cap_pcaps_per_app(_pkt_df(), 2)
    for app in ("Telegram", "Skype"):
        stamps = set(out[out["application"] == app]["capture_timestamp"])
        assert stamps == {1000, 1001}
    assert len(out) == 8  # 2 apps x 2 pcaps x 2 flows


def test_cap_pcaps_per_app_none_keeps_all():
    df = _pkt_df()
    assert len(cap_pcaps_per_app(df, None)) == len(df)
