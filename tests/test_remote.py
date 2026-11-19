import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from data_loading.ingest.remote import RemoteRangeFile


class _RangeServer:
    """Threaded HTTP server that honors Range requests for a fixed byte string."""

    def __init__(self, data: bytes):
        self.data = data
        self.requested_ranges = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())

    def _make_handler(self):
        data = self.data
        requested = self.requested_ranges

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_HEAD(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

            def do_GET(self):
                header = self.headers.get("Range")
                if header and header.startswith("bytes="):
                    start_s, _, end_s = header[6:].partition("-")
                    start = int(start_s)
                    end = int(end_s) if end_s else len(data) - 1
                    end = min(end, len(data) - 1)
                    requested.append((start, end))
                    body = data[start : end + 1]
                    self.send_response(206)
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header(
                        "Content-Range", f"bytes {start}-{end}/{len(data)}"
                    )
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    self.wfile.write(data)

        return _Handler

    @property
    def url(self):
        host, port = self._server.server_address
        return f"http://{host}:{port}/file.bin"

    def __enter__(self):
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()


def test_remote_range_file_read_and_seek():
    data = bytes(range(256)) * 20  # 5120 bytes
    with _RangeServer(data) as server:
        f = RemoteRangeFile(server.url, "test-agent", block_size=1024)
        try:
            assert f.tell() == 0
            assert f.read(10) == data[:10]
            f.seek(100)
            assert f.read(2000) == data[100:2100]
            f.seek(-5, io.SEEK_END)
            assert f.tell() == len(data) - 5
            assert f.read() == data[-5:]
            f.seek(0)
            assert f.read() == data
            assert f.read(10) == b""  # EOF
            assert f.readable() is True
            assert f.seekable() is True
        finally:
            f.close()
    assert server.requested_ranges  # range requests were actually used


from data_loading.ingest import remote
from utils.schema.parsing import DEFAULT_FILENAME_PATTERN as FILENAME_PATTERN

COPLIOT = "1711364914_Copilot_6.45_0.65.pcap"
CHROME = "1711364914_Google Chrome_6.45_0.65.pcap"
TELEGRAM_TRAILING = "1712045358_Telegram_3.69-3.11_.pcap"


def test_select_all_when_none():
    members = [COPLIOT, CHROME]
    assert remote.select_application_members(members, None, FILENAME_PATTERN) == members


def test_select_all_when_empty():
    members = [COPLIOT, CHROME]
    assert remote.select_application_members(members, [], FILENAME_PATTERN) == members


def test_select_case_insensitive():
    members = [COPLIOT, CHROME]
    assert remote.select_application_members(
        members, ["google chrome"], FILENAME_PATTERN
    ) == [CHROME]


def test_select_skips_unparseable():
    members = [COPLIOT, "readme.txt"]
    assert remote.select_application_members(
        members, ["copilot"], FILENAME_PATTERN
    ) == [COPLIOT]


def test_select_skips_directories():
    class _Dir:
        filename = COPLIOT

        def is_dir(self):
            return True

    members = [COPLIOT, _Dir()]
    assert remote.select_application_members(
        members, ["copilot"], FILENAME_PATTERN
    ) == [COPLIOT]


def test_select_tolerates_trailing_underscore_before_extension():
    members = [TELEGRAM_TRAILING]
    assert remote.select_application_members(
        members, ["telegram"], FILENAME_PATTERN
    ) == [TELEGRAM_TRAILING]


from utils.schema.data_download import DatasetConfig, DownloadConfig
from utils.schema.extraction import ExtractConfig


def _dataset():
    return DatasetConfig(dataset_id="96jwzrp7fd", name="Qom", version=1)


def _download_cfg(**overrides):
    defaults = {"solid_fallback": True}
    defaults.update(overrides)
    return DownloadConfig(**defaults)


class _FakeMember:
    def __init__(self, name, is_dir=False, data_offset=0, compress_size=1, file_size=1):
        self.filename = name
        self._is_dir = is_dir
        self.data_offset = data_offset
        self.compress_size = compress_size
        self.file_size = file_size

    def is_dir(self):
        return self._is_dir


class _FakeRarFile:
    def __init__(self, fileobj):
        self.fileobj = fileobj
        self.solid = False

    def is_solid(self):
        return self.solid

    def infolist(self):
        return [_FakeMember(COPLIOT), _FakeMember(CHROME)]

    def getinfo(self, name):
        return _FakeMember(name)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _solid_rar_factory(fileobj):
    rar = _FakeRarFile(fileobj)
    rar.solid = True
    return rar


class _FakeRemoteFile:
    def seek(self, offset, whence=0):
        return offset

    def read(self, size=-1):
        return b"\0" * max(size, 0)

    def prefetch(self, offset, size):
        return size

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _count_calls(calls):
    def _download_file(url, dest, ua, chunk):
        calls["download_file"] = calls.get("download_file", 0) + 1
        return dest

    return _download_file


def _stub_downloads(monkeypatch, filenames):
    monkeypatch.setattr(
        remote.download,
        "list_dataset_files",
        lambda *a, **k: [{"filename": name} for name in filenames],
    )
    monkeypatch.setattr(remote.download, "filter_archives", lambda files, exts: files)
    monkeypatch.setattr(
        remote.download,
        "resolve_download_url",
        lambda info: "http://example.com/" + info["filename"],
    )


def _stub_sevenzip(monkeypatch):
    monkeypatch.setattr(remote, "resolve_sevenzip", lambda cfg=None: "7z.exe")


def _stub_sparse_and_extract(monkeypatch, tmp_path, payload=b"pcap-bytes-"):
    monkeypatch.setattr(
        remote, "write_sparse_archive", lambda rf, name: tmp_path / "sparse.rar"
    )

    def _fake_extract(sevenzip, sparse, member, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload + member.encode())
        return dest

    monkeypatch.setattr(remote, "extract_member_7z", _fake_extract)


def test_download_application_pcaps_selective(monkeypatch, tmp_path):
    monkeypatch.setattr(
        remote, "RemoteRangeFile", lambda url, ua, **kw: _FakeRemoteFile()
    )
    monkeypatch.setattr(remote.rarfile, "RarFile", lambda f: _FakeRarFile(f))
    _stub_downloads(monkeypatch, ["Qom-01.rar"])
    _stub_sevenzip(monkeypatch)
    _stub_sparse_and_extract(monkeypatch, tmp_path)
    calls = {}
    monkeypatch.setattr(remote.download, "download_file", _count_calls(calls))

    raw = tmp_path / "raw" / "Qom"
    interim = tmp_path / "interim" / "Qom"

    result = remote.download_application_pcaps(
        _dataset(),
        _download_cfg(),
        ExtractConfig(),
        ["google chrome"],
        FILENAME_PATTERN,
        raw,
        interim,
    )

    assert [p.name for p in result] == [CHROME]
    assert (
        interim / "google_chrome" / CHROME
    ).read_bytes() == b"pcap-bytes-" + CHROME.encode()
    assert not (interim / "copilot" / COPLIOT).exists()
    assert "download_file" not in calls  # no full download


def test_download_application_pcaps_solid_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(
        remote, "RemoteRangeFile", lambda url, ua, **kw: _FakeRemoteFile()
    )
    monkeypatch.setattr(remote.rarfile, "RarFile", _solid_rar_factory)
    _stub_downloads(monkeypatch, ["Qom-01.rar"])

    def fake_download_file(url, dest, ua, chunk):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake-rar")
        return dest

    monkeypatch.setattr(remote.download, "download_file", fake_download_file)

    def fake_extract(dest, out, cfg):
        (out / COPLIOT).write_bytes(b"fake-pcap")
        return [out / COPLIOT]

    monkeypatch.setattr(remote.extract, "extract_archive", fake_extract)

    result = remote.download_application_pcaps(
        _dataset(),
        _download_cfg(solid_fallback=True),
        ExtractConfig(),
        ["copilot"],
        FILENAME_PATTERN,
        tmp_path / "raw" / "Qom",
        tmp_path / "interim" / "Qom",
    )

    assert [p.name for p in result] == [COPLIOT]
    assert (tmp_path / "raw" / "Qom" / "Qom-01.rar").read_bytes() == b"fake-rar"


def test_download_application_pcaps_solid_skip(monkeypatch, tmp_path):
    monkeypatch.setattr(
        remote, "RemoteRangeFile", lambda url, ua, **kw: _FakeRemoteFile()
    )
    monkeypatch.setattr(remote.rarfile, "RarFile", _solid_rar_factory)
    _stub_downloads(monkeypatch, ["Qom-01.rar"])
    calls = {}
    monkeypatch.setattr(remote.download, "download_file", _count_calls(calls))

    result = remote.download_application_pcaps(
        _dataset(),
        _download_cfg(solid_fallback=False),
        ExtractConfig(),
        ["copilot"],
        FILENAME_PATTERN,
        tmp_path / "raw" / "Qom",
        tmp_path / "interim" / "Qom",
    )

    assert result == []
    assert "download_file" not in calls


def test_download_application_pcaps_zip_fallback(monkeypatch, tmp_path):
    _stub_downloads(monkeypatch, ["Qom-01.zip"])

    def fake_download_file(url, dest, ua, chunk):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"zip")
        return dest

    monkeypatch.setattr(remote.download, "download_file", fake_download_file)

    def fake_extract(dest, out, cfg):
        (out / COPLIOT).write_bytes(b"copilot-pcap")
        (out / CHROME).write_bytes(b"chrome-pcap")
        return [out / COPLIOT, out / CHROME]

    monkeypatch.setattr(remote.extract, "extract_archive", fake_extract)

    result = remote.download_application_pcaps(
        _dataset(),
        _download_cfg(),
        ExtractConfig(),
        ["copilot"],
        FILENAME_PATTERN,
        tmp_path / "raw" / "Qom",
        tmp_path / "interim" / "Qom",
    )

    assert [p.name for p in result] == [COPLIOT]


def test_download_application_pcaps_skips_existing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        remote, "RemoteRangeFile", lambda url, ua, **kw: _FakeRemoteFile()
    )
    monkeypatch.setattr(remote.rarfile, "RarFile", lambda f: _FakeRarFile(f))
    _stub_downloads(monkeypatch, ["Qom-01.rar"])
    calls = {}
    monkeypatch.setattr(remote.download, "download_file", _count_calls(calls))

    interim = tmp_path / "interim" / "Qom"
    chrome_dir = interim / "google_chrome"
    chrome_dir.mkdir(parents=True, exist_ok=True)
    (chrome_dir / CHROME).write_bytes(b"already-there")

    result = remote.download_application_pcaps(
        _dataset(),
        _download_cfg(),
        ExtractConfig(),
        ["google chrome"],
        FILENAME_PATTERN,
        tmp_path / "raw" / "Qom",
        interim,
    )

    assert [p.name for p in result] == [CHROME]
    assert (chrome_dir / CHROME).read_bytes() == b"already-there"  # not overwritten
    assert "download_file" not in calls


class _TruncatingRangeServer(_RangeServer):
    """Range server that returns at most 10 bytes regardless of the request."""

    def _make_handler(self):
        data = self.data
        requested = self.requested_ranges

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_HEAD(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

            def do_GET(self):
                header = self.headers.get("Range")
                if header and header.startswith("bytes="):
                    start_s, _, end_s = header[6:].partition("-")
                    start = int(start_s)
                    requested.append((start, end_s))
                    body = data[start : start + 10]
                    self.send_response(206)
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    self.wfile.write(data)

        return _Handler


def test_remote_range_file_short_read_raises():
    data = bytes(range(256)) * 20
    with _TruncatingRangeServer(data) as server:
        f = RemoteRangeFile(server.url, "test-agent", block_size=1024)
        try:
            with pytest.raises(RuntimeError, match="Short read"):
                f.read(100)
        finally:
            f.close()


class _FlakyRangeServer(_RangeServer):
    """Range server that fails the first ``fail_times`` range GETs, then serves normally."""

    def __init__(
        self, data, fail_times=1, fail_status=429, fail_content_type="text/html"
    ):
        self.fail_times = fail_times
        self.fail_status = fail_status
        self.fail_content_type = fail_content_type
        self.fail_count = 0
        super().__init__(data)

    def _make_handler(self):
        data = self.data
        requested = self.requested_ranges
        server = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_HEAD(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

            def do_GET(self):
                if server.fail_count < server.fail_times:
                    server.fail_count += 1
                    body = b"<html>challenge</html>"
                    self.send_response(server.fail_status)
                    self.send_header("Content-Type", server.fail_content_type)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                header = self.headers.get("Range")
                if header and header.startswith("bytes="):
                    start_s, _, end_s = header[6:].partition("-")
                    start = int(start_s)
                    end = int(end_s) if end_s else len(data) - 1
                    end = min(end, len(data) - 1)
                    requested.append((start, end))
                    body = data[start : end + 1]
                    self.send_response(206)
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header(
                        "Content-Range", f"bytes {start}-{end}/{len(data)}"
                    )
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    self.wfile.write(data)

        return _Handler


def test_remote_range_file_retries_on_429():
    data = bytes(range(256)) * 20  # 5120 bytes
    with _FlakyRangeServer(data, fail_times=1, fail_status=429) as server:
        f = RemoteRangeFile(server.url, "test-agent", block_size=1024, backoff_base=0.0)
        try:
            assert f.read(10) == data[:10]
        finally:
            f.close()


def test_remote_range_file_retries_on_200_html():
    data = bytes(range(256)) * 20
    with _FlakyRangeServer(
        data, fail_times=1, fail_status=200, fail_content_type="text/html"
    ) as server:
        f = RemoteRangeFile(server.url, "test-agent", block_size=1024, backoff_base=0.0)
        try:
            assert f.read(10) == data[:10]
        finally:
            f.close()


def test_select_application_members_accepts_bare_string():
    members = [COPLIOT, CHROME]
    assert remote.select_application_members(
        members, "google chrome", FILENAME_PATTERN
    ) == [CHROME]


def test_download_application_pcaps_missing_sevenzip_skips(monkeypatch, tmp_path):
    monkeypatch.setattr(
        remote, "RemoteRangeFile", lambda url, ua, **kw: _FakeRemoteFile()
    )
    monkeypatch.setattr(remote.rarfile, "RarFile", lambda f: _FakeRarFile(f))
    _stub_downloads(monkeypatch, ["Qom-01.rar"])

    def _no_sevenzip(cfg=None):
        raise RuntimeError("7-Zip not found")

    monkeypatch.setattr(remote, "resolve_sevenzip", _no_sevenzip)
    calls = {}
    monkeypatch.setattr(remote.download, "download_file", _count_calls(calls))

    result = remote.download_application_pcaps(
        _dataset(),
        _download_cfg(),
        ExtractConfig(),
        ["google chrome"],
        FILENAME_PATTERN,
        tmp_path / "raw" / "Qom",
        tmp_path / "interim" / "Qom",
    )

    assert result == []
    assert "download_file" not in calls


def test_download_application_pcaps_unwritable_member_skipped(monkeypatch, tmp_path):
    monkeypatch.setattr(
        remote, "RemoteRangeFile", lambda url, ua, **kw: _FakeRemoteFile()
    )
    monkeypatch.setattr(remote.rarfile, "RarFile", lambda f: _FakeRarFile(f))
    _stub_downloads(monkeypatch, ["Qom-01.rar"])
    _stub_sevenzip(monkeypatch)
    monkeypatch.setattr(
        remote, "write_sparse_archive", lambda rf, name: tmp_path / "sparse.rar"
    )
    calls = {}
    monkeypatch.setattr(remote.download, "download_file", _count_calls(calls))

    real_open = open

    def fake_open(file, mode="r", *args, **kwargs):
        if "w" in mode:
            raise OSError("disk full")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)

    result = remote.download_application_pcaps(
        _dataset(),
        _download_cfg(solid_fallback=False),
        ExtractConfig(),
        ["google chrome"],
        FILENAME_PATTERN,
        tmp_path / "raw" / "Qom",
        tmp_path / "interim" / "Qom",
    )

    assert result == []
    assert "download_file" not in calls
