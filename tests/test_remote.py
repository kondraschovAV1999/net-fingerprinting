import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from data_loading.remote import RemoteRangeFile


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
