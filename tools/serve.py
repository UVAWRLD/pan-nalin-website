# Local dev server: like `python3 -m http.server`, but
#   • tells the browser to check for a newer copy of every file on each load
#     (Cache-Control: no-cache), so rebuilt 3D files show up on a normal reload;
#   • accepts POST /__notes from the ?tune notes tool and writes the file into
#     notes/ in the project (git-ignored), so Claude can read it directly.
#   python3 tools/serve.py 8417
import json, re, sys
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTES = ROOT / "notes"


class DevHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_POST(self):
        if self.path.split("?")[0] != "/__notes":
            self.send_error(404); return
        length = int(self.headers.get("Content-Length", 0))
        if length > 60_000_000:
            self.send_error(413); return
        body = self.rfile.read(length)
        try:
            json.loads(body)
        except ValueError:
            self.send_error(400, "expected JSON"); return
        NOTES.mkdir(exist_ok=True)
        name = re.sub(r"[^\w.-]", "", self.headers.get("X-Name", "")) or \
            datetime.now().strftime("hero-notes-%Y-%m-%dT%H-%M-%S")
        path = NOTES / f"{name}.json"
        path.write_bytes(body)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"saved": str(path.relative_to(ROOT))}).encode())


port = int(sys.argv[1]) if len(sys.argv) > 1 else 8417
ThreadingHTTPServer(("127.0.0.1", port), partial(DevHandler, directory=str(ROOT))).serve_forever()
