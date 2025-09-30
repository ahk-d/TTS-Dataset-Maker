import http.server
import socketserver
import os
import sys
from functools import partial


class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()


def run_server(directory: str, port: int = 8000):
    os.chdir(directory)
    handler_class = partial(CORSRequestHandler, directory=directory)
    with socketserver.ThreadingTCPServer(("0.0.0.0", port), handler_class) as httpd:
        print(f"Serving {directory} at http://localhost:{port}/ with CORS enabled")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()


if __name__ == "__main__":
    # Usage: python scripts/serve_static.py [directory] [port]
    base_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.getcwd(), "dataset")
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    run_server(base_dir, port)


