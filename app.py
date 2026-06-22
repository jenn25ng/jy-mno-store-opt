"""Polaris Colab 배포 진입점 — prototype/ 정적 대시보드를 8080에서 서빙.

- 0.0.0.0:8080 리슨 (플랫폼 필수 규칙)
- GET /health → 200 (헬스체크 필수)
- GET / → 대시보드(simulation_dashboard.html)
- 그 외 경로는 prototype/ 하위 정적 파일(simulation_results.json 등) 서빙
의존성 없음(Python 표준 라이브러리만 사용).
"""
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = int(os.getenv("PORT", "8080"))
WEB_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prototype")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_ROOT, **kwargs)

    def do_GET(self):
        if self.path == "/health":
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path in ("/", ""):
            self.path = "/simulation_dashboard.html"
        return super().do_GET()

    def log_message(self, fmt, *args):  # 컨테이너 로그 간결화
        pass


if __name__ == "__main__":
    httpd = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Serving {WEB_ROOT} on 0.0.0.0:{PORT}", flush=True)
    httpd.serve_forever()
