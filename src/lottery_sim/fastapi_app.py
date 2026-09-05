import os
import sqlite3
import webbrowser
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs

from lottery_sim.auth import AuthStore, SESSION_COOKIE_NAME, SESSION_TTL_DAYS
from lottery_sim.dashboard import (
    _get_dashboard_job,
    _job_snapshot,
    _job_sse_events,
    export_dashboard_snapshot,
    load_dashboard_model,
    render_dashboard_html,
    save_dashboard_config,
    start_dashboard_job,
)
from lottery_sim.user_workspace import workspace_for_user
from lottery_sim.dashboard_service import project_id


FASTAPI_INSTALL_HINT = "pip install fastapi uvicorn"


def create_fastapi_app(reports_dir: Path, repo_root: Path):
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"FastAPI 服务需要先安装依赖：{FASTAPI_INSTALL_HINT}") from exc

    root_path = Path(repo_root)
    auth_store = AuthStore(root_path / "data" / "app.sqlite3")
    _bootstrap_admin_user(auth_store)
    app = FastAPI(title="彩票模拟分析系统")

    def current_user(request: Request):
        token = request.cookies.get(SESSION_COOKIE_NAME, "")
        if not token:
            return None
        return auth_store.get_session_user(token)

    def unauthorized_api_response():
        return JSONResponse({"ok": False, "error": "authentication required"}, status_code=401)

    @app.get("/", response_class=HTMLResponse)
    @app.get("/index.html", response_class=HTMLResponse)
    async def index(request: Request):
        user = current_user(request)
        if user is None:
            return RedirectResponse("/login", status_code=303)
        workspace = workspace_for_user(root_path, user.username)
        page = render_dashboard_html(load_dashboard_model(workspace.reports_dir))
        return HTMLResponse(_inject_user_bar(page, user.username))

    @app.get("/login", response_class=HTMLResponse)
    async def login_form(request: Request):
        if current_user(request) is not None:
            return RedirectResponse("/", status_code=303)
        return HTMLResponse(_render_login_page())

    @app.get("/register", response_class=HTMLResponse)
    async def register_form(request: Request):
        if current_user(request) is not None:
            return RedirectResponse("/", status_code=303)
        return HTMLResponse(_render_register_page())

    @app.post("/login")
    async def login(request: Request):
        form = _parse_form_body(await request.body())
        user = auth_store.authenticate(form.get("username", ""), form.get("password", ""))
        if user is None:
            return HTMLResponse(_render_login_page("Invalid username or password."), status_code=401)
        token = auth_store.create_session(user.username)
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(
            SESSION_COOKIE_NAME,
            token,
            max_age=SESSION_TTL_DAYS * 24 * 60 * 60,
            httponly=True,
            samesite="lax",
        )
        return response

    @app.post("/register")
    async def register(request: Request):
        form = _parse_form_body(await request.body())
        username = form.get("username", "")
        password = form.get("password", "")
        if not str(username).strip() or not str(password):
            return HTMLResponse(_render_register_page("请输入账号和密码。"), status_code=400)
        try:
            user = auth_store.create_user(username, password)
        except ValueError:
            return HTMLResponse(_render_register_page("账号只能包含字母、数字、下划线或短横线。"), status_code=400)
        except sqlite3.IntegrityError:
            return HTMLResponse(_render_register_page("账号已存在，请直接登录。"), status_code=409)
        token = auth_store.create_session(user.username)
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(
            SESSION_COOKIE_NAME,
            token,
            max_age=SESSION_TTL_DAYS * 24 * 60 * 60,
            httponly=True,
            samesite="lax",
        )
        return response

    @app.post("/logout")
    async def logout(request: Request):
        token = request.cookies.get(SESSION_COOKIE_NAME, "")
        if token:
            auth_store.delete_session(token)
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(SESSION_COOKIE_NAME)
        return response

    @app.get("/health")
    async def health():
        return {"ok": True, "server": "fastapi", "project_id": project_id(root_path)}

    @app.post("/api/jobs/{action}")
    async def start_job(action: str, request: Request):
        user = current_user(request)
        if user is None:
            return unauthorized_api_response()
        workspace = workspace_for_user(root_path, user.username)
        params = request.query_params
        game_code = params.get("game", "")
        options = {
            key: value
            for key, value in params.items()
            if key != "game"
        }
        job = start_dashboard_job(
            action,
            root_path,
            game_code=game_code,
            options=options,
            data_dir=workspace.shared_data_dir,
            report_dir=workspace.reports_dir,
            recommendation_dir=workspace.recommendation_dir,
            model_dir=workspace.model_dir,
            history_data_dir=workspace.data_dir,
            user_key=user.username,
        )
        status_code = 202 if job.status == "running" else 400
        return JSONResponse(_job_snapshot(job), status_code=status_code)

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str, request: Request):
        user = current_user(request)
        if user is None:
            return unauthorized_api_response()
        job = _get_dashboard_job(job_id)
        if job is None:
            return JSONResponse({"ok": False, "error": "job not found"}, status_code=404)
        if job.user_key and job.user_key != user.username:
            return JSONResponse({"ok": False, "error": "job not found"}, status_code=404)
        return _job_snapshot(job)

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str, request: Request):
        user = current_user(request)
        if user is None:
            return unauthorized_api_response()
        job = _get_dashboard_job(job_id)
        if job is None:
            return JSONResponse({"ok": False, "error": "job not found"}, status_code=404)
        if job.user_key and job.user_key != user.username:
            return JSONResponse({"ok": False, "error": "job not found"}, status_code=404)
        return StreamingResponse(_job_sse_events(job), media_type="text/event-stream")

    @app.post("/api/config")
    async def config(request: Request):
        user = current_user(request)
        if user is None:
            return unauthorized_api_response()
        workspace = workspace_for_user(root_path, user.username)
        body = await request.body()
        form = {}
        if body:
            form = {
                key: values[0]
                for key, values in parse_qs(body.decode("utf-8")).items()
                if values
            }
        save_dashboard_config(workspace.history_db, form)
        return {"ok": True}

    @app.get("/api/export")
    async def export(request: Request, format: str = "csv"):  # noqa: A002
        user = current_user(request)
        if user is None:
            return unauthorized_api_response()
        workspace = workspace_for_user(root_path, user.username)
        try:
            path = export_dashboard_snapshot(
                load_dashboard_model(workspace.reports_dir),
                workspace.exports_dir,
                format,
            )
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        return {"ok": True, "path": str(path)}

    return app


def _bootstrap_admin_user(auth_store: AuthStore) -> None:
    username = os.environ.get("LOTTERY_ADMIN_USER", "admin")
    password = os.environ.get("LOTTERY_ADMIN_PASSWORD", "admin")
    auth_store.bootstrap_admin(username, password)


def _parse_form_body(body: bytes) -> dict:
    if not body:
        return {}
    return {
        key: values[0]
        for key, values in parse_qs(body.decode("utf-8")).items()
        if values
    }


def _render_login_page(error: str = "") -> str:
    error_html = f"<p class=\"login-error\">{_html_escape(error)}</p>" if error else ""
    return f"""
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>彩票模拟分析系统 - 登录</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f4f6f8;
  --panel: #ffffff;
  --text: #172033;
  --muted: #667085;
  --line: #d7dde5;
  --blue: #1d4ed8;
  --blue-hover: #1e40af;
  --green: #16835f;
  --red: #b42318;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  min-height: 100vh;
  font-family: Arial, "Microsoft YaHei", "PingFang SC", sans-serif;
  background: var(--bg);
  color: var(--text);
}}
.login-shell {{
  min-height: 100vh;
  display: grid;
  grid-template-columns: minmax(280px, 0.88fr) minmax(340px, 1fr);
  align-items: stretch;
}}
.login-aside {{
  display: grid;
  align-content: center;
  gap: 22px;
  padding: 54px;
  background: #172033;
  color: #f8fafc;
}}
.brand-lockup {{ display: grid; gap: 12px; }}
.brand-mark {{
  width: 46px;
  height: 46px;
  display: grid;
  place-items: center;
  border: 1px solid rgba(255,255,255,.28);
  border-radius: 8px;
  background: rgba(255,255,255,.08);
  font-size: 22px;
  font-weight: 700;
}}
.brand-lockup h1 {{
  margin: 0;
  font-size: 30px;
  line-height: 1.18;
  letter-spacing: 0;
}}
.brand-lockup p {{
  margin: 0;
  max-width: 440px;
  color: #cbd5e1;
  line-height: 1.7;
  font-size: 15px;
}}
.login-points {{
  display: grid;
  gap: 10px;
  margin: 0;
  padding: 0;
  list-style: none;
  color: #e2e8f0;
  font-size: 14px;
}}
.login-points li {{
  display: flex;
  gap: 8px;
  align-items: center;
}}
.login-points li::before {{
  content: "";
  width: 7px;
  height: 7px;
  flex: 0 0 auto;
  border-radius: 50%;
  background: #34d399;
}}
.login-main {{
  min-height: 100vh;
  display: grid;
  place-items: center;
  padding: 36px 22px;
}}
.login-card {{
  width: min(420px, 100%);
  display: grid;
  gap: 18px;
  padding: 30px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: 0 18px 45px rgba(16, 24, 40, .10);
}}
.login-card header {{ display: grid; gap: 6px; }}
.login-card h2 {{
  margin: 0;
  font-size: 24px;
  line-height: 1.25;
  letter-spacing: 0;
}}
.login-card p {{
  margin: 0;
  color: var(--muted);
  font-size: 14px;
  line-height: 1.6;
}}
.field {{
  display: grid;
  gap: 7px;
  color: #344054;
  font-size: 14px;
}}
.field input {{
  width: 100%;
  height: 42px;
  border: 1px solid #c8ced8;
  border-radius: 6px;
  padding: 0 12px;
  color: var(--text);
  background: #fff;
  font-size: 15px;
}}
.field input:focus {{
  border-color: var(--blue);
  outline: 3px solid rgba(29, 78, 216, .14);
}}
.login-actions {{
  display: grid;
  gap: 12px;
  margin-top: 2px;
}}
.login-button {{
  height: 42px;
  border: 0;
  border-radius: 6px;
  background: var(--blue);
  color: #fff;
  font-size: 15px;
  font-weight: 700;
  cursor: pointer;
}}
.login-button:hover {{ background: var(--blue-hover); }}
.auth-switch {{
  margin: 0;
  color: var(--muted);
  text-align: center;
  font-size: 14px;
}}
.register-link {{
  color: var(--blue);
  font-weight: 700;
  text-decoration: none;
}}
.register-link:hover {{
  color: var(--blue-hover);
  text-decoration: underline;
}}
.login-error {{
  margin: 0;
  padding: 10px 12px;
  border: 1px solid #f3b8ae;
  border-radius: 6px;
  background: #fff4f2;
  color: var(--red);
  font-size: 14px;
}}
@media (max-width: 820px) {{
  .login-shell {{ grid-template-columns: 1fr; }}
  .login-aside {{
    min-height: auto;
    padding: 30px 22px;
  }}
  .brand-lockup h1 {{ font-size: 24px; }}
  .login-points {{ display: none; }}
  .login-main {{
    min-height: auto;
    place-items: start center;
  }}
  .login-card {{ padding: 22px; }}
}}
</style>
</head>
<body>
<main class="login-shell">
<section class="login-aside" aria-label="系统信息">
  <div class="brand-lockup">
    <div class="brand-mark">彩</div>
    <h1>彩票模拟分析系统</h1>
    <p>多用户服务模式已启用。开奖基础数据共享，推荐记录、报告、模型和配置按账号隔离。</p>
  </div>
  <ul class="login-points">
    <li>个人工作区独立保存</li>
    <li>后台任务按账号隔离</li>
    <li>适合 Docker 或局域网部署</li>
  </ul>
</section>
<section class="login-main" aria-label="登录表单">
<form class="login-card" method="post" action="/login">
<header>
<h2>账号登录</h2>
<p>请输入管理员或已创建用户的账号密码。</p>
</header>
{error_html}
<label class="field">账号<input name="username" autocomplete="username" required></label>
<label class="field">密码<input name="password" type="password" autocomplete="current-password" required></label>
<div class="login-actions">
<button class="login-button" type="submit">登录</button>
<p class="auth-switch">还没有账号？<a class="register-link" href="/register">注册新账号</a></p>
</div>
</form>
</section>
</main>
</body>
</html>
""".strip()


def _render_register_page(error: str = "") -> str:
    page = _render_login_page(error)
    replacements = (
        ("<title>彩票模拟分析系统 - 登录</title>", "<title>彩票模拟分析系统 - 注册</title>"),
        ('aria-label="登录表单"', 'aria-label="注册表单"'),
        ('method="post" action="/login"', 'method="post" action="/register"'),
        ("<h2>账号登录</h2>", "<h2>注册账号</h2>"),
        ("<p>请输入管理员或已创建用户的账号密码。</p>", "<p>创建后会自动进入个人工作区。</p>"),
        ('autocomplete="current-password"', 'autocomplete="new-password"'),
        ('<button class="login-button" type="submit">登录</button>', '<button class="login-button" type="submit">注册</button>'),
        ('还没有账号？<a class="register-link" href="/register">注册新账号</a>', '已有账号？<a class="register-link" href="/login">返回登录</a>'),
    )
    for old, new in replacements:
        page = page.replace(old, new, 1)
    return page


def _inject_user_bar(page: str, username: str) -> str:
    bar = f"""
<div style="position:sticky;top:0;z-index:20;display:flex;justify-content:flex-end;gap:10px;align-items:center;padding:8px 18px;background:#ffffff;border-bottom:1px solid #d6dae1;font-family:Arial,Microsoft YaHei,sans-serif;">
<span>{_html_escape(username)}</span>
<form method="post" action="/logout" style="margin:0;">
<button type="submit" style="height:30px;border:1px solid #c8ced8;border-radius:6px;background:#fff;color:#111827;cursor:pointer;">退出登录</button>
</form>
</div>
""".strip()
    if "<body>" in page:
        return page.replace("<body>", f"<body>\n{bar}", 1)
    return f"{bar}\n{page}"


def _html_escape(value: str) -> str:
    import html

    return html.escape(str(value or ""))


def serve_fastapi_dashboard(
    reports_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
    repo_root: Optional[Path] = None,
) -> None:
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"FastAPI 服务需要先安装依赖：{FASTAPI_INSTALL_HINT}") from exc

    app = create_fastapi_app(Path(reports_dir), Path(repo_root or Path.cwd()))
    url = f"http://{host}:{port}"
    print(f"FastAPI 仪表盘地址：{url}")
    if open_browser:
        webbrowser.open(url)
    uvicorn.run(app, host=host, port=port, log_level="info")
