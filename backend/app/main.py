"""FastAPI application entrypoint for JiraGuru MVP."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

from .config import settings
from .routers import auth, projects, tickets
from .storage import ensure_default_user, ensure_projects_file, ensure_dirs

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

ensure_dirs(settings.data_dir)
ensure_default_user()
ensure_projects_file()
ensure_dirs(STATIC_DIR)

app = FastAPI(
    title="JiraGuru Backend",
    version="0.2.0",
    description="File-based FastAPI backend scaffold for JiraGuru MVP.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(auth.router)
app.include_router(projects.router)
app.include_router(tickets.router)


@app.get("/health", tags=["health"])
def healthcheck() -> dict[str, str]:
    """Simple healthcheck endpoint."""
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def login_page(request: Request):
    """Serve login page."""
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/dashboard/owner", include_in_schema=False)
def owner_dashboard(request: Request):
    """Serve owner dashboard."""
    return templates.TemplateResponse("owner_dashboard.html", {"request": request})


@app.get("/dashboard/user", include_in_schema=False)
def user_dashboard(request: Request):
    """Serve user dashboard."""
    return templates.TemplateResponse("user_dashboard.html", {"request": request})


@app.get("/projects/{project_id}/view", include_in_schema=False)
def view_project_page(request: Request, project_id: str):
    """Serve project view page."""
    return templates.TemplateResponse("project_view.html", {"request": request, "project_id": project_id})


@app.get("/stories/generate", include_in_schema=False)
def generate_stories_page(request: Request):
    """Serve the generate stories page."""
    return templates.TemplateResponse("generate_stories.html", {"request": request})


@app.get("/tickets/{project_id}/{ticket_key}", include_in_schema=False)
def ticket_detail_page(request: Request, project_id: str, ticket_key: str):
    """Serve ticket detail/edit page."""
    return templates.TemplateResponse(
        "ticket_detail.html", {"request": request, "project_id": project_id, "ticket_key": ticket_key}
    )
