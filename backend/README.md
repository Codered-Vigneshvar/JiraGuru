# JiraGuru Backend Scaffold (FastAPI)

Lightweight, file-based FastAPI backend for managing projects, sprints, and uploading requirement documents. No database or Docker required.

## Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env  # add GEMINI_API_KEY when ready
```

## Run the API

```bash
uvicorn app.main:app --reload
```

The server runs at http://127.0.0.1:8000 with docs at /docs and /redoc. Data is stored under `backend/data/` and will be created automatically.

## Example API Calls

Create a project:

```bash
curl -X POST http://127.0.0.1:8000/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "Demo Project", "description": "First project"}'
```

List projects:

```bash
curl http://127.0.0.1:8000/projects
```

Create a sprint under project 1:

```bash
curl -X POST http://127.0.0.1:8000/projects/1/sprints \
  -H "Content-Type: application/json" \
  -d '{"name": "Sprint 1", "goal": "Initial delivery"}'
```

List sprints for project 1:

```bash
curl http://127.0.0.1:8000/projects/1/sprints
```

Upload a requirement document (PDF/DOCX/TXT) for project 1, sprint 1:

```bash
curl -X POST http://127.0.0.1:8000/projects/1/sprints/1/requirements \
  -F "file=@/path/to/requirements.txt"
```

View extracted chunks for project 1, sprint 1:

```bash
curl http://127.0.0.1:8000/projects/1/sprints/1/requirements/chunks
```

## Notes

- Storage uses JSON files under `backend/data/` for projects, sprints, and requirement chunks.
- Future AI features belong in `app/ai/` (see `story_gen.py` placeholder).
