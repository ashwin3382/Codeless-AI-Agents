import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services import engine, Base, seed_default_admin, seed_default_mcp_server
from routers import auth, mcp_servers, agents, chat, documents, sessions, notifications

# Bind and construct relational database structures
Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: seed a starter user (ADMIN_USERNAME / ADMIN_PASSWORD, defaults
    # to admin/changeme123) if the users table is empty, so the API is
    # immediately usable for local testing with zero manual setup.
    seed_default_admin()
    seed_default_mcp_server()
    yield
    # Shutdown: nothing to clean up yet - add it here if that changes.


app = FastAPI(title="Production Codeless Agent Space Plane", lifespan=lifespan)

# CORS is configurable via ALLOWED_ORIGINS (comma-separated). Defaults to "*"
# for zero-friction local testing, but credentials are automatically
# disabled whenever the wildcard is in play - browsers reject (and it's
# unsafe to serve) allow_credentials=True together with allow_origins=["*"].
# For anything beyond local testing, set ALLOWED_ORIGINS to your real
# frontend origin(s) so credentialed requests work properly.
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
_uses_wildcard = "*" in ALLOWED_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=not _uses_wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(mcp_servers.router)
app.include_router(agents.router)
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(sessions.router)
app.include_router(notifications.router)
