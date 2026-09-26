from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import handle_conversation


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

FRONTEND_DIR = BASE_DIR / "frontend"


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="AI IT Support Agent",
    description="5-node Agentic AI IT Support System",
    version="1.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =========================================================
# SESSION STORAGE
# =========================================================

sessions = {}


def create_new_session():
    """
    Creates a fresh conversation state.
    """

    return {
        "service_solution_given": False,
        "service_issue_unresolved": False,

        "ticket_offer_pending": False,
        "ticket_waiting_for_email": False,

        "ticket_title": "",
        "ticket_description": "",

        "user_email": ""
    }


# =========================================================
# REQUEST MODELS
# =========================================================

class ChatRequest(BaseModel):

    session_id: str
    message: str


class ResetRequest(BaseModel):

    session_id: str


# =========================================================
# ROOT / FRONTEND
# =========================================================

@app.get("/")
def root():

    index_file = FRONTEND_DIR / "index.html"

    if not index_file.exists():

        return {
            "message": "AI IT Support Agent API is running",
            "status": "online",
            "error": "frontend/index.html not found"
        }

    return FileResponse(
        index_file
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/api/health")
def health():

    return {
        "status": "healthy",
        "service": "AI IT Support Agent"
    }


# =========================================================
# CHAT API
# =========================================================

@app.post("/api/chat")
def chat(
    request: ChatRequest
):

    # -----------------------------------------------------
    # Clean input
    # -----------------------------------------------------

    session_id = (
        request.session_id
        .strip()
    )

    message = (
        request.message
        .strip()
    )


    # -----------------------------------------------------
    # Validate session ID
    # -----------------------------------------------------

    if not session_id:

        return {
            "success": False,
            "answer": "Session ID is required.",
            "show_ticket_option": False,
            "waiting_for_email": False
        }


    # -----------------------------------------------------
    # Validate message
    # -----------------------------------------------------

    if not message:

        return {
            "success": False,
            "answer": "Please enter a message.",
            "show_ticket_option": False,
            "waiting_for_email": False
        }


    # -----------------------------------------------------
    # Create session if new
    # -----------------------------------------------------

    if session_id not in sessions:

        sessions[session_id] = (
            create_new_session()
        )


    conversation_state = (
        sessions[session_id]
    )


    # -----------------------------------------------------
    # Run Agent
    # -----------------------------------------------------

    try:

        answer, updated_state = (
            handle_conversation(
                message,
                conversation_state
            )
        )


        # -------------------------------------------------
        # Save updated state
        # -------------------------------------------------

        sessions[session_id] = (
            updated_state
        )


        # -------------------------------------------------
        # Response
        # -------------------------------------------------

        return {

            "success": True,

            "session_id":
                session_id,

            "answer":
                answer,

            # ---------------------------------------------
            # Ticket state
            # ---------------------------------------------

            "show_ticket_option":
                updated_state.get(
                    "ticket_offer_pending",
                    False
                ),

            "waiting_for_email":
                updated_state.get(
                    "ticket_waiting_for_email",
                    False
                ),

            # ---------------------------------------------
            # Ticket information
            # ---------------------------------------------

            "ticket_title":
                updated_state.get(
                    "ticket_title",
                    ""
                ),

            "ticket_description":
                updated_state.get(
                    "ticket_description",
                    ""
                ),

            # ---------------------------------------------
            # Email
            # ---------------------------------------------

            "email":
                updated_state.get(
                    "user_email",
                    ""
                )
        }


    except Exception as e:

        print("\nCHAT ERROR:")
        print(e)


        return {

            "success": False,

            "session_id":
                session_id,

            "answer":
                "An error occurred while processing your request.",

            "error":
                str(e),

            "show_ticket_option":
                False,

            "waiting_for_email":
                False,

            "ticket_title":
                "",

            "ticket_description":
                ""
        }


# =========================================================
# RESET CHAT
# =========================================================

@app.post("/api/reset")
def reset_chat(
    request: ResetRequest
):

    session_id = (
        request.session_id
        .strip()
    )


    if not session_id:

        return {

            "success": False,

            "message":
                "Session ID is required."
        }


    sessions[session_id] = (
        create_new_session()
    )


    return {

        "success": True,

        "session_id":
            session_id,

        "message":
            "Conversation reset successfully."
    }


# =========================================================
# SESSION STATUS
# =========================================================

@app.get(
    "/api/session/{session_id}"
)
def session_status(
    session_id: str
):

    session_id = (
        session_id.strip()
    )


    if session_id not in sessions:

        return {

            "success": True,

            "exists": False,

            "session_id":
                session_id
        }


    state = sessions[
        session_id
    ]


    return {

        "success": True,

        "exists": True,

        "session_id":
            session_id,

        "service_solution_given":
            state.get(
                "service_solution_given",
                False
            ),

        "service_issue_unresolved":
            state.get(
                "service_issue_unresolved",
                False
            ),

        "ticket_offer_pending":
            state.get(
                "ticket_offer_pending",
                False
            ),

        "ticket_waiting_for_email":
            state.get(
                "ticket_waiting_for_email",
                False
            ),

        "ticket_title":
            state.get(
                "ticket_title",
                ""
            ),

        "ticket_description":
            state.get(
                "ticket_description",
                ""
            )
    }


# =========================================================
# FRONTEND STATIC FILES
# =========================================================
#
# IMPORTANT:
# This must come AFTER all /api/... routes.
#
# =========================================================

if FRONTEND_DIR.exists():

    app.mount(
        "/",
        StaticFiles(
            directory=str(
                FRONTEND_DIR
            ),
            html=True
        ),
        name="frontend"
    )