from __future__ import annotations

import os
import re
import sqlite3
import uuid
from functools import lru_cache
from typing import TypedDict, Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from mail import send_email


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

DB_FILE = os.getenv("DB_FILE", "database/employee.db")
TABLE_NAME = os.getenv("TABLE_NAME", "employees")
MAX_SQL_ROWS = int(os.getenv("MAX_SQL_ROWS", "20"))

POLICY_VECTOR_DB = os.getenv("POLICY_VECTOR_DB", "vector_db")
SERVICE_VECTOR_DB = os.getenv("SERVICE_VECTOR_DB", "vector_db2")

MAX_SERVICE_ATTEMPTS = int(os.getenv("MAX_SERVICE_ATTEMPTS", "2"))


# =========================================================
# STATE
# =========================================================

class AgentState(TypedDict, total=False):
    question: str
    route: Literal["nl2sql", "rag"]
    rag_route: Literal["policy", "service"]

    nl2sql_sql: str
    nl2sql_result: str
    policy_result: str
    service_result: str
    service_answer_found: bool
    final_answer: str
    trace: list


# =========================================================
# STRUCTURED OUTPUT
# =========================================================

class SupervisorDecision(BaseModel):
    route: Literal["nl2sql", "rag"]
    reason: str = Field(description="Reason for selecting the route.")


class RagDecision(BaseModel):
    route: Literal["policy", "service"]
    reason: str = Field(description="Reason for selecting the route.")


class SQLDecision(BaseModel):
    sql: str
    explanation: str


# =========================================================
# HELPERS
# =========================================================

def get_llm():
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0)


def clean_text(value):
    return "" if value is None else str(value).strip()


def add_trace(state, message):
    return state.get("trace", []) + [message]


# =========================================================
# SUPERVISOR
# =========================================================

def supervisor_node(state: AgentState):
    question = clean_text(state["question"])

    print("\n" + "=" * 60)
    print("SUPERVISOR")
    print("=" * 60)
    print("Question:", question)

    prompt = """
You are the Supervisor of an enterprise AI support system.

Choose exactly ONE route.

NL2SQL:
Choose NL2SQL when the user asks about employee
information stored in the database.

Examples:
employee name
employee salary
employee department
employee designation
employee manager
joining date
employee records
number of employees
database information

RAG:
Choose RAG for:

1. Company policy questions:
leave policy
holiday policy
notice period
resignation
probation
benefits
HR rules
company rules

2. IT service questions:
WiFi not working
printer problem
Teams problem
Outlook problem
VPN problem
laptop problem
computer problem
software problem
technical problem
application problem
IT support
troubleshooting

IMPORTANT:
Only SERVICE questions can eventually create an IT support ticket.

NL2SQL and POLICY questions must NEVER create an IT support ticket.

Return only the structured decision.
"""

    try:
        router = get_llm().with_structured_output(SupervisorDecision)
        decision = router.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=question)
        ])

        print("Route:", decision.route)
        print("Reason:", decision.reason)

        return {
            "route": decision.route,
            "trace": [f"Supervisor -> {decision.route}"]
        }

    except Exception as e:
        print("Supervisor error:", e)

        q = question.lower()

        database_words = [
            "employee", "employees", "salary", "department",
            "designation", "manager", "joining", "join date",
            "employee name", "employee names", "database"
        ]

        route = (
            "nl2sql"
            if any(word in q for word in database_words)
            else "rag"
        )

        return {
            "route": route,
            "trace": [f"Supervisor -> {route} (fallback)"]
        }


# =========================================================
# DATABASE
# =========================================================

def get_database_schema():
    if not os.path.exists(DB_FILE):
        raise FileNotFoundError(f"Database not found: {DB_FILE}")

    uri = f"file:{os.path.abspath(DB_FILE)}?mode=ro"

    with sqlite3.connect(uri, uri=True) as conn:
        cursor = conn.cursor()
        cursor.execute(f'PRAGMA table_info("{TABLE_NAME}")')
        columns = cursor.fetchall()

        if not columns:
            raise ValueError(f"Table not found: {TABLE_NAME}")

        schema = []

        for column in columns:
            _, name, data_type, _, _, primary_key = column
            line = f"{name} {data_type or 'TEXT'}"

            if primary_key:
                line += " PRIMARY KEY"

            schema.append(line)

    return "\n".join(schema)


def validate_sql(sql):
    sql = clean_text(sql).rstrip(";").strip()

    if not sql:
        raise ValueError("Empty SQL.")

    if ";" in sql:
        raise ValueError("Multiple SQL statements are not allowed.")

    lowered = re.sub(r"\s+", " ", sql.lower()).strip()

    if not lowered.startswith("select "):
        raise ValueError("Only SELECT queries are allowed.")

    blocked = [
        "insert", "update", "delete", "drop", "alter",
        "create", "replace", "attach", "detach", "pragma",
        "vacuum", "reindex", "begin", "commit", "rollback"
    ]

    for word in blocked:
        if re.search(rf"\b{word}\b", lowered):
            raise ValueError(f"Blocked SQL keyword: {word}")

    if TABLE_NAME.lower() not in lowered:
        raise ValueError(f"SQL must use table {TABLE_NAME}")

    if not re.search(r"\blimit\s+\d+\b", lowered):
        sql += f" LIMIT {MAX_SQL_ROWS}"

    return sql


def execute_sql(sql):
    uri = f"file:{os.path.abspath(DB_FILE)}?mode=ro"

    with sqlite3.connect(uri, uri=True) as conn:
        cursor = conn.cursor()
        cursor.execute(sql)

        rows = cursor.fetchall()
        columns = [item[0] for item in cursor.description]

    if not rows:
        return "No matching records were found in the employee database."

    result = [" | ".join(columns), "-" * 60]

    for row in rows:
        result.append(
            " | ".join("" if value is None else str(value) for value in row)
        )

    return "\n".join(result)


# =========================================================
# NL2SQL
# =========================================================

def nl2sql_node(state: AgentState):
    question = state["question"]

    print("\n" + "=" * 60)
    print("NL2SQL")
    print("=" * 60)

    try:
        schema = get_database_schema()
        llm = get_llm()

        prompt = f"""
You are an NL2SQL agent.

Database table:
{TABLE_NAME}

Columns:
{schema}

User question:
{question}

Rules:
1. Generate exactly one SELECT query.
2. Use only {TABLE_NAME}.
3. Never modify the database.
4. Do not invent columns.
5. Do not use INSERT.
6. Do not use UPDATE.
7. Do not use DELETE.
8. Do not use DROP.
9. Do not use ALTER.
10. Do not use CREATE.
11. Do not use multiple statements.
12. If the question cannot be answered from this database,
generate a safe query that returns no useful records
or explain the limitation.
"""

        generator = llm.with_structured_output(SQLDecision)

        decision = generator.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=question)
        ])

        safe_sql = validate_sql(decision.sql)
        database_result = execute_sql(safe_sql)

        answer_prompt = f"""
User question:
{question}

Database result:
{database_result}

Answer the user using ONLY the database result.

If the information is not present in the database, clearly say:

"I could not find that information in the employee database."

Do not guess.
Do not create an IT ticket.
Do not suggest an IT ticket.
"""

        answer = llm.invoke([
            SystemMessage(content="You are an employee database assistant."),
            HumanMessage(content=answer_prompt)
        ]).content

        print("SQL:", safe_sql)
        print("Database result:", database_result)

        return {
            "nl2sql_sql": safe_sql,
            "nl2sql_result": database_result,
            "final_answer": clean_text(answer),
            "trace": add_trace(state, "NL2SQL -> answer")
        }

    except Exception as e:
        print("NL2SQL error:", e)

        return {
            "final_answer":
                "I could not find that information in the employee database.",
            "nl2sql_result": str(e),
            "trace": add_trace(state, "NL2SQL -> no answer")
        }


# =========================================================
# RAG ROUTER
# =========================================================

def rag_node(state: AgentState):
    question = state["question"]

    print("\n" + "=" * 60)
    print("RAG ROUTER")
    print("=" * 60)

    prompt = """
Choose exactly one:

POLICY:
Company policies, HR rules, leave, holiday,
resignation, notice period, benefits, probation.

SERVICE:
IT problems such as WiFi, printer, Teams,
Outlook, VPN, laptop, computer, software,
application, technical issue, troubleshooting,
IT support.

Only SERVICE can create an IT support ticket.
POLICY cannot create a ticket.

Return the structured decision.
"""

    try:
        router = get_llm().with_structured_output(RagDecision)

        decision = router.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=question)
        ])

        print("RAG route:", decision.route)

        return {
            "rag_route": decision.route,
            "trace": add_trace(state, f"RAG -> {decision.route}")
        }

    except Exception as e:
        print("RAG router error:", e)

        q = question.lower()

        service_words = [
            "wifi", "wi-fi", "printer", "teams", "outlook",
            "vpn", "laptop", "computer", "software",
            "technical", "troubleshoot", "not working",
            "error", "it support"
        ]

        route = (
            "service"
            if any(word in q for word in service_words)
            else "policy"
        )

        return {
            "rag_route": route,
            "trace": add_trace(state, f"RAG -> {route} (fallback)")
        }


# =========================================================
# VECTOR DATABASES
# =========================================================

@lru_cache(maxsize=1)
def load_policy_db():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    return FAISS.load_local(
        POLICY_VECTOR_DB,
        embeddings,
        allow_dangerous_deserialization=True
    )


@lru_cache(maxsize=1)
def load_service_db():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    return FAISS.load_local(
        SERVICE_VECTOR_DB,
        embeddings,
        allow_dangerous_deserialization=True
    )


def retrieve_documents(vectorstore, question, k=5):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    return retriever.invoke(question)


# =========================================================
# POLICY
# =========================================================

def policy_node(state: AgentState):
    question = state["question"]

    print("\n" + "=" * 60)
    print("POLICY")
    print("=" * 60)

    try:
        documents = retrieve_documents(
            load_policy_db(),
            question,
            k=3
        )

        if not documents:
            return {
                "policy_result": "",
                "final_answer":
                    "I could not find that policy information "
                    "in the Company Policy Knowledge Base.",
                "trace":
                    add_trace(state, "Policy -> no information")
            }

        context = "\n\n".join(
            doc.page_content for doc in documents
        )

        prompt = f"""
You are a Company Policy Assistant.

User question:
{question}

Policy document information:
{context}

Rules:
1. Answer ONLY from the policy information.
2. Do not invent policy.
3. Do not use outside knowledge.
4. If the requested policy is not mentioned,
say:

"I could not find that information
in the Company Policy Knowledge Base."

5. Do not create a support ticket.
"""

        answer = get_llm().invoke([
            SystemMessage(
                content="Answer only company policy questions."
            ),
            HumanMessage(content=prompt)
        ]).content

        answer = clean_text(answer)

        return {
            "policy_result": answer,
            "final_answer": answer,
            "trace": add_trace(state, "Policy -> answer")
        }

    except Exception as e:
        print("Policy error:", e)

        return {
            "policy_result": "",
            "final_answer":
                "I could not find that information "
                "in the Company Policy Knowledge Base.",
            "trace":
                add_trace(state, "Policy -> no answer")
        }


# =========================================================
# SERVICE HELPERS
# =========================================================

def service_has_no_answer(answer):
    if not answer:
        return True

    text = answer.lower().strip()

    bad_phrases = [
        "no relevant information",
        "no relevant answer",
        "not found",
        "could not find",
        "cannot find",
        "unable to find",
        "no matching information",
        "i don't know",
        "i do not know",
        "not available"
    ]

    return any(phrase in text for phrase in bad_phrases)


# =========================================================
# SERVICE
# =========================================================

def service_node(state: AgentState):
    question = state["question"]

    print("\n" + "=" * 60)
    print("SERVICE")
    print("=" * 60)

    try:
        documents = retrieve_documents(
            load_service_db(),
            question,
            k=5
        )

        if not documents:
            return {
                "service_result": "",
                "service_answer_found": False,
                "final_answer":
                    "I could not find a suitable solution "
                    "in the IT Service Knowledge Base.",
                "trace":
                    add_trace(state, "Service -> no information")
            }

        context = "\n\n".join(
            doc.page_content for doc in documents
        )

        prompt = f"""
You are an IT Service Support Assistant.

User problem:
{question}

Service Knowledge Base:
{context}

IMPORTANT:

1. Use ONLY the Service Knowledge Base.
2. Give ONE useful troubleshooting approach at a time.
3. Do NOT give a large list of solutions.
4. Do NOT invent technical information.
5. If the user's message contains a new error or symptom, use it.
6. If the knowledge base cannot answer the question,
clearly say that you could not find a suitable solution.
7. Do NOT create a ticket.
8. Do NOT tell the user to create a ticket.
9. Speak naturally like an IT support person.
10. After giving the troubleshooting step,
ask the user to try it and tell you what happened.

Give the most relevant troubleshooting approach.
"""

        answer = get_llm().invoke([
            SystemMessage(
                content="You are a professional IT support assistant."
            ),
            HumanMessage(content=prompt)
        ]).content

        answer = clean_text(answer)

        if service_has_no_answer(answer):
            return {
                "service_result": "",
                "service_answer_found": False,
                "final_answer":
                    "I could not find a suitable solution "
                    "in the IT Service Knowledge Base.",
                "trace":
                    add_trace(state, "Service -> no answer")
            }

        print("Service answer:", answer)

        return {
            "service_result": answer,
            "service_answer_found": True,
            "final_answer": answer,
            "trace": add_trace(state, "Service -> solution")
        }

    except Exception as e:
        print("Service error:", e)

        return {
            "service_result": "",
            "service_answer_found": False,
            "final_answer":
                "I could not retrieve a suitable solution "
                "from the IT Service Knowledge Base.",
            "trace": add_trace(state, "Service -> error")
        }


# =========================================================
# GRAPH
# =========================================================

def supervisor_router(state: AgentState):
    return state.get("route", "rag")


def rag_router(state: AgentState):
    return state.get("rag_route", "service")


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("nl2sql", nl2sql_node)
    graph.add_node("rag", rag_node)
    graph.add_node("policy", policy_node)
    graph.add_node("service", service_node)

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "nl2sql": "nl2sql",
            "rag": "rag"
        }
    )

    graph.add_conditional_edges(
        "rag",
        rag_router,
        {
            "policy": "policy",
            "service": "service"
        }
    )

    graph.add_edge("nl2sql", END)
    graph.add_edge("policy", END)
    graph.add_edge("service", END)

    return graph.compile()


APP = build_graph()


# =========================================================
# TICKET
# =========================================================

def generate_ticket_id():
    return f"TKT-{uuid.uuid4().hex[:8].upper()}"


def generate_ticket_title(question):
    q = question.lower()

    if "wifi" in q or "wi-fi" in q:
        return "WiFi Connectivity Issue"

    if "printer" in q:
        return "Printer Issue"

    if "teams" in q:
        return "Microsoft Teams Issue"

    if "outlook" in q:
        return "Microsoft Outlook Issue"

    if "vpn" in q:
        return "VPN Connectivity Issue"

    if "laptop" in q:
        return "Laptop Support Issue"

    if "computer" in q:
        return "Computer Support Issue"

    return "IT Service Support Issue"


def create_ticket_description(original_problem, history):
    if history:
        steps = "\n".join(
            f"{i}. {item}"
            for i, item in enumerate(history, start=1)
        )
    else:
        steps = "No troubleshooting solution was successfully completed."

    return f"""
Original User Problem:
{original_problem}

Troubleshooting Attempts:

{steps}

Current Status:
Issue remains unresolved.

Escalation:
User requested IT support after the AI troubleshooting process.
"""


# =========================================================
# EMAIL
# =========================================================

def is_valid_email(email):
    pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    return bool(re.fullmatch(pattern, email))


def send_ticket_email(user_email, ticket_id, title, description):
    subject = f"IT Support Ticket - {ticket_id}"

    body = f"""
Hello,

Your IT support ticket has been created.

==================================================
IT SUPPORT TICKET
==================================================

Ticket ID:
{ticket_id}

Ticket Title:
{title}

{description}

==================================================

This ticket was created by the AI IT Support Agent.

Regards,
IT Support Team
"""

    return send_email(
        receiver_email=user_email,
        subject=subject,
        body=body
    )


# =========================================================
# USER INTENT
# =========================================================

def user_requests_ticket(question):
    q = question.lower().strip()

    phrases = [
        "raise ticket",
        "raise a ticket",
        "create ticket",
        "create a ticket",
        "open ticket",
        "open a ticket",
        "make a ticket",
        "submit ticket",
        "submit a ticket",
        "i need a ticket",
        "i want a ticket",
        "please raise a ticket"
    ]

    return any(phrase in q for phrase in phrases)


def user_says_yes(question):
    return question.lower().strip() in [
        "yes",
        "yes please",
        "yeah",
        "yep",
        "sure",
        "okay",
        "ok",
        "go ahead",
        "please do"
    ]


def user_says_no(question):
    return question.lower().strip() in [
        "no",
        "no thanks",
        "not now",
        "cancel"
    ]


def classify_service_feedback(question):
    q = question.lower().strip()

    if user_requests_ticket(q):
        return "ticket"

    solved = [
        "yes",
        "yes it works",
        "yes it is working",
        "working now",
        "works now",
        "it works",
        "fixed",
        "solved",
        "problem solved",
        "problem fixed",
        "that worked",
        "it worked"
    ]

    if any(p == q or p in q for p in solved):
        return "solved"

    not_solved = [
        "no",
        "nope",
        "still not working",
        "still doesn't work",
        "still does not work",
        "not working",
        "didn't work",
        "did not work",
        "doesn't work",
        "does not work",
        "not solved",
        "not fixed",
        "still broken",
        "same problem",
        "problem is still there",
        "issue is still there",
        "nothing changed",
        "no change",
        "not resolved",
        "that didn't help",
        "it didn't help"
    ]

    if any(p == q or p in q for p in not_solved):
        return "not_solved"

    trying = [
        "okay",
        "ok",
        "sure",
        "i'll try",
        "i will try",
        "let me try",
        "let me check",
        "i'll check",
        "i will check"
    ]

    if any(p == q or p in q for p in trying):
        return "trying"

    information_words = [
        "error",
        "shows",
        "showing",
        "message",
        "code",
        "says",
        "display",
        "red light",
        "blue light",
        "offline"
    ]

    if any(word in q for word in information_words):
        return "new_information"

    return "unclear"


# =========================================================
# RUN AGENT
# =========================================================

def run_agent(question):
    state: AgentState = {
        "question": question,
        "trace": []
    }

    return APP.invoke(state)


# =========================================================
# CONVERSATION MANAGER
# =========================================================

def handle_conversation(question, conversation):
    question = question.strip()

    if not question:
        return "Please enter your question.", conversation

    # -----------------------------------------------------
    # WAITING FOR TICKET YES / NO
    # -----------------------------------------------------

    if conversation.get("ticket_offer_pending", False):

        if user_says_yes(question):
            conversation["ticket_offer_pending"] = False
            conversation["waiting_for_email"] = True

            return (
                "Sure. I'll create the IT support ticket for you.\n\n"
                "Please provide your Gmail address.",
                conversation
            )

        if user_says_no(question):
            conversation["ticket_offer_pending"] = False
            conversation["service_active"] = False

            return (
                "Okay. I won't create a ticket. "
                "If you need help later, just let me know.",
                conversation
            )

        return (
            "Would you like me to raise an IT support ticket? "
            "Please answer Yes or No.",
            conversation
        )

    # -----------------------------------------------------
    # WAITING FOR EMAIL
    # -----------------------------------------------------

    if conversation.get("waiting_for_email", False):

        if not is_valid_email(question):
            return (
                "Please provide a valid Gmail address, "
                "for example: name@gmail.com",
                conversation
            )

        email = question
        ticket_id = generate_ticket_id()

        title = conversation.get(
            "ticket_title",
            "IT Service Support Issue"
        )

        description = conversation.get(
            "ticket_description",
            ""
        )

        print("\n" + "=" * 60)
        print("CREATING TICKET")
        print("=" * 60)

        success = send_ticket_email(
            user_email=email,
            ticket_id=ticket_id,
            title=title,
            description=description
        )

        conversation["waiting_for_email"] = False
        conversation["service_active"] = False

        if success:
            conversation["last_ticket_id"] = ticket_id
            conversation["last_ticket_email"] = email

            return (
                f"✅ Ticket created successfully.\n\n"
                f"Ticket ID: {ticket_id}\n"
                f"Ticket Title: {title}\n\n"
                f"The ticket details have been sent to {email}.",
                conversation
            )

        return (
            "I prepared the ticket, but I could not send the email. "
            "Please check the email configuration.",
            conversation
        )

    # -----------------------------------------------------
    # ACTIVE SERVICE
    # -----------------------------------------------------

    if conversation.get("service_active", False):

        if user_requests_ticket(question):
            original_problem = conversation.get(
                "original_service_question",
                question
            )

            conversation["ticket_title"] = generate_ticket_title(
                original_problem
            )

            conversation["ticket_description"] = create_ticket_description(
                original_problem,
                conversation.get("troubleshooting_history", [])
            )

            conversation["ticket_offer_pending"] = False
            conversation["waiting_for_email"] = True

            return (
                "Sure. I'll raise an IT support ticket for this problem.\n\n"
                "Please provide your Gmail address so I can send you "
                "the ticket details.",
                conversation
            )

        feedback = classify_service_feedback(question)
        print("Service feedback:", feedback)

        # SOLVED
        if feedback == "solved":
            conversation["service_active"] = False
            conversation["service_attempts"] = 0
            conversation["troubleshooting_history"] = []

            return (
                "Great! I'm glad the problem has been resolved. 😊",
                conversation
            )

        # TRYING
        if feedback == "trying":
            return (
                "Sure. Please try the troubleshooting step and let me "
                "know what happens.\n\n"
                "You can tell me whether it worked or tell me about "
                "any error you see.",
                conversation
            )

        # NOT SOLVED / NEW INFORMATION
        if feedback in ("not_solved", "new_information"):

            attempts = conversation.get("service_attempts", 0)

            # MAX ATTEMPTS
            if attempts >= MAX_SERVICE_ATTEMPTS:
                original_problem = conversation.get(
                    "original_service_question",
                    question
                )

                conversation["ticket_title"] = generate_ticket_title(
                    original_problem
                )

                conversation["ticket_description"] = create_ticket_description(
                    original_problem,
                    conversation.get("troubleshooting_history", [])
                )

                conversation["ticket_offer_pending"] = True

                return (
                    "I understand. The troubleshooting steps have not "
                    "resolved the problem.\n\n"
                    "I've tried several approaches, so the next step "
                    "is to involve IT support.\n\n"
                    "Would you like me to raise an IT support ticket?",
                    conversation
                )

            # BUILD NEW SEARCH QUERY
            original_problem = conversation.get(
                "original_service_question",
                question
            )

            history = conversation.get(
                "troubleshooting_history",
                []
            )

            history_text = "\n".join(history)

            new_query = f"""
Original IT problem:
{original_problem}

Latest user feedback:
{question}

Previous troubleshooting steps:
{history_text}

The previous troubleshooting did not solve the problem.

Find a DIFFERENT solution from the Service Knowledge Base.

Do not repeat previous steps.

Give ONE troubleshooting approach only.
"""

            result = run_agent(new_query)
            service_answer = result.get("service_result", "")

            # NEW SOLUTION
            if (
                service_answer
                and not service_has_no_answer(service_answer)
            ):
                conversation["service_attempts"] = attempts + 1
                conversation["troubleshooting_history"].append(
                    service_answer
                )

                return (
                    "Thanks for letting me know. "
                    "Let's try another approach.\n\n"
                    + service_answer
                    + "\n\nPlease try this and tell me what happens.",
                    conversation
                )

            # NO NEW SOLUTION
            conversation["service_attempts"] = attempts + 1

            return (
                "I couldn't find another suitable solution in the "
                "Service Knowledge Base.\n\n"
                "Could you give me the exact error message or describe "
                "what happens when you try the previous step?",
                conversation
            )

        # UNCLEAR
        return (
            "I want to make sure I understand what happened.\n\n"
            "Did the troubleshooting step solve the problem?\n\n"
            "You can say:\n"
            "• Yes, it works\n"
            "• No, still not working\n"
            "• I got an error: ...",
            conversation
        )

    # -----------------------------------------------------
    # TICKET WITHOUT SERVICE ISSUE
    # -----------------------------------------------------

    if user_requests_ticket(question):
        return (
            "IT support tickets are only available for unresolved "
            "IT service problems.\n\n"
            "Please describe the IT problem you are experiencing first.",
            conversation
        )

    # -----------------------------------------------------
    # NEW QUESTION
    # -----------------------------------------------------

    conversation["service_active"] = False
    conversation["service_attempts"] = 0
    conversation["troubleshooting_history"] = []
    conversation["ticket_offer_pending"] = False
    conversation["waiting_for_email"] = False

    result = run_agent(question)

    # -----------------------------------------------------
    # SERVICE RESULT
    # -----------------------------------------------------

    if (
        result.get("route") == "rag"
        and result.get("rag_route") == "service"
    ):
        service_answer = result.get("service_result", "")

        # SERVICE ANSWER FOUND
        if (
            service_answer
            and not service_has_no_answer(service_answer)
        ):
            conversation["service_active"] = True
            conversation["service_attempts"] = 1
            conversation["original_service_question"] = question
            conversation["troubleshooting_history"] = [service_answer]
            conversation["ticket_title"] = generate_ticket_title(question)

            return (
                service_answer
                + "\n\nPlease try this and tell me what happens.",
                conversation
            )

        # SERVICE HAS NO ANSWER
        conversation["service_active"] = True
        conversation["service_attempts"] = 0
        conversation["original_service_question"] = question
        conversation["ticket_title"] = generate_ticket_title(question)

        return (
            "I couldn't find a direct solution for this problem in "
            "the IT Service Knowledge Base.\n\n"
            "Could you provide a little more information, such as "
            "the exact error message, the device or application "
            "you are using, and what happens when the problem occurs?",
            conversation
        )

    # NL2SQL / POLICY
    return (
        result.get(
            "final_answer",
            "I could not generate an answer."
        ),
        conversation
    )


# =========================================================
# MAIN
# =========================================================

def main():
    print("\n" + "=" * 60)
    print("       AI IT SUPPORT AGENT")
    print("=" * 60)

    print("\nSupervisor -> NL2SQL OR RAG")
    print("RAG -> Policy OR Service")

    print(
        "\nService -> Troubleshooting -> "
        "User Feedback -> Retry -> Ticket"
    )

    print(
        f"\nMaximum service attempts: {MAX_SERVICE_ATTEMPTS}"
    )

    print("\nType 'exit' to quit.")

    conversation = {
        "service_active": False,
        "service_attempts": 0,
        "original_service_question": "",
        "troubleshooting_history": [],
        "ticket_title": "",
        "ticket_description": "",
        "ticket_offer_pending": False,
        "waiting_for_email": False,
        "last_ticket_id": "",
        "last_ticket_email": ""
    }

    while True:
        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            print("\nGoodbye!")
            break

        if not question:
            continue

        try:
            answer, conversation = handle_conversation(
                question,
                conversation
            )

            print("\nAgent:")
            print(answer)

        except Exception as e:
            print("\nERROR:")
            print(e)


# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    main()