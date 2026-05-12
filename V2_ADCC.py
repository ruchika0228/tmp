# -*- coding: utf-8 -*-
"""
Super Admin WebSocket Server  (v14.1)
=====================================
Changes from v14.0:

  OLLAMA / DOCKER-COMPOSE FIXES
  -----------------------------
  1. OLLAMA_BASE_URL now reads from env var OLLAMA_BASE_URL so that
     Docker-Compose service-name resolution works out of the box
     (e.g. http://ollama:11434 instead of http://localhost:11434).

  2. call_llm() - the bare `except Exception: pass` is replaced with
     proper logging + structured fallback so every Ollama error
     (connection refused, model not found, timeout, JSON parse failure)
     is printed to stdout AND returns a safe dict instead of crashing.

  3. Model name sanitisation - OLLAMA_MODEL is stripped of whitespace
     at startup to prevent subtle "model not found" errors from
     copy-paste artefacts.

  4. argparse --model choices updated to accept:
     "qwen/qwen3-32b", "gemma3:4b", "qwen3.5:cloud", "qwen3.5:397b-cloud", "kimi-k2.6:cloud".

  5. Ollama reachability check at startup - prints a clear warning
     if the Ollama endpoint is unreachable so operators know
     immediately instead of seeing silent LLM errors at runtime.

All v14.0 session-management features are 100% preserved.
"""

import json
import httpx
import asyncio
import re
import uuid
import argparse
from datetime import datetime
from typing import Optional

from dateutil import parser as dateutil_parser
import websockets
from websockets.legacy.server import WebSocketServerProtocol
from groq import Groq
from dotenv import load_dotenv

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
import os

# Load environment variables from .env
load_dotenv()

# FIX 1: Read OLLAMA_BASE_URL from environment so Docker-Compose
#         service-name resolution works (e.g. http://ollama:11434).
#         Falls back to localhost for local dev.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
BACKEND_BASE = os.getenv("BACKEND_BASE", "http://localhost:4000")
# FIX 3: Strip whitespace from model name to avoid silent "not found" errors.
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen3.5:cloud").strip()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

MAX_RETRIES     = 3
DATE_FORMAT_OUT = "%Y-%m-%dT%H:%M"

# ---------------------------------------------
# FIXED BOT REPLIES
# ---------------------------------------------

IDENTITY_REPLY = (
    "I am an ITSM BOT. I assist with IT Service Management concepts and processes."
)
GREETING_REPLY = (
    "Hello! I am an ITSM BOT. How can I help you with ITSM today?"
)
CAPABILITY_REPLY = (
    "I can help you with ITSM topics such as Helpdesk, Asset Management, CMDB, "
    "Problem Management, Change Management, Service Management, User Management, "
    "and Data Administration."
)
IRRELEVANT_REPLY = (
    "I can't provide an answer to this question. "
    "Please ask something related to ITSM."
)
FAQ_FALLBACK_REPLY = (
    "I can help with ITSM topics including:\n"
    " Helpdesk (incidents & service requests)\n"
    " Asset & Inventory Management\n"
    " Configuration Management (CMDB)\n"
    " Problem Management\n"
    " Change Management\n"
    " Service Management\n"
    " User & Identity Management\n"
    " Data Administration\n\n"
    "Please ask a specific question about any of these modules."
)

# ---------------------------------------------
# BRAND FILTER
# ---------------------------------------------

_BRAND_PATTERNS = [
    (re.compile(r"\biTop\b",  re.IGNORECASE), "the ITSM system"),
    (re.compile(r"\bGLPi?\b", re.IGNORECASE), "the ITSM system"),
    (re.compile(r"\bglpi\b",  re.IGNORECASE), "the ITSM system"),
    (re.compile(r"\bitop\b",  re.IGNORECASE), "the ITSM system"),
]

def scrub_brands(text: str) -> str:
    if not text:
        return text
    for pattern, replacement in _BRAND_PATTERNS:
        text = pattern.sub(replacement, text)
    return text

# ---------------------------------------------
# PRE-LLM MESSAGE CLASSIFIER
# ---------------------------------------------

_IDENTITY_PATTERNS = re.compile(
    r"\b(who\s+are\s+you|what\s+are\s+you|introduce\s+yourself"
    r"|your\s+name|what('s|\s+is)\s+(your\s+name|you)|tell\s+me\s+about\s+yourself)\b",
    re.IGNORECASE,
)
_GREETING_PATTERNS = re.compile(
    r"^\s*(hi+|hello+|hey+|good\s+(morning|afternoon|evening|day)|greetings|howdy|sup|what'?s\s+up)\s*[!.?]*\s*$",
    re.IGNORECASE,
)
_CAPABILITY_PATTERNS = re.compile(
    r"\b(how\s+can\s+you\s+help|what\s+can\s+you\s+do|what\s+do\s+you\s+do"
    r"|what\s+are\s+your\s+(capabilities|features|functions)"
    r"|how\s+do\s+you\s+(help|assist|work)|what\s+(help|assistance)\s+can\s+you\s+(provide|give|offer))\b",
    re.IGNORECASE,
)

def _pre_classify(text: str) -> Optional[str]:
    if _IDENTITY_PATTERNS.search(text):
        return "identity"
    if _GREETING_PATTERNS.match(text):
        return "greeting"
    if _CAPABILITY_PATTERNS.search(text):
        return "capability"
    return None

# ---------------------------------------------
# ITSM KEYWORD SET
# ---------------------------------------------

ITSM_KEYWORDS = {
    "helpdesk","incident","request","ticket","service request",
    "problem","known error","kedb","rca","root cause",
    "change","change management","change request","approve",
    "cmdb","configuration","ci","configuration item",
    "asset","inventory","hardware","software","network device",
    "service","service catalog","sla","service level","slt",
    "service level target","contract","service contract",
    "user","identity","role","rbac","permission","access",
    "data admin","data administration","workflow","automation",
    "create","raise","log","open","document","provision",
    "assign","escalate","resolve","close","link",
    "priority","impact","urgency","title","description",
    "start_date","end_date","fallback","category","outage",
    "org","organisation","organization","department",
    "itsm","itil","lifecycle","workaround","solution",
    "escalation","notification","audit","sla breach",
    "response time","resolution time","availability",
    "service name","service description","service type",
}

def _is_itsm_related(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in ITSM_KEYWORDS)

# ---------------------------------------------
# DATE NORMALIZER
# ---------------------------------------------

_DATE_CLEANUPS = [
    (re.compile(r"^(\d{2})[.\-/](\d{2})[.\-/](\d{4})(.*)$"),
     lambda m: f"{m.group(3)}-{m.group(2)}-{m.group(1)}{m.group(4)}"),
    (re.compile(r"^(\d{4})/(\d{2})/(\d{2})(.*)$"),
     lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}{m.group(4)}"),
    (re.compile(r"^(\d{4}-\d{2}-\d{2}[T ]?\d{2}:\d{2}):\d{2}.*$"),
     lambda m: m.group(1)),
]

def normalize_date(raw: str) -> Optional[str]:
    if not raw or not raw.strip():
        return raw
    value = raw.strip()
    for pattern, replacer in _DATE_CLEANUPS:
        m = pattern.match(value)
        if m:
            value = replacer(m)
            break
    if "T" not in value and " " in value:
        value = value.replace(" ", "T", 1)
    if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$", value):
        return value
    if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        return value + "T00:00"
    for dayfirst in (False, True):
        try:
            dt = dateutil_parser.parse(value, dayfirst=dayfirst)
            return dt.strftime(DATE_FORMAT_OUT)
        except (ValueError, OverflowError):
            continue
    return None

# ---------------------------------------------
# OUTAGE NORMALIZER
# ---------------------------------------------

def normalize_outage(value: str) -> bool:
    return str(value).strip().lower() in ("yes", "true", "1")

# ---------------------------------------------
# NUMBERED OPTION LISTS
# ---------------------------------------------

PRIORITY_OPTIONS = [
    ("Critical", "1"),
    ("High",     "2"),
    ("Medium",   "3"),
    ("Low",      "4"),
]

DOMAIN_OPTIONS = [
    ("Application", "Application"),
    ("Desktop",     "Desktop"),
    ("Network",     "Network"),
    ("Server",      "Server"),
]

CATEGORY_OPTIONS = [
    ("Normal",    "normal"),
    ("Standard",  "standard"),
    ("Emergency", "emergency"),
]

OUTAGE_OPTIONS = [
    ("No",  "no"),
    ("Yes", "yes"),
]

URGENCY_OPTIONS = [
    ("Critical", "1"),
    ("High",     "2"),
    ("Medium",   "3"),
    ("Low",      "4"),
]

IMPACT_OPTIONS = [
    ("A department", "1"),
    ("A service",    "2"),
    ("A person",     "3"),
]

SERVICE_STATUS_OPTIONS = [
    ("Active",   "active"),
    ("Inactive", "inactive"),
    ("Draft",    "draft"),
]

SLT_PRIORITY_OPTIONS = [
    ("Critical", "1"),
    ("High",     "2"),
    ("Medium",   "3"),
    ("Low",      "4"),
]

SLT_UNIT_OPTIONS = [
    ("Minutes", "minutes"),
    ("Hours",   "hours"),
    ("Days",    "days"),
]

SLT_REQUEST_TYPE_OPTIONS = [
    ("Incident",        "incident"),
    ("Service Request", "service_request"),
]

SLT_METRIC_OPTIONS = [
    ("Time to own",     "Time to own"),
    ("Time to resolve", "Time to resolve"),
]

CONTRACT_CURRENCY_OPTIONS = [
    ("USD - US Dollar",     "USD"),
    ("EUR - Euro",          "EUR"),
    ("GBP - British Pound", "GBP"),
    ("INR - Indian Rupee",  "INR"),
    ("Other",               "OTHER"),
]

# ---------------------------------------------
# FIELD METADATA
# ---------------------------------------------

FIELD_META = {
    "title"                  : {"hint": "Short title"},
    "description"            : {"hint": "Detailed description"},
    "service_name"           : {"hint": "Affected service name"},
    "priority"               : {"hint": "1=Critical  2=High  3=Medium  4=Low",
                                 "allowed": ["1","2","3","4"]},
    "request_type"           : {"hint": "incident  or  service_request",
                                 "allowed": ["incident","service_request"]},
    "servicesubcategory_name": {"hint": "Service sub-category name"},
    "org_name"               : {"hint": "Organisation name (exact, as registered in the system)"},
    "impact"                 : {"hint": "1=A department  2=A service  3=A person",
                                 "allowed": ["1","2","3"]},
    "urgency"                : {"hint": "1=Critical  2=High  3=Medium  4=Low",
                                 "allowed": ["1","2","3","4"]},
    "name"                   : {"hint": "Known error name / title"},
    "symptom"                : {"hint": "Observable symptom"},
    "workaround"             : {"hint": "Temporary workaround"},
    "root_cause"             : {"hint": "Root cause analysis"},
    "solution"               : {"hint": "Permanent solution"},
    "error_code"             : {"hint": "Error code (if any)"},
    "domain"                 : {"hint": "1=Application  2=Desktop  3=Network  4=Server",
                                 "allowed": ["1","2","3","4",
                                             "Application","Desktop","Network","Server"]},
    "fallback_plan"          : {"hint": "Rollback / fallback plan"},
    "start_date"             : {"hint": "Any date format -> auto-converted to YYYY-MM-DDTHH:MM"},
    "end_date"               : {"hint": "Any date format -> auto-converted to YYYY-MM-DDTHH:MM"},
    "category"               : {"hint": "1=Normal  2=Standard  3=Emergency",
                                 "allowed": ["1","2","3",
                                             "normal","standard","emergency",
                                             "Normal","Standard","Emergency"]},
    "outage"                 : {"hint": "1=No  2=Yes",
                                 "allowed": ["1","2","yes","no","Yes","No"]},
    "change_ref"             : {"hint": "Change reference ID  e.g. R-000087"},
    "comment"                : {"hint": "Comment (optional)"},
    "first_name"             : {"hint": "New user's first name"},
    "last_name"              : {"hint": "New user's last name"},
    "email"                  : {"hint": "New user's email address"},
    "employee_id"            : {"hint": "Employee / staff ID (optional)"},
    "department"             : {"hint": "User's department"},
    "role"                   : {"hint": "Job role / designation"},
    "service_description"    : {"hint": "Detailed description of the service"},
    "service_status"         : {"hint": "active / inactive / draft",
                                 "allowed": ["active","inactive","draft",
                                             "Active","Inactive","Draft"]},
    "slt_name"               : {"hint": "SLT name / title"},
    "slt_priority"           : {"hint": "1=Critical  2=High  3=Medium  4=Low",
                                 "allowed": ["1","2","3","4"]},
    "slt_metric"             : {"hint": "1=Time to own  2=Time to resolve"},
    "slt_value"              : {"hint": "Target value as a number  e.g. 30"},
    "slt_unit"               : {"hint": "minutes / hours / days",
                                 "allowed": ["minutes","hours","days",
                                             "Minutes","Hours","Days"]},
    "slt_request_type"       : {"hint": "incident  or  service_request",
                                 "allowed": ["incident","service_request"]},
    "sla_name"               : {"hint": "SLA name / title"},
    "sla_org_name"           : {"hint": "Organisation name for this SLA"},
    "sla_description"        : {"hint": "Brief description of the SLA"},
    "contract_name"          : {"hint": "Contract title / name"},
    "contract_org_name"      : {"hint": "Organisation name for this contract"},
    "contract_provider_org"  : {"hint": "Provider / vendor organisation name"},
    "contract_start_date"    : {"hint": "Contract start date  e.g. 2026-01-01T00:00"},
    "contract_end_date"      : {"hint": "Contract end date    e.g. 2027-01-01T00:00"},
    "contract_cost"          : {"hint": "Contract cost as a number  e.g. 50000"},
    "contract_currency"      : {"hint": "Currency code  e.g. USD, EUR, GBP, INR"},
    "contract_description"   : {"hint": "Brief description of the contract"},
    "linked_service_name"    : {"hint": "Service this contract is linked to (select from list)"},
    "linked_sla_name"        : {"hint": "SLA linked to this contract (select from list)"},
    "linked_slt_name"        : {"hint": "SLT linked to this SLA (select from list, optional)"},
}

# ---------------------------------------------
# FIELD DISPLAY LABELS
# ---------------------------------------------

FIELD_LABELS: dict = {
    "title"                  : "Title",
    "description"            : "Description",
    "service_name"           : "Service Name",
    "priority"               : "Priority",
    "request_type"           : "Request Type",
    "servicesubcategory_name": "Service Sub-Category",
    "org_name"               : "Organisation",
    "impact"                 : "Impact",
    "urgency"                : "Urgency",
    "name"                   : "Known Error Name",
    "symptom"                : "Symptom",
    "workaround"             : "Workaround",
    "root_cause"             : "Root Cause",
    "solution"               : "Solution",
    "error_code"             : "Error Code",
    "domain"                 : "Domain",
    "fallback_plan"          : "Fallback Plan",
    "start_date"             : "Start Date",
    "end_date"               : "End Date",
    "category"               : "Category",
    "outage"                 : "Outage",
    "change_ref"             : "Change Reference",
    "comment"                : "Comment",
    "first_name"             : "First Name",
    "last_name"              : "Last Name",
    "email"                  : "Email",
    "employee_id"            : "Employee ID",
    "department"             : "Department",
    "role"                   : "Role",
    "service_description"    : "Service Description",
    "service_status"         : "Service Status",
    "slt_name"               : "SLT Name",
    "slt_priority"           : "SLT Priority",
    "slt_metric"             : "SLT Metric",
    "slt_value"              : "SLT Target Value",
    "slt_unit"               : "SLT Unit",
    "slt_request_type"       : "SLT Request Type",
    "sla_name"               : "SLA Name",
    "sla_org_name"           : "SLA Organisation",
    "sla_description"        : "SLA Description",
    "linked_slt_name"        : "Linked SLT",
    "contract_name"          : "Contract Name",
    "contract_org_name"      : "Contract Organisation",
    "contract_provider_org"  : "Provider Organisation",
    "contract_start_date"    : "Contract Start Date",
    "contract_end_date"      : "Contract End Date",
    "contract_cost"          : "Contract Cost",
    "contract_currency"      : "Contract Currency",
    "contract_description"   : "Contract Description",
    "linked_service_name"    : "Linked Service",
    "linked_sla_name"        : "Linked SLA",
}

def _field_label(field: str) -> str:
    return FIELD_LABELS.get(field, field.replace("_", " ").title())

# ---------------------------------------------
# NUMBERED-LIST HELPER
# ---------------------------------------------

def _make_numbered_list_msg(field: str, options_with_labels: list) -> str:
    label = _field_label(field)
    lines = [f"Please select a {label}:\n"]
    for i, (display, _) in enumerate(options_with_labels, 1):
        lines.append(f"{i}. {display}")
    lines.append("\nType the number or the exact name.")
    return "\n".join(lines)


async def _resolve_numbered_option(
    ctx: "ConnCtx",
    value: str,
    options_with_labels: list,
    field: str,
    session_id: str = "",
) -> Optional[str]:
    v = value.strip()

    if len(options_with_labels) == 1:
        display, api_val = options_with_labels[0]
        await send_reply(ctx, f"[OK] Auto-selected: {display}", session_id=session_id)
        return api_val

    if v.isdigit():
        idx = int(v) - 1
        if 0 <= idx < len(options_with_labels):
            display, api_val = options_with_labels[idx]
            await send_reply(ctx, f"[OK] Selected: {display}", session_id=session_id)
            return api_val
        await send_error(ctx, f"Invalid number. Please enter 1-{len(options_with_labels)}.", session_id=session_id)
        return None

    lower = v.lower()
    matches = [(d, a) for d, a in options_with_labels if d.lower().startswith(lower)]
    if len(matches) == 1:
        display, api_val = matches[0]
        await send_reply(ctx, f"[OK] Selected: {display}", session_id=session_id)
        return api_val
    if len(matches) > 1:
        await send_error(ctx,
            f"Ambiguous - did you mean: {', '.join(d for d, _ in matches)}? "
            "Please be more specific.", session_id=session_id)
        return None

    await send_error(ctx,
        f"Invalid value '{v}'. "
        f"Please enter a number (1-{len(options_with_labels)}) "
        f"or one of: {', '.join(d for d, _ in options_with_labels)}.",
        session_id=session_id)
    return None

# ---------------------------------------------
# INTENT SCHEMAS
# ---------------------------------------------

INTENT_SCHEMAS = {
    "create_ticket": {
        "label"   : "Helpdesk Ticket",
        "required": ["title","description","priority"],
        "optional": ["service_name","request_type","servicesubcategory_name","org_name"],
    },
    "create_problem": {
        "label"   : "Problem Management",
        "required": ["title","description","impact","urgency"],
        "optional": ["org_name","service_name"],
    },
    "create_known_error": {
        "label"   : "Known Error Record",
        "required": ["name","symptom"],
        "optional": ["workaround","root_cause","solution","error_code","domain","org_name"],
    },
    "create_change": {
        "label"   : "Change Management",
        "required": ["title","description","start_date","end_date","fallback_plan"],
        "optional": ["category","outage"],
    },
    "approve_change": {
        "label"   : "Approve Change",
        "required": ["change_ref"],
        "optional": ["comment"],
    },
    "create_service": {
        "label"   : "Create Service",
        "required": ["service_name","service_description"],
        "optional": ["service_status","org_name"],
    },
    "create_slt": {
        "label"   : "Create SLT",
        "required": ["slt_name","slt_priority","slt_metric","slt_value","slt_unit","slt_request_type"],
        "optional": [],
    },
    "create_sla": {
        "label"   : "Create SLA",
        "required": ["sla_name","sla_description"],
        "optional": ["sla_org_name","linked_slt_name"],
    },
    "create_contract": {
        "label"   : "Create Service Contract",
        "required": ["contract_name","contract_org_name","contract_provider_org",
                     "linked_service_name","linked_sla_name",
                     "contract_start_date","contract_end_date",
                     "contract_cost","contract_currency","contract_description"],
        "optional": [],
    },
}

TICKET_SUB_INTENT_SCHEMAS = {
    "helpdesk": {
        "label"   : "Helpdesk Ticket",
        "required": ["title","description","priority"],
        "optional": ["service_name","request_type","servicesubcategory_name","org_name"],
    },
    "provision": {
        "label"   : "New Provision User Ticket",
        "required": ["first_name","last_name","email","department","role"],
        "optional": ["employee_id","org_name","service_name"],
    },
    "known_error": {
        "label"   : "Known Error Helpdesk Ticket",
        "required": ["name","symptom","priority"],
        "optional": ["error_code","workaround","org_name","service_name"],
    },
}

# ---------------------------------------------
# ERROR -> FIELD MAPPING
# ---------------------------------------------

ERROR_FIELD_HINTS = [
    ("org_id",             ["org_name","sla_org_name","contract_org_name"]),
    ("organization",       ["org_name","sla_org_name","contract_org_name"]),
    ("service_id",         ["linked_service_name","service_name"]),
    ("servicesubcategory", ["servicesubcategory_name"]),
    ("title",              ["title"]),
    ("description",        ["description","service_description","sla_description",
                             "contract_description"]),
    ("priority",           ["priority","slt_priority"]),
    ("request_type",       ["request_type","slt_request_type"]),
    ("impact",             ["impact"]),
    ("urgency",            ["urgency"]),
    ("category",           ["category"]),
    ("start_date",         ["start_date","contract_start_date"]),
    ("end_date",           ["end_date","contract_end_date"]),
    ("change_ref",         ["change_ref"]),
    ("first_name",         ["first_name"]),
    ("last_name",          ["last_name"]),
    ("email",              ["email"]),
    ("department",         ["department"]),
    ("role",               ["role"]),
    ("name",               ["name","sla_name","slt_name","contract_name"]),
    ("symptom",            ["symptom"]),
    ("sla",                ["linked_sla_name","sla_name"]),
    ("slt",                ["linked_slt_name","slt_name"]),
    ("service_name",       ["service_name","linked_service_name"]),
    ("metric",             ["slt_metric"]),
    ("value",              ["slt_value","contract_cost"]),
    ("unit",               ["slt_unit"]),
    ("cost",               ["contract_cost"]),
    ("currency",           ["contract_currency"]),
    ("provider",           ["contract_provider_org"]),
    ("cannot be empty",    []),
    ("null",               []),
]

def _fields_from_error(error_text, all_fields):
    lower, culprits, seen = error_text.lower(), [], set()
    for kw, fields in ERROR_FIELD_HINTS:
        if kw in lower:
            for f in fields:
                if f in all_fields and f not in seen:
                    culprits.append(f)
                    seen.add(f)
    return culprits if culprits else all_fields

# ---------------------------------------------
# DEBUG LOG HELPER
# ---------------------------------------------

def _log(event: str, addr: str, session_id: str = "", session_name: str = "", extra: str = ""):
    ts      = datetime.now().strftime("%H:%M:%S")
    sid_tag = ""
    if session_id:
        name_part = f" | '{session_name}'" if session_name else ""
        sid_tag   = f"  [session: {session_id}{name_part}]"
    extra_part = f"  {extra}" if extra else ""
    print(f"[{ts}] {event:<10} {addr}{sid_tag}{extra_part}", flush=True)

# ---------------------------------------------
# SESSION
# ---------------------------------------------

class Session:
    def __init__(self, name: str):
        self.session_id   : str                    = str(uuid.uuid4())
        self.name         : str                    = name.strip() or "Session"
        self.created_at   : datetime               = datetime.now()
        self.input_queue  : asyncio.Queue          = asyncio.Queue()
        self.active_task  : Optional[asyncio.Task] = None
        self.history      : list                   = []

    def to_dict(self) -> dict:
        return {
            "session_id" : self.session_id,
            "name"       : self.name,
            "created_at" : self.created_at.isoformat(),
            "msg_count"  : len(self.history),
        }

    def add_history(self, role: str, text: str):
        self.history.append({"role": role, "text": text, "ts": datetime.now().isoformat()})

    def is_busy(self) -> bool:
        return self.active_task is not None and not self.active_task.done()

    def cancel_task(self):
        if self.active_task and not self.active_task.done():
            self.active_task.cancel()

# ---------------------------------------------
# CONNECTION CONTEXT
# ---------------------------------------------

class ConnCtx:
    def __init__(self, ws: WebSocketServerProtocol, token: str):
        self.ws              = ws
        self.token           = token
        self.addr            = str(ws.remote_address)
        self._sessions       : dict[str, Session] = {}
        self._active_session : Optional[Session]  = None
        self._legacy_queue   : asyncio.Queue      = asyncio.Queue()

    def create_session(self, name: str = "Default") -> Session:
        s = Session(name)
        self._sessions[s.session_id] = s
        return s

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        s = self._sessions.pop(session_id, None)
        if s:
            s.cancel_task()
            if self._active_session and self._active_session.session_id == session_id:
                remaining = sorted(self._sessions.values(), key=lambda x: x.created_at, reverse=True)
                self._active_session = remaining[0] if remaining else None
            return True
        return False

    def set_active(self, session_id: str) -> Optional[Session]:
        s = self._sessions.get(session_id)
        if s:
            self._active_session = s
        return s

    def active_session(self) -> Optional[Session]:
        return self._active_session

    def list_sessions(self) -> list:
        return sorted(self._sessions.values(), key=lambda s: s.created_at)

    def resolve_session(self, session_id: str = "") -> Optional[Session]:
        if session_id:
            return self._sessions.get(session_id)
        return self._active_session

    @property
    def input_queue(self) -> asyncio.Queue:
        s = self._active_session
        return s.input_queue if s else self._legacy_queue

# ---------------------------------------------
# WEBSOCKET SEND HELPERS
# ---------------------------------------------

def _sanitize_dict(obj: dict) -> dict:
    clean = {}
    for k, v in obj.items():
        if v is None:
            clean[k] = ""
        else:
            clean[k] = v
    return clean

async def _send(ctx: ConnCtx, obj: dict, session_id: str = ""):
    obj = _sanitize_dict(obj)
    if session_id:
        obj["session_id"] = session_id
    elif ctx._active_session:
        obj["session_id"] = ctx._active_session.session_id
    for key in ("text", "hint", "reply", "error"):
        if key in obj and isinstance(obj[key], str):
            obj[key] = scrub_brands(obj[key])
    try:
        await ctx.ws.send(json.dumps(obj))
    except websockets.exceptions.ConnectionClosed:
        pass

async def send_status(ctx, text, session_id=""):
    await _send(ctx, {"type": "status", "text": text or ""}, session_id)

async def send_error(ctx, text, session_id=""):
    await _send(ctx, {"type": "error", "text": text or ""}, session_id)

async def send_reply(ctx, text, session_id=""):
    await _send(ctx, {"type": "reply", "text": scrub_brands(text or "")}, session_id)

async def send_intent(ctx, intent, sub_intent="", session_id=""):
    schema     = INTENT_SCHEMAS.get(intent, {})
    base_label = schema.get("label", intent)
    sub_schema = TICKET_SUB_INTENT_SCHEMAS.get(sub_intent, {})
    label      = sub_schema.get("label", base_label) if sub_intent else base_label
    await _send(ctx, {"type": "intent", "intent": intent or "",
                      "sub_intent": sub_intent or "", "label": label or ""}, session_id)

async def send_params(ctx, params, label="params", session_id=""):
    await _send(ctx, {"type": "params", "label": label or "params", "params": params or {}}, session_id)

async def send_api_call(ctx, method, url, payload, session_id=""):
    await _send(ctx, {"type": "api_call", "method": method or "",
                      "url": url or "", "payload": payload or {}}, session_id)

async def send_api_resp(ctx, status, data, session_id=""):
    await _send(ctx, {"type": "api_resp", "status": status, "data": data}, session_id)

async def send_input_req(ctx: ConnCtx, field: str, required: bool = False,
                         options: list | None = None,
                         suppress_hint: bool = False, session_id: str = ""):
    if suppress_hint:
        payload: dict = {
            "type"    : "input_req",
            "key"     : field,
            "suppress": True,
            "required": required,
        }
        if options:
            payload["options"] = options
    else:
        meta = FIELD_META.get(field, {})
        payload = {
            "type"    : "input_req",
            "field"   : field,
            "hint"    : meta.get("hint", ""),
            "label"   : _field_label(field),
            "suppress": False,
            "allowed" : meta.get("allowed", []),
            "required": required,
        }
        if options:
            payload["options"] = options
    await _send(ctx, payload, session_id)

async def send_date_fix(ctx, field, from_val, to_val, session_id=""):
    await _send(ctx, {"type": "date_fix", "field": field or "",
                      "from": from_val or "", "to": to_val or ""}, session_id)

async def send_retry(ctx, attempt, max_retries, error_text, fields, session_id=""):
    await _send(ctx, {"type": "retry", "attempt": attempt,
                      "max": max_retries,
                      "error": scrub_brands(error_text or ""),
                      "fields": fields or []}, session_id)

async def send_list_as_reply(ctx: ConnCtx, field: str, options: list, session_id: str = ""):
    label_map = {
        "org_name"             : "Organisation",
        "sla_org_name"         : "Organisation",
        "contract_org_name"    : "Organisation",
        "contract_provider_org": "Provider Organisation",
        "service_name"         : "Service",
        "linked_service_name"  : "Service",
        "linked_sla_name"      : "SLA",
        "linked_slt_name"      : "SLT",
    }
    label = label_map.get(field, _field_label(field))
    lines = [f"Please select a {label} from the list below:\n"]
    for i, name in enumerate(options, 1):
        lines.append(f"{i}. {name}")
    lines.append("\nType the number or the exact name.")
    await send_reply(ctx, "\n".join(lines), session_id=session_id)

# ---------------------------------------------
# SESSION SEND HELPERS
# ---------------------------------------------

async def send_session_created(ctx: ConnCtx, session: Session, active: bool = True):
    d = session.to_dict()
    d["type"]   = "session_created"
    d["active"] = active
    await _send(ctx, d, session_id=session.session_id)

async def send_session_switched(ctx: ConnCtx, session: Session):
    d = session.to_dict()
    d["type"] = "session_switched"
    await _send(ctx, d, session_id=session.session_id)

async def send_session_list(ctx: ConnCtx):
    sessions = [s.to_dict() for s in ctx.list_sessions()]
    active   = ctx.active_session()
    await _send(ctx, {
        "type"             : "session_list",
        "sessions"         : sessions,
        "active_session_id": active.session_id if active else "",
    })

async def send_session_deleted(ctx: ConnCtx, session_id: str):
    active = ctx.active_session()
    await _send(ctx, {
        "type"             : "session_deleted",
        "session_id"       : session_id,
        "active_session_id": active.session_id if active else "",
    })

async def send_session_renamed(ctx: ConnCtx, session: Session):
    d = session.to_dict()
    d["type"] = "session_renamed"
    await _send(ctx, d, session_id=session.session_id)

async def send_session_error(ctx: ConnCtx, text: str):
    await _send(ctx, {"type": "session_error", "text": text})

# ---------------------------------------------
# FIELD LABEL ANNOUNCEMENT HELPER
# ---------------------------------------------

async def send_field_label(ctx: ConnCtx, field: str, required: bool,
                           include_hint: bool = False,
                           session_id: str = "") -> None:
    label  = _field_label(field)
    bullet = "*" if required else "o"
    suffix = "" if required else "  (optional)"
    if include_hint:
        meta = FIELD_META.get(field, {})
        hint = meta.get("hint", "")
        if hint:
            await send_reply(ctx, f"{bullet} {label}:  {hint}{suffix}", session_id=session_id)
            return
    await send_reply(ctx, f"{bullet} {label}:{suffix}", session_id=session_id)

# ---------------------------------------------
# GET HELPERS
# ---------------------------------------------

async def _get_organizations(ctx: ConnCtx) -> list:
    url = f"{BACKEND_BASE}/api/data-admin/organizations"
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            r = await client.get(url, headers={
                "Authorization": f"Bearer {ctx.token}",
                "accept"       : "application/json",
            })
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    for key in ("organizations", "data", "results", "items"):
                        if isinstance(data.get(key), list):
                            return data[key]
            return []
        except httpx.HTTPError:
            return []

async def _get_services(ctx: ConnCtx) -> list:
    url = f"{BACKEND_BASE}/api/services/list"
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            r = await client.get(url, headers={
                "Authorization": f"Bearer {ctx.token}",
                "accept"       : "application/json",
            })
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    for key in ("services", "data", "results", "items"):
                        if isinstance(data.get(key), list):
                            return data[key]
            return []
        except httpx.HTTPError:
            return []

async def _get_slas(ctx: ConnCtx) -> list:
    url = f"{BACKEND_BASE}/api/services/sla-details"
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            r = await client.get(url, headers={
                "Authorization": f"Bearer {ctx.token}",
                "accept"       : "application/json",
            })
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    for key in ("slas", "data", "results", "items"):
                        if isinstance(data.get(key), list):
                            return data[key]
            return []
        except httpx.HTTPError:
            return []

async def _get_slts(ctx: ConnCtx) -> list:
    url = f"{BACKEND_BASE}/api/services/slts"
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            r = await client.get(url, headers={
                "Authorization": f"Bearer {ctx.token}",
                "accept"       : "application/json",
            })
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    for key in ("slts", "data", "results", "items"):
                        if isinstance(data.get(key), list):
                            return data[key]
            return []
        except httpx.HTTPError:
            return []

def _extract_names(items: list, *name_keys) -> list:
    names = []
    default_keys = name_keys or (
        "name","org_name","service_name","organization_name","title",
        "sla_name","slt_name","contract_name",
    )
    for item in items:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            for k in default_keys:
                if item.get(k):
                    names.append(str(item[k]))
                    break
            else:
                for v in item.values():
                    if isinstance(v, str) and v.strip():
                        names.append(v)
                        break
    return names

def _find_id_by_name(items: list, chosen_name: str, *id_keys) -> Optional[str]:
    default_id_keys = id_keys or ("id", "uuid", "service_id", "sla_id", "slt_id")
    name_keys = ("name","service_name","sla_name","slt_name","org_name","title")
    for item in items:
        if not isinstance(item, dict):
            continue
        matched = False
        for nk in name_keys:
            if item.get(nk, "").strip().lower() == chosen_name.strip().lower():
                matched = True
                break
        if matched:
            for ik in default_id_keys:
                if item.get(ik):
                    return str(item[ik])
    return None

# ---------------------------------------------
# LIVE-LIST FIELDS
# ---------------------------------------------

_LIVE_LIST_FIELDS = {
    "org_name", "sla_org_name", "contract_org_name",
    "contract_provider_org",
    "service_name",
    "linked_service_name",
    "linked_sla_name", "linked_slt_name",
}

# ---------------------------------------------
# WAIT FOR CLIENT INPUT
# ---------------------------------------------

async def wait_for_input(ctx: ConnCtx, field: str, required: bool,
                         options: list | None = None,
                         current_intent: str = "",
                         session_id: str = "") -> str:
    meta    = FIELD_META.get(field, {})
    timeout = 300.0 if required else 60.0

    force_plain = (field == "service_name" and current_intent == "create_service")

    if not force_plain:
        if field in ("org_name", "sla_org_name") and options is None:
            raw = await _get_organizations(ctx)
            options = _extract_names(raw) if isinstance(raw, list) else []

        elif field in ("contract_org_name", "contract_provider_org") and options is None:
            raw = await _get_organizations(ctx)
            options = _extract_names(raw) if isinstance(raw, list) else []

        elif field == "service_name" and options is None:
            raw = await _get_services(ctx)
            options = _extract_names(raw) if isinstance(raw, list) else []

        elif field == "linked_service_name" and options is None:
            raw = await _get_services(ctx)
            options = _extract_names(raw) if isinstance(raw, list) else []

        elif field == "linked_sla_name" and options is None:
            raw = await _get_slas(ctx)
            options = _extract_names(raw, "sla_name", "name", "title") if isinstance(raw, list) else []

        elif field == "linked_slt_name" and options is None:
            raw = await _get_slts(ctx)
            options = _extract_names(raw, "slt_name", "name", "title") if isinstance(raw, list) else []

    numbered_opts: Optional[list] = None
    if field == "priority":
        numbered_opts = PRIORITY_OPTIONS
    elif field == "urgency":
        numbered_opts = URGENCY_OPTIONS
    elif field == "impact":
        numbered_opts = IMPACT_OPTIONS
    elif field == "domain":
        numbered_opts = DOMAIN_OPTIONS
    elif field == "category":
        numbered_opts = CATEGORY_OPTIONS
    elif field == "outage":
        numbered_opts = OUTAGE_OPTIONS
    elif field == "service_status":
        numbered_opts = SERVICE_STATUS_OPTIONS
    elif field == "slt_priority":
        numbered_opts = SLT_PRIORITY_OPTIONS
    elif field == "slt_unit":
        numbered_opts = SLT_UNIT_OPTIONS
    elif field in ("slt_request_type", "request_type"):
        numbered_opts = SLT_REQUEST_TYPE_OPTIONS
    elif field == "contract_currency":
        numbered_opts = CONTRACT_CURRENCY_OPTIONS
    elif field == "slt_metric":
        numbered_opts = SLT_METRIC_OPTIONS

    if field == "service_name" and not force_plain and options is not None and len(options) == 0:
        force_plain = True

    is_live_list = bool(options) and not force_plain
    is_numbered  = numbered_opts is not None
    is_plain     = not is_numbered and not is_live_list

    if is_live_list and options and len(options) == 1:
        chosen = options[0]
        await send_field_label(ctx, field, required, include_hint=False, session_id=session_id)
        await send_reply(ctx, f"[OK] Auto-selected only available option: {chosen}", session_id=session_id)
        return chosen

    if is_plain:
        await send_field_label(ctx, field, required, include_hint=True, session_id=session_id)
        await send_input_req(ctx, field, required, suppress_hint=True, session_id=session_id)

    elif is_numbered:
        await send_field_label(ctx, field, required, include_hint=False, session_id=session_id)
        await send_reply(ctx, _make_numbered_list_msg(field, numbered_opts), session_id=session_id)
        await send_input_req(ctx, field, required, suppress_hint=True, session_id=session_id)

    else:
        await send_field_label(ctx, field, required, include_hint=False, session_id=session_id)
        await send_list_as_reply(ctx, field, options, session_id=session_id)
        await send_input_req(ctx, field, required, options=options, suppress_hint=True, session_id=session_id)

    while True:
        try:
            msg = await asyncio.wait_for(ctx.input_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            if required:
                await send_error(ctx,
                    f"Timed out waiting for required field '{field}'. "
                    "Please re-send your request.", session_id=session_id)
                raise
            return ""

        value = str(msg.get("value", "")).strip()

        if is_numbered:
            if not value:
                if required:
                    await send_error(ctx, f"'{_field_label(field)}' is required.", session_id=session_id)
                    await send_input_req(ctx, field, required, suppress_hint=True, session_id=session_id)
                    continue
                return ""
            resolved = await _resolve_numbered_option(ctx, value, numbered_opts, field, session_id=session_id)
            if resolved is None:
                await send_input_req(ctx, field, required, suppress_hint=True, session_id=session_id)
                continue
            return resolved

        if is_live_list and value:
            if len(options) == 1:
                resolved = options[0]
                await send_reply(ctx, f"[OK] Auto-selected: {resolved}", session_id=session_id)
                return resolved

            if value.isdigit():
                idx = int(value) - 1
                if 0 <= idx < len(options):
                    resolved = options[idx]
                    await send_reply(ctx, f"[OK] Selected: {resolved}", session_id=session_id)
                    value = resolved
                else:
                    await send_error(ctx, f"Invalid number. Please enter 1-{len(options)}.", session_id=session_id)
                    await send_input_req(ctx, field, required,
                                         options=options, suppress_hint=True, session_id=session_id)
                    continue
            else:
                lower   = value.lower()
                matches = [o for o in options if o.lower().startswith(lower)]
                if len(matches) == 1:
                    resolved = matches[0]
                    await send_reply(ctx, f"[OK] Selected: {resolved}", session_id=session_id)
                    value = resolved
                elif len(matches) > 1:
                    await send_error(ctx,
                        f"Ambiguous - did you mean: {', '.join(matches)}? "
                        "Please be more specific.", session_id=session_id)
                    await send_input_req(ctx, field, required,
                                         options=options, suppress_hint=True, session_id=session_id)
                    continue

        if field in ("start_date", "end_date",
                     "contract_start_date", "contract_end_date") and value:
            normalised = normalize_date(value)
            if normalised is None:
                await send_error(ctx,
                    f"Cannot parse date '{value}'. Example: 2026-01-20T11:20 or 20-01-2026",
                    session_id=session_id)
                await send_input_req(ctx, field, required, suppress_hint=True, session_id=session_id)
                continue
            if normalised != value:
                await send_reply(ctx, f"[OK] Date auto-corrected: {value} -> {normalised}", session_id=session_id)
                await send_date_fix(ctx, field, value, normalised, session_id=session_id)
            value = normalised

        allowed = meta.get("allowed", [])
        if allowed and value and value not in allowed:
            await send_error(ctx,
                f"Invalid value '{value}'. Allowed: {', '.join(allowed)}",
                session_id=session_id)
            await send_input_req(ctx, field, required, suppress_hint=True, session_id=session_id)
            continue

        if required and not value:
            await send_error(ctx, f"'{_field_label(field)}' is required.", session_id=session_id)
            await send_input_req(ctx, field, required, suppress_hint=True, session_id=session_id)
            continue

        return value

# ---------------------------------------------
# DATE NORMALISE ON PARAMS DICT (SILENT)
# ---------------------------------------------

async def normalize_params_dates(ctx, params: dict, session_id: str = "") -> dict:
    date_fields = ("start_date", "end_date",
                   "contract_start_date", "contract_end_date")
    for field in date_fields:
        raw = params.get(field, "")
        if not raw:
            continue
        normalised = normalize_date(raw)
        if normalised is None:
            await send_reply(ctx, f"[!] Could not parse date for '{field}': '{raw}' - will re-prompt.", session_id=session_id)
            params[field] = ""
        elif normalised != raw:
            await send_date_fix(ctx, field, raw, normalised, session_id=session_id)
            await send_reply(ctx, f"[OK] Date auto-corrected: {raw} -> {normalised}", session_id=session_id)
            params[field] = normalised
    return params

# ---------------------------------------------
# PROMPT MISSING PARAMS
# ---------------------------------------------

async def prompt_missing_params(ctx, params, required_fields, optional_fields,
                                current_intent: str = "", session_id: str = ""):
    for f in required_fields:
        if not params.get(f, "").strip():
            params[f] = await wait_for_input(ctx, f, required=True,
                                              current_intent=current_intent,
                                              session_id=session_id)
    for f in optional_fields:
        if not params.get(f, "").strip():
            params[f] = await wait_for_input(ctx, f, required=False,
                                              current_intent=current_intent,
                                              session_id=session_id)
    return params

# ---------------------------------------------
# LOCAL REPLY BUILDER
# ---------------------------------------------

def _build_local_reply(intent: str, status: int, data: dict | str,
                        params: dict) -> str:
    ok = str(status).startswith("2")

    ref_id = ""
    if isinstance(data, dict):
        for key in ("id", "uuid", "ticket_id", "ref", "reference",
                    "change_ref", "problem_id", "service_id", "sla_id",
                    "slt_id", "contract_id", "name"):
            val = data.get(key)
            if val:
                ref_id = str(val)
                break

    if ok:
        label_map = {
            "create_ticket"      : "Helpdesk ticket",
            "create_problem"     : "Problem record",
            "create_known_error" : "Known Error record",
            "create_change"      : "Change request",
            "approve_change"     : "Change",
            "create_service"     : "Service",
            "create_slt"         : "Service Level Target (SLT)",
            "create_sla"         : "Service Level Agreement (SLA)",
            "create_contract"    : "Service Contract",
        }
        label = label_map.get(intent, "Record")
        ref_part = f" (Reference: {ref_id})" if ref_id else ""
        return f"[OK] {label} created successfully{ref_part}."
    else:
        err_text = ""
        if isinstance(data, dict):
            err_text = data.get("detail", data.get("message", json.dumps(data)))
        else:
            err_text = str(data)
        return f"[FAIL] Operation failed (HTTP {status}): {scrub_brands(err_text)}"


# ---------------------------------------------
# PAYLOAD BUILDERS
# ---------------------------------------------

def _build_ticket(p):
    return {
        "title"                  : p.get("title",""),
        "description"            : p.get("description",""),
        "service_name"           : p.get("service_name",""),
        "priority"               : p.get("priority","3"),
        "request_type"           : p.get("request_type","incident"),
        "org_name"               : p.get("org_name",""),
        "servicesubcategory_name": p.get("servicesubcategory_name",""),
    }

def _build_provision_ticket(p):
    full_name  = f"{p.get('first_name','').strip()} {p.get('last_name','').strip()}".strip()
    emp_id     = p.get("employee_id","").strip()
    auto_title = f"New User Provisioning - {full_name}"
    auto_desc  = (
        f"Provision new user account.\n"
        f"Name       : {full_name}\n"
        f"Email      : {p.get('email','')}\n"
        f"Department : {p.get('department','')}\n"
        f"Role       : {p.get('role','')}"
        + (f"\nEmployee ID: {emp_id}" if emp_id else "")
    )
    return {
        "title"                  : auto_title,
        "description"            : auto_desc,
        "service_name"           : p.get("service_name",""),
        "priority"               : "3",
        "request_type"           : "service_request",
        "org_name"               : p.get("org_name",""),
        "servicesubcategory_name": "",
    }

def _build_known_error_ticket(p):
    name       = p.get("name","").strip()
    symptom    = p.get("symptom","").strip()
    workaround = p.get("workaround","").strip()
    error_code = p.get("error_code","").strip()
    auto_title = f"[Known Error] {name}" if name else "[Known Error] Ticket"
    auto_desc  = f"Known Error ticket raised.\nError Name: {name}\nSymptom   : {symptom}"
    if error_code:
        auto_desc += f"\nError Code: {error_code}"
    if workaround:
        auto_desc += f"\nWorkaround: {workaround}"
    return {
        "title"                  : auto_title,
        "description"            : auto_desc,
        "service_name"           : p.get("service_name",""),
        "priority"               : p.get("priority","3"),
        "request_type"           : "incident",
        "org_name"               : p.get("org_name",""),
        "servicesubcategory_name": "",
    }

def _build_problem(p):
    return {
        "title"       : p.get("title",""),
        "description" : p.get("description",""),
        "org_name"    : p.get("org_name",""),
        "service_name": p.get("service_name",""),
        "impact"      : p.get("impact","3"),
        "urgency"     : p.get("urgency","3"),
    }

def _build_known_error(p):
    return {
        "name"      : p.get("name",""),
        "symptom"   : p.get("symptom",""),
        "workaround": p.get("workaround",""),
        "root_cause": p.get("root_cause",""),
        "solution"  : p.get("solution",""),
        "error_code": p.get("error_code",""),
        "domain"    : p.get("domain",""),
        "org_name"  : p.get("org_name",""),
    }

def _build_change(p):
    return {
        "title"        : p.get("title",""),
        "description"  : p.get("description",""),
        "fallback_plan": p.get("fallback_plan",""),
        "category"     : p.get("category","normal"),
        "outage"       : normalize_outage(p.get("outage","no")),
        "start_date"   : p.get("start_date",""),
        "end_date"     : p.get("end_date",""),
    }

def _build_approve(p):
    return {"comment": p.get("comment","")}

def _build_service(p):
    return {
        "name"       : p.get("service_name",""),
        "org_name"   : p.get("org_name",""),
        "description": p.get("service_description",""),
        "status"     : p.get("service_status","active"),
    }

def _build_slt(p):
    raw_value = p.get("slt_value", "0")
    try:
        value_int = int(float(raw_value))
    except (ValueError, TypeError):
        value_int = 0
    return {
        "name"        : p.get("slt_name",""),
        "priority"    : p.get("slt_priority","3"),
        "metric"      : p.get("slt_metric",""),
        "value"       : value_int,
        "unit"        : p.get("slt_unit","minutes"),
        "request_type": p.get("slt_request_type","incident"),
    }

def _build_sla(p):
    slt_id  = p.get("_slt_id","").strip()
    slt_ids = [slt_id] if slt_id else []
    return {
        "name"       : p.get("sla_name",""),
        "org_name"   : p.get("sla_org_name",""),
        "description": p.get("sla_description",""),
        "slt_ids"    : slt_ids,
    }

def _build_contract(p):
    raw_cost = p.get("contract_cost","0")
    try:
        cost_float = float(raw_cost)
    except (ValueError, TypeError):
        cost_float = 0.0

    start = p.get("contract_start_date","") or p.get("start_date","")
    end   = p.get("contract_end_date","")   or p.get("end_date","")

    return {
        "name"             : p.get("contract_name",""),
        "org_name"         : p.get("contract_org_name",""),
        "provider_org_name": p.get("contract_provider_org",""),
        "service_id"       : p.get("_service_id",""),
        "sla_id"           : p.get("_sla_id",""),
        "start_date"       : start,
        "end_date"         : end,
        "cost"             : cost_float,
        "cost_currency"    : p.get("contract_currency",""),
        "description"      : p.get("contract_description",""),
    }

# ---------------------------------------------
# GENERIC POST
# ---------------------------------------------

async def _post(ctx: ConnCtx, url, payload, session_id: str = ""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] API_REQ    {url}", flush=True)
    await send_api_call(ctx, "POST", url, payload, session_id=session_id)
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            r = await client.post(url, headers={
                "Authorization": f"Bearer {ctx.token}",
                "Content-Type" : "application/json",
                "accept"       : "application/json",
            }, json=payload)
            is_json = "application/json" in r.headers.get("content-type","")
            data    = r.json() if is_json else r.text
            print(f"[{ts}] API_RESP   {r.status_code} from {url}", flush=True)
            await send_api_resp(ctx, r.status_code, data, session_id=session_id)
            return r.status_code, data
        except httpx.HTTPError as e:
            print(f"[{ts}] API_ERROR  {e} for {url}", flush=True)
            await send_error(ctx, f"HTTP error: {e}", session_id=session_id)
            return 0, str(e)

# ---------------------------------------------
# RETRY WRAPPER
# ---------------------------------------------

async def _post_with_retry(ctx: ConnCtx, url, build_fn, params, all_fields,
                           current_intent: str = "", session_id: str = ""):
    for attempt in range(1, MAX_RETRIES + 2):
        params = await normalize_params_dates(ctx, params, session_id=session_id)
        status, data = await _post(ctx, url, build_fn(params), session_id=session_id)
        if str(status).startswith("2"):
            return status, data, params
        if attempt > MAX_RETRIES:
            break
        err_text       = data.get("detail", json.dumps(data)) if isinstance(data, dict) else str(data)
        culprit_fields = _fields_from_error(err_text, all_fields)
        await send_retry(ctx, attempt, MAX_RETRIES, err_text, culprit_fields, session_id=session_id)
        for f in culprit_fields:
            params[f] = await wait_for_input(ctx, f, required=True,
                                              current_intent=current_intent,
                                              session_id=session_id)
        await send_params(ctx, params, "updated_params", session_id=session_id)
    return status, data, params

# ---------------------------------------------
# LIVE LIST SELECTION HELPERS
# ---------------------------------------------

async def _read_live_list_selection(ctx: ConnCtx, options: list,
                                    required: bool = True,
                                    session_id: str = "") -> str:
    if len(options) == 1:
        await send_reply(ctx, f"[OK] Auto-selected: {options[0]}", session_id=session_id)
        return options[0]

    timeout = 300.0 if required else 60.0
    while True:
        try:
            msg = await asyncio.wait_for(ctx.input_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            if required:
                raise
            return ""
        value = str(msg.get("value","")).strip()
        if not value:
            if required:
                await send_error(ctx, "This field is required.", session_id=session_id)
                continue
            return ""
        if value.isdigit():
            idx = int(value) - 1
            if 0 <= idx < len(options):
                resolved = options[idx]
                await send_reply(ctx, f"[OK] Selected: {resolved}", session_id=session_id)
                return resolved
            await send_error(ctx, f"Invalid number. Please enter 1-{len(options)}.", session_id=session_id)
            continue
        lower   = value.lower()
        matches = [o for o in options if o.lower().startswith(lower)]
        if len(matches) == 1:
            await send_reply(ctx, f"[OK] Selected: {matches[0]}", session_id=session_id)
            return matches[0]
        if len(matches) > 1:
            await send_error(ctx,
                f"Ambiguous - did you mean: {', '.join(matches)}? Please be more specific.",
                session_id=session_id)
            continue
        exact = [o for o in options if o.lower() == lower]
        if exact:
            await send_reply(ctx, f"[OK] Selected: {exact[0]}", session_id=session_id)
            return exact[0]
        await send_error(ctx,
            f"'{value}' not found. Please enter a number or exact name from the list.",
            session_id=session_id)


async def _read_plain_input(ctx: ConnCtx, required: bool = True,
                             timeout: float = 300.0,
                             session_id: str = "") -> str:
    while True:
        try:
            msg = await asyncio.wait_for(ctx.input_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            if required:
                raise
            return ""
        value = str(msg.get("value","")).strip()
        if not value and required:
            await send_error(ctx, "This field is required.", session_id=session_id)
            continue
        return value

# ---------------------------------------------
# SERVICE MANAGEMENT - SPECIAL DISPATCH HELPERS
# ---------------------------------------------

async def _dispatch_create_sla(ctx: ConnCtx, params: dict, session_id: str = ""):
    schema     = INTENT_SCHEMAS["create_sla"]
    req_fields = schema["required"]
    opt_fields = schema["optional"]
    all_fields = req_fields + opt_fields

    for f in all_fields:
        params.setdefault(f, "")

    await send_params(ctx, params, "extracted_params", session_id=session_id)

    for f in req_fields:
        if not params.get(f, "").strip():
            params[f] = await wait_for_input(ctx, f, required=True,
                                              current_intent="create_sla",
                                              session_id=session_id)
    for f in opt_fields:
        if not params.get(f, "").strip():
            params[f] = await wait_for_input(ctx, f, required=False,
                                              current_intent="create_sla",
                                              session_id=session_id)

    await send_reply(ctx,
        "Would you like to:\n"
        "1. Create a new SLT and link it to this SLA\n"
        "2. Link an existing SLT\n"
        "3. Skip SLT (create SLA without linking an SLT)\n\n"
        "Type 1, 2, or 3.", session_id=session_id)
    await send_input_req(ctx, "slt_choice", required=True, suppress_hint=True, session_id=session_id)

    slt_id: Optional[str] = None

    while True:
        try:
            msg = await asyncio.wait_for(ctx.input_queue.get(), timeout=120.0)
        except asyncio.TimeoutError:
            await send_error(ctx, "Timed out. Proceeding without SLT.", session_id=session_id)
            break

        choice = str(msg.get("value","")).strip()

        if choice in ("1", "create", "new"):
            await send_reply(ctx, "Let's create a new SLT. Please provide the following details:", session_id=session_id)
            slt_params: dict = {}
            slt_schema = INTENT_SCHEMAS["create_slt"]
            for f in slt_schema["required"]:
                slt_params[f] = await wait_for_input(ctx, f, required=True,
                                                      current_intent="create_slt",
                                                      session_id=session_id)

            await send_params(ctx, slt_params, "slt_params", session_id=session_id)
            slt_url = f"{BACKEND_BASE}/api/services/slts"
            slt_status, slt_data = await _post(ctx, slt_url, _build_slt(slt_params), session_id=session_id)

            if str(slt_status).startswith("2"):
                if isinstance(slt_data, dict):
                    slt_id = (slt_data.get("id") or slt_data.get("uuid") or
                               slt_data.get("slt_id") or "")
                await send_reply(ctx, f"[OK] SLT created successfully.{(' ID: ' + slt_id) if slt_id else ''}", session_id=session_id)
            else:
                err = slt_data.get("detail", str(slt_data)) if isinstance(slt_data, dict) else str(slt_data)
                await send_error(ctx, f"SLT creation failed: {scrub_brands(err)}. Proceeding without SLT.", session_id=session_id)
            break

        elif choice in ("2", "existing", "link"):
            raw_slts   = await _get_slts(ctx)
            slt_names  = _extract_names(raw_slts, "slt_name","name","title")
            if not slt_names:
                await send_reply(ctx, "No existing SLTs found. Proceeding without SLT.", session_id=session_id)
                break

            if len(slt_names) == 1:
                chosen_name = slt_names[0]
                await send_reply(ctx, f"[OK] Auto-selected only available SLT: {chosen_name}", session_id=session_id)
                slt_id = _find_id_by_name(raw_slts, chosen_name) or chosen_name
                break

            await send_list_as_reply(ctx, "linked_slt_name", slt_names, session_id=session_id)
            await send_input_req(ctx, "linked_slt_name", required=False,
                                  options=slt_names, suppress_hint=True, session_id=session_id)
            try:
                sel_msg = await asyncio.wait_for(ctx.input_queue.get(), timeout=120.0)
            except asyncio.TimeoutError:
                await send_reply(ctx, "Timed out. Proceeding without SLT.", session_id=session_id)
                break
            sel_val = str(sel_msg.get("value","")).strip()
            if sel_val:
                if sel_val.isdigit():
                    idx = int(sel_val) - 1
                    if 0 <= idx < len(slt_names):
                        chosen_name = slt_names[idx]
                        await send_reply(ctx, f"[OK] Selected SLT: {chosen_name}", session_id=session_id)
                        slt_id = _find_id_by_name(raw_slts, chosen_name)
                        if not slt_id:
                            slt_id = chosen_name
                else:
                    matches = [o for o in slt_names if o.lower().startswith(sel_val.lower())]
                    if len(matches) == 1:
                        chosen_name = matches[0]
                        await send_reply(ctx, f"[OK] Selected SLT: {chosen_name}", session_id=session_id)
                        slt_id = _find_id_by_name(raw_slts, chosen_name)
                        if not slt_id:
                            slt_id = chosen_name
            break

        elif choice in ("3", "skip", "none", ""):
            await send_reply(ctx, "Skipping SLT. SLA will be created without a linked SLT.", session_id=session_id)
            break

        else:
            await send_error(ctx, "Please enter 1, 2, or 3.", session_id=session_id)
            await send_input_req(ctx, "slt_choice", required=True, suppress_hint=True, session_id=session_id)
            continue

    params["_slt_id"] = slt_id or ""
    await send_params(ctx, params, "final_params", session_id=session_id)

    sla_url = f"{BACKEND_BASE}/api/services/slas"
    status, data = await _post(ctx, sla_url, _build_sla(params), session_id=session_id)

    if not str(status).startswith("2"):
        for attempt in range(1, MAX_RETRIES + 1):
            err_text = data.get("detail", json.dumps(data)) if isinstance(data, dict) else str(data)
            culprit_fields = _fields_from_error(err_text, all_fields)
            await send_retry(ctx, attempt, MAX_RETRIES, err_text, culprit_fields, session_id=session_id)
            for f in culprit_fields:
                params[f] = await wait_for_input(ctx, f, required=True,
                                                  current_intent="create_sla",
                                                  session_id=session_id)
            await send_params(ctx, params, "updated_params", session_id=session_id)
            status, data = await _post(ctx, sla_url, _build_sla(params), session_id=session_id)
            if str(status).startswith("2"):
                break

    return status, data, params


async def _dispatch_create_contract(ctx: ConnCtx, params: dict, session_id: str = ""):
    schema     = INTENT_SCHEMAS["create_contract"]
    req_fields = schema["required"]
    opt_fields = schema["optional"]
    all_fields = req_fields + opt_fields

    for f in all_fields:
        params.setdefault(f, "")

    for f in ("linked_service_name", "linked_sla_name",
              "contract_provider_org", "contract_org_name"):
        params[f] = ""

    await send_params(ctx, params, "extracted_params", session_id=session_id)

    skip_fields = {"linked_service_name", "linked_sla_name",
                   "contract_provider_org", "contract_org_name"}
    text_req_fields = [f for f in req_fields if f not in skip_fields]
    for f in text_req_fields:
        if not params.get(f, "").strip():
            params[f] = await wait_for_input(ctx, f, required=True,
                                              current_intent="create_contract",
                                              session_id=session_id)

    # Contract Organisation
    raw_orgs = await _get_organizations(ctx)
    org_names = _extract_names(raw_orgs)
    if org_names:
        if len(org_names) == 1:
            chosen = org_names[0]
            await send_field_label(ctx, "contract_org_name", required=True, include_hint=False, session_id=session_id)
            await send_reply(ctx, f"[OK] Auto-selected only available Organisation: {chosen}", session_id=session_id)
            params["contract_org_name"] = chosen
        else:
            await send_field_label(ctx, "contract_org_name", required=True, include_hint=False, session_id=session_id)
            await send_list_as_reply(ctx, "contract_org_name", org_names, session_id=session_id)
            await send_input_req(ctx, "contract_org_name", required=True,
                                  options=org_names, suppress_hint=True, session_id=session_id)
            params["contract_org_name"] = await _read_live_list_selection(
                ctx, org_names, required=True, session_id=session_id)
    else:
        await send_reply(ctx, "[!] No organisations found. Please enter Contract Organisation name manually.", session_id=session_id)
        await send_field_label(ctx, "contract_org_name", required=True, include_hint=True, session_id=session_id)
        await send_input_req(ctx, "contract_org_name", required=True, suppress_hint=True, session_id=session_id)
        params["contract_org_name"] = await _read_plain_input(ctx, required=True, session_id=session_id)

    # Provider Organisation
    raw_orgs_provider = await _get_organizations(ctx)
    provider_org_names = _extract_names(raw_orgs_provider)
    if provider_org_names:
        if len(provider_org_names) == 1:
            chosen = provider_org_names[0]
            await send_field_label(ctx, "contract_provider_org", required=True, include_hint=False, session_id=session_id)
            await send_reply(ctx, f"[OK] Auto-selected only available Provider Organisation: {chosen}", session_id=session_id)
            params["contract_provider_org"] = chosen
        else:
            await send_field_label(ctx, "contract_provider_org", required=True, include_hint=False, session_id=session_id)
            await send_list_as_reply(ctx, "contract_provider_org", provider_org_names, session_id=session_id)
            await send_input_req(ctx, "contract_provider_org", required=True,
                                  options=provider_org_names, suppress_hint=True, session_id=session_id)
            params["contract_provider_org"] = await _read_live_list_selection(
                ctx, provider_org_names, required=True, session_id=session_id)
    else:
        await send_reply(ctx, "[!] No organisations found. Please enter Provider Organisation name manually.", session_id=session_id)
        await send_field_label(ctx, "contract_provider_org", required=True, include_hint=True, session_id=session_id)
        await send_input_req(ctx, "contract_provider_org", required=True, suppress_hint=True, session_id=session_id)
        params["contract_provider_org"] = await _read_plain_input(ctx, required=True, session_id=session_id)

    # Resolve Service
    raw_services  = await _get_services(ctx)
    service_names = _extract_names(raw_services)
    if service_names:
        if len(service_names) == 1:
            chosen = service_names[0]
            await send_field_label(ctx, "linked_service_name", required=True, include_hint=False, session_id=session_id)
            await send_reply(ctx, f"[OK] Auto-selected only available Service: {chosen}", session_id=session_id)
            params["linked_service_name"] = chosen
            params["_service_id"] = _find_id_by_name(raw_services, chosen) or ""
        else:
            await send_field_label(ctx, "linked_service_name", required=True, include_hint=False, session_id=session_id)
            await send_list_as_reply(ctx, "linked_service_name", service_names, session_id=session_id)
            await send_input_req(ctx, "linked_service_name", required=True,
                                  options=service_names, suppress_hint=True, session_id=session_id)
            svc_val = await _read_live_list_selection(ctx, service_names, required=True, session_id=session_id)
            params["linked_service_name"] = svc_val
            params["_service_id"] = _find_id_by_name(raw_services, svc_val) or ""
    else:
        await send_reply(ctx, "[!] No services found. Please enter Service ID manually.", session_id=session_id)
        await send_field_label(ctx, "linked_service_name", required=True, include_hint=True, session_id=session_id)
        await send_input_req(ctx, "linked_service_name", required=True, suppress_hint=True, session_id=session_id)
        raw_in = await _read_plain_input(ctx, required=True, session_id=session_id)
        params["linked_service_name"] = raw_in
        params["_service_id"] = raw_in

    # Resolve SLA
    raw_slas  = await _get_slas(ctx)
    sla_names = _extract_names(raw_slas, "sla_name","name","title")
    if sla_names:
        if len(sla_names) == 1:
            chosen = sla_names[0]
            await send_field_label(ctx, "linked_sla_name", required=True, include_hint=False, session_id=session_id)
            await send_reply(ctx, f"[OK] Auto-selected only available SLA: {chosen}", session_id=session_id)
            params["linked_sla_name"] = chosen
            params["_sla_id"] = _find_id_by_name(raw_slas, chosen) or ""
        else:
            await send_field_label(ctx, "linked_sla_name", required=True, include_hint=False, session_id=session_id)
            await send_list_as_reply(ctx, "linked_sla_name", sla_names, session_id=session_id)
            await send_input_req(ctx, "linked_sla_name", required=True,
                                  options=sla_names, suppress_hint=True, session_id=session_id)
            sla_val = await _read_live_list_selection(ctx, sla_names, required=True, session_id=session_id)
            params["linked_sla_name"] = sla_val
            params["_sla_id"] = _find_id_by_name(raw_slas, sla_val) or ""
    else:
        await send_reply(ctx, "[!] No SLAs found. Please enter SLA ID manually.", session_id=session_id)
        await send_field_label(ctx, "linked_sla_name", required=True, include_hint=True, session_id=session_id)
        await send_input_req(ctx, "linked_sla_name", required=True, suppress_hint=True, session_id=session_id)
        raw_in = await _read_plain_input(ctx, required=True, session_id=session_id)
        params["linked_sla_name"] = raw_in
        params["_sla_id"] = raw_in

    for fld in ("contract_start_date","contract_end_date"):
        raw = params.get(fld,"")
        if raw:
            normalised = normalize_date(raw)
            if normalised and normalised != raw:
                await send_date_fix(ctx, fld, raw, normalised, session_id=session_id)
                await send_reply(ctx, f"[OK] Date auto-corrected: {raw} -> {normalised}", session_id=session_id)
                params[fld] = normalised

    await send_params(ctx, params, "final_params", session_id=session_id)

    url = f"{BACKEND_BASE}/api/services/contracts"
    status, data = await _post(ctx, url, _build_contract(params), session_id=session_id)

    if not str(status).startswith("2"):
        for attempt in range(1, MAX_RETRIES + 1):
            err_text = data.get("detail", json.dumps(data)) if isinstance(data, dict) else str(data)
            culprit_fields = _fields_from_error(err_text, all_fields)
            await send_retry(ctx, attempt, MAX_RETRIES, err_text, culprit_fields, session_id=session_id)
            for f in culprit_fields:
                params[f] = await wait_for_input(ctx, f, required=True,
                                                  current_intent="create_contract",
                                                  session_id=session_id)
            await send_params(ctx, params, "updated_params", session_id=session_id)
            for fld in ("contract_start_date","contract_end_date"):
                raw = params.get(fld,"")
                if raw:
                    n = normalize_date(raw)
                    if n and n != raw:
                        params[fld] = n
            status, data = await _post(ctx, url, _build_contract(params), session_id=session_id)
            if str(status).startswith("2"):
                break

    return status, data, params

# ---------------------------------------------
# DISPATCH
# ---------------------------------------------

async def dispatch(ctx: ConnCtx, intent: str, sub_intent: str, params: dict,
                   session_id: str = ""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] DISPATCH   intent={intent}  sub={sub_intent}", flush=True)
    if intent == "create_sla":
        return await _dispatch_create_sla(ctx, params, session_id=session_id)

    if intent == "create_contract":
        return await _dispatch_create_contract(ctx, params, session_id=session_id)

    if intent == "create_ticket":
        effective_sub = sub_intent if sub_intent in TICKET_SUB_INTENT_SCHEMAS else "helpdesk"
        sub_schema    = TICKET_SUB_INTENT_SCHEMAS[effective_sub]
        req_fields    = sub_schema["required"]
        opt_fields    = sub_schema["optional"]
    else:
        schema     = INTENT_SCHEMAS[intent]
        req_fields = schema["required"]
        opt_fields = schema["optional"]

    all_fields = req_fields + opt_fields
    for f in all_fields:
        params.setdefault(f, "")

    fields_to_clear = []
    if intent == "create_ticket":
        fields_to_clear = ["priority", "service_name"]
    elif intent == "create_problem":
        fields_to_clear = ["impact", "urgency"]
    elif intent == "create_change":
        fields_to_clear = ["category", "outage"]
    elif intent == "create_known_error":
        fields_to_clear = ["domain"]
    elif intent == "create_service":
        fields_to_clear = ["service_status"]
    elif intent == "create_slt":
        fields_to_clear = ["slt_priority", "slt_unit", "slt_request_type", "slt_metric"]

    for clear_field in fields_to_clear:
        if clear_field in all_fields:
            params[clear_field] = ""

    params = await normalize_params_dates(ctx, params, session_id=session_id)
    await send_params(ctx, params, "extracted_params", session_id=session_id)
    params = await prompt_missing_params(ctx, params, req_fields, opt_fields,
                                         current_intent=intent, session_id=session_id)
    params = await normalize_params_dates(ctx, params, session_id=session_id)
    await send_params(ctx, params, "final_params", session_id=session_id)

    if intent == "create_ticket":
        url = f"{BACKEND_BASE}/api/tickets"
        if effective_sub == "provision":
            build = _build_provision_ticket
        elif effective_sub == "known_error":
            build = _build_known_error_ticket
        else:
            build = _build_ticket

    elif intent == "create_problem":
        url, build = f"{BACKEND_BASE}/api/problems", _build_problem

    elif intent == "create_known_error":
        url, build = f"{BACKEND_BASE}/api/known-errors", _build_known_error

    elif intent == "create_change":
        url, build = f"{BACKEND_BASE}/api/changes", _build_change

    elif intent == "approve_change":
        change_ref = params.get("change_ref","").strip()
        if not change_ref:
            await send_error(ctx, "change_ref is empty - cannot approve.", session_id=session_id)
            return 400, {"detail": "change_ref missing"}, params
        url, build = f"{BACKEND_BASE}/api/changes/{change_ref}/approve", _build_approve

    elif intent == "create_service":
        url, build = f"{BACKEND_BASE}/api/services/", _build_service

    elif intent == "create_slt":
        url, build = f"{BACKEND_BASE}/api/services/slts", _build_slt

    else:
        await send_error(ctx, f"Unknown intent: {intent}", session_id=session_id)
        return 400, {}, params

    return await _post_with_retry(ctx, url, build, params, all_fields,
                                   current_intent=intent, session_id=session_id)

# ---------------------------------------------
# LLM SYSTEM PROMPT  (unchanged)
# ---------------------------------------------

SYSTEM_PROMPT = """\
Role: Super Admin ITSM Assistant. Output: JSON only.
Schema: {"intent": str, "sub_intent": str, "params": dict, "reply": str}

GLOBAL RULES:
1. Brand Ban: NEVER use "iTop", "GLPi", "glpi", or "itop". Use "the ITSM system".
2. Identity: "I am an ITSM BOT. I assist with IT Service Management concepts and processes."
3. Greet: "Hello! I am an ITSM BOT. How can I help you with ITSM today?"
4. Capability: "I can help with Helpdesk, Assets, CMDB, Problem, Change, Service, User, and Data Admin."
5. Domain: ONLY ITSM. If off-topic, intent="irrelevant", reply="I can't provide an answer to this question. Please ask something related to ITSM."

INTENT ROUTING:
| Request | Intent | Sub-Intent | Required Params |
| :--- | :--- | :--- | :--- |
| Helpdesk / Incident | create_ticket | helpdesk | title, description |
| Change Management / Change Request / Change Ticket | create_change | | title, description, start_date, end_date, fallback_plan |
| User Provisioning | create_ticket | provision | first_name, last_name, email, department, role |
| Problem / RCA | create_problem | | title, description |
| Approve Change | approve_change | | change_ref |
| Service | create_service | | service_name, service_description |
| SLA | create_sla | | sla_name, sla_description |
| SLT | create_slt | | slt_name, slt_metric, slt_value, slt_unit, slt_request_type |
| Contract | create_contract | | contract_name, contract_org_name, contract_provider_org |
| Known Error Record | create_known_error | | name, symptom |
| Known Error Ticket | create_ticket | known_error | name, symptom |
| Knowledge / How-to | faq | | |

FIELD RULES:
- Use "" for: priority, impact, urgency, category, outage, domain, service_status, slt_priority, slt_unit, slt_request_type, slt_metric, contract_currency, org_name, sla_org_name, contract_org_name, contract_provider_org, service_name, linked_service_name, linked_sla_name, linked_slt_name.
- User will select these from lists/numbered options provided by the system.
- Extract dates exactly; the system normalizes them.

REPLY STYLE:
- Action (No result yet): Confirm module + understanding.
- FAQ: Structured answer from KB (bullets/headings).
- Identity/Greet/Capability/Irrelevant: Use exact phrases from GLOBAL RULES.

ROUTING LOGIC:
- If user mentions "Change", "Change Management", or "Change Ticket" -> Use intent: `create_change`.
- If user mentions "Problem" -> Use intent: `create_problem`.
- If user mentions "SLA", "SLT", or "Contract" -> Use appropriate Service intent.
- Use `create_ticket` ONLY for Helpdesk incidents/requests.
"""

# ---------------------------------------------
# LLM HELPERS
# ---------------------------------------------

def _parse_llm(raw: str) -> dict:
    raw = raw.strip()
    # Strip <think>...</think> tags produced by reasoning models (e.g. qwen3)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    for pat in (r"^```json\s*", r"^```\s*"):
        raw = re.sub(pat, "", raw)
    raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        r = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        r = json.loads(m.group()) if m else {}
    r.setdefault("intent",     "unknown")
    r.setdefault("sub_intent", "")
    r.setdefault("params",     {})
    r.setdefault("reply",      raw)
    r["reply"] = scrub_brands(r.get("reply", "") or "")
    return r


# FIX 2: call_llm - proper error logging + typed fallback
# ---------------------------------------------

async def call_llm(user_message: str, history: list = None) -> dict:
    """
    Call the LLM (Groq or Ollama) and return a parsed dict.
    """
    _FALLBACK = {
        "intent"    : "unknown",
        "sub_intent": "",
        "params"    : {},
        "reply"     : "An error occurred while processing your request. Please try again.",
    }

    ts = datetime.now().strftime("%H:%M:%S")

    # Use Groq if model looks like a Groq model and client is available
    if groq_client and ("/" in OLLAMA_MODEL or "llama" in OLLAMA_MODEL.lower() or "qwen" in OLLAMA_MODEL.lower() or "gemma" in OLLAMA_MODEL.lower() or "kimi" in OLLAMA_MODEL.lower()):
        try:
            print(f"[{ts}] LLM_REQ    model={OLLAMA_MODEL} (via Groq)", flush=True)
            
            # --- TOKEN OPTIMIZATION: Windowed History ---
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            
            # Add last 5 messages from history if available
            if history:
                # history is a list of {"role": "...", "text": "..."}
                for h in history[-5:]:
                    messages.append({"role": h["role"], "content": h["text"]})
            else:
                messages.append({"role": "user", "content": user_message})

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: groq_client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
            ))
            
            content = response.choices[0].message.content
            parsed = _parse_llm(content)
            print(
                f"[{ts}] LLM_OK     intent={parsed.get('intent')}  "
                f"sub={parsed.get('sub_intent')} (Groq)",
                flush=True,
            )
            return parsed

        except Exception as e:
            print(f"[{ts}] LLM_ERROR  Groq call failed: {e}", flush=True)
            return _FALLBACK

    # Fallback to Ollama
    url     = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model" : OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            print(f"[{ts}] LLM_REQ    model={OLLAMA_MODEL}  url={url}", flush=True)
            r = await client.post(url, json=payload)

            if r.status_code != 200:
                body_preview = r.text[:300].replace("\n", " ")
                print(
                    f"[{ts}] LLM_ERROR  HTTP {r.status_code}  url={url}\n"
                    f"           body preview: {body_preview}",
                    flush=True,
                )
                if r.status_code == 404:
                    print(
                        f"[{ts}] LLM_HINT   Model '{OLLAMA_MODEL}' not found in Ollama.\n"
                        f"           Run:  docker exec <ollama-container> ollama pull {OLLAMA_MODEL}",
                        flush=True,
                    )
                return _FALLBACK

            resp_json = r.json()
            content   = resp_json.get("message", {}).get("content", "")
            if not content:
                print(f"[{ts}] LLM_WARN   Empty 'content' in Ollama response: {str(resp_json)[:200]}", flush=True)
                return _FALLBACK

            parsed = _parse_llm(content)
            print(
                f"[{ts}] LLM_OK     intent={parsed.get('intent')}  "
                f"sub={parsed.get('sub_intent')} (Ollama)",
                flush=True,
            )
            return parsed

        except Exception as e:
            print(f"[{ts}] LLM_ERROR  Ollama call failed: {type(e).__name__}: {e}", flush=True)
            return _FALLBACK

# ---------------------------------------------
# PROCESS ONE MESSAGE  (session-aware)
# ---------------------------------------------

async def process(ctx: ConnCtx, message: str, session_id: str = ""):
    session = ctx.resolve_session(session_id)
    if session:
        session.add_history("user", message)

    await send_status(ctx, "Processing your request...", session_id=session_id)

    pre = _pre_classify(message)
    if pre == "identity":
        await send_intent(ctx, "identity", session_id=session_id)
        await send_reply(ctx, IDENTITY_REPLY, session_id=session_id)
        if session:
            session.add_history("bot", IDENTITY_REPLY)
        return
    if pre == "greeting":
        await send_intent(ctx, "greeting", session_id=session_id)
        await send_reply(ctx, GREETING_REPLY, session_id=session_id)
        if session:
            session.add_history("bot", GREETING_REPLY)
        return
    if pre == "capability":
        await send_intent(ctx, "capability", session_id=session_id)
        await send_reply(ctx, CAPABILITY_REPLY, session_id=session_id)
        if session:
            session.add_history("bot", CAPABILITY_REPLY)
        return

    history = session.history if session else []
    llm1       = await call_llm(message, history=history)
    intent     = llm1.get("intent",     "unknown").strip().lower()
    sub_intent = llm1.get("sub_intent", "").strip().lower()
    params     = {k: str(v) for k, v in llm1.get("params", {}).items()}
    llm_reply  = llm1.get("reply", "")

    if intent in ("identity", "greeting", "capability"):
        fixed = {"identity": IDENTITY_REPLY, "greeting": GREETING_REPLY,
                 "capability": CAPABILITY_REPLY}
        await send_intent(ctx, intent, session_id=session_id)
        await send_reply(ctx, fixed[intent], session_id=session_id)
        if session:
            session.add_history("bot", fixed[intent])
        return

    if intent == "unknown":
        intent = "faq" if _is_itsm_related(message) else "irrelevant"

    await send_intent(ctx, intent, sub_intent, session_id=session_id)

    if intent == "faq":
        reply = scrub_brands(llm_reply).strip()
        final = reply if reply and len(reply) >= 15 else FAQ_FALLBACK_REPLY
        await send_reply(ctx, final, session_id=session_id)
        if session:
            session.add_history("bot", final)
        return

    if intent == "irrelevant":
        await send_reply(ctx, IRRELEVANT_REPLY, session_id=session_id)
        if session:
            session.add_history("bot", IRRELEVANT_REPLY)
        return

    if intent not in INTENT_SCHEMAS:
        reply = scrub_brands(llm_reply).strip() or FAQ_FALLBACK_REPLY
        final = reply if _is_itsm_related(message) else IRRELEVANT_REPLY
        await send_reply(ctx, final, session_id=session_id)
        if session:
            session.add_history("bot", final)
        return

    if llm_reply and len(llm_reply) >= 10:
        await send_reply(ctx, llm_reply, session_id=session_id)

    try:
        status, data, params = await dispatch(ctx, intent, sub_intent, params,
                                               session_id=session_id)
    except asyncio.TimeoutError:
        return

    final_reply = _build_local_reply(intent, status, data, params)
    await send_reply(ctx, final_reply, session_id=session_id)
    if session:
        session.add_history("bot", final_reply)

# ---------------------------------------------
# SESSION COMMAND HANDLER
# ---------------------------------------------

async def handle_session_command(ctx: ConnCtx, msg: dict):
    mtype = msg.get("type","")

    if mtype == "session_create":
        name    = str(msg.get("name", "")).strip() or "New Session"
        session = ctx.create_session(name)
        ctx.set_active(session.session_id)
        await send_session_created(ctx, session, active=True)
        _log("SESSION", ctx.addr,
             session_id=session.session_id,
             session_name=session.name,
             extra="CREATED")
        return True

    if mtype == "session_switch":
        sid = str(msg.get("session_id","")).strip()
        session = ctx.set_active(sid)
        if not session:
            await send_session_error(ctx, f"Session '{sid}' not found.")
            _log("SESSION", ctx.addr, session_id=sid, extra="SWITCH FAILED (not found)")
        else:
            await send_session_switched(ctx, session)
            _log("SESSION", ctx.addr,
                 session_id=session.session_id,
                 session_name=session.name,
                 extra="SWITCHED")
        return True

    if mtype == "session_list":
        await send_session_list(ctx)
        return True

    if mtype == "session_delete":
        sid = str(msg.get("session_id","")).strip()
        if not sid:
            await send_session_error(ctx, "session_id is required for session_delete.")
            return True
        ok = ctx.delete_session(sid)
        if ok:
            await send_session_deleted(ctx, sid)
            _log("SESSION", ctx.addr, session_id=sid, extra="DELETED")
        else:
            await send_session_error(ctx, f"Session '{sid}' not found.")
            _log("SESSION", ctx.addr, session_id=sid, extra="DELETE FAILED (not found)")
        return True

    if mtype == "session_rename":
        sid  = str(msg.get("session_id","")).strip()
        name = str(msg.get("name","")).strip()
        if not sid:
            await send_session_error(ctx, "session_id is required for session_rename.")
            return True
        session = ctx.get_session(sid)
        if not session:
            await send_session_error(ctx, f"Session '{sid}' not found.")
            _log("SESSION", ctx.addr, session_id=sid, extra="RENAME FAILED (not found)")
        elif not name:
            await send_session_error(ctx, "name cannot be empty for session_rename.")
        else:
            old_name     = session.name
            session.name = name
            await send_session_renamed(ctx, session)
            _log("SESSION", ctx.addr,
                 session_id=session.session_id,
                 session_name=name,
                 extra=f"RENAMED (was '{old_name}')")
        return True

    return False

# ---------------------------------------------
# WEBSOCKET HANDLER
# ---------------------------------------------

async def handler(ws: WebSocketServerProtocol):
    addr = str(ws.remote_address)
    _log("CONNECT", addr)

    try:
        raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
        msg = json.loads(raw)
    except (asyncio.TimeoutError, json.JSONDecodeError, websockets.exceptions.ConnectionClosed):
        try:
            await ws.send(json.dumps({"type":"error",
                "text":'First message must be: {"type":"auth","token":"<JWT>"}'}))
        except Exception:
            pass
        return

    if msg.get("type") != "auth" or not msg.get("token","").strip():
        try:
            await ws.send(json.dumps({"type":"error",
                "text":'Expected {"type":"auth","token":"<JWT>"}'}))
        except Exception:
            pass
        return

    ctx = ConnCtx(ws, msg["token"].strip())
    await _send(ctx, {"type":"status","text":"Authenticated. Send your ITSM request."})
    _log("AUTH OK", addr)

    try:
        async for raw_msg in ws:
            try:
                msg = json.loads(raw_msg)
            except json.JSONDecodeError:
                await send_error(ctx, "Invalid JSON.")
                continue

            mtype = msg.get("type","")

            if mtype.startswith("session_"):
                await handle_session_command(ctx, msg)
                continue

            if mtype == "input":
                sid     = str(msg.get("session_id","")).strip()
                session = ctx.resolve_session(sid)

                if session is None:
                    await send_error(ctx, "No active session. Please create a session first.")
                    continue

                if session.is_busy():
                    await session.input_queue.put(msg)
                else:
                    await send_error(ctx,
                        "No active request waiting for input in this session. "
                        "Please send a new ITSM request first.",
                        session_id=session.session_id)

            elif mtype == "message":
                text = msg.get("text","").strip()
                if not text:
                    await send_error(ctx, "Empty message.")
                    continue
                if text.lower() in ("exit","quit","bye"):
                    await _send(ctx, {"type":"status","text":"Goodbye."})
                    break

                sid     = str(msg.get("session_id","")).strip()
                session = ctx.resolve_session(sid)

                if session is None:
                    session = ctx.create_session("Default")
                    ctx.set_active(session.session_id)
                    await send_session_created(ctx, session, active=True)
                    _log("SESSION", addr,
                         session_id=session.session_id,
                         session_name=session.name,
                         extra="AUTO-CREATED (Default)")

                if sid and ctx.active_session() and ctx.active_session().session_id != session.session_id:
                    ctx.set_active(session.session_id)

                if session.is_busy():
                    await session.input_queue.put({"type":"input","value":text})
                    continue

                sid_for_task = session.session_id
                _log("MESSAGE", addr,
                     session_id=sid_for_task,
                     session_name=session.name,
                     extra=f"intent dispatch -> '{text[:60]}{'...' if len(text) > 60 else ''}'")
                session.active_task = asyncio.create_task(
                    process(ctx, text, session_id=sid_for_task)
                )

            else:
                await send_error(ctx, f"Unexpected message type: {mtype}")

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        sessions      = ctx.list_sessions()
        session_ids   = ", ".join(s.session_id for s in sessions) if sessions else "none"
        session_names = ", ".join(f"'{s.name}'" for s in sessions) if sessions else "none"
        for s in sessions:
            s.cancel_task()
        _log("DISCONNECT", addr,
             extra=f"({len(sessions)} sessions: ids=[{session_ids}] names=[{session_names}])")

# ---------------------------------------------
# FIX 5: STARTUP REACHABILITY CHECK
# ---------------------------------------------

async def _check_ollama_reachable():
    """
    Pings Ollama at startup and logs a clear warning if unreachable.
    Also verifies the configured model exists in the local library.
    Does NOT abort startup - Ollama may become available shortly after.
    """
    ts = datetime.now().strftime("%H:%M:%S")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # /api/tags lists locally available models
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if r.status_code == 200:
                tags_data = r.json()
                models    = [m.get("name","") for m in tags_data.get("models", [])]
                if any(OLLAMA_MODEL in m for m in models):
                    print(
                        f"[{ts}] OLLAMA_OK  Reachable at {OLLAMA_BASE_URL}  "
                        f"model '{OLLAMA_MODEL}' found [OK]",
                        flush=True,
                    )
                else:
                    print(
                        f"[{ts}] OLLAMA_WARN  Reachable at {OLLAMA_BASE_URL} but "
                        f"model '{OLLAMA_MODEL}' NOT found in library.\n"
                        f"             Available: {models}\n"
                        f"             Run:  ollama pull {OLLAMA_MODEL}",
                        flush=True,
                    )
            else:
                print(
                    f"[{ts}] OLLAMA_WARN  Unexpected status {r.status_code} from {OLLAMA_BASE_URL}/api/tags",
                    flush=True,
                )
    except httpx.ConnectError:
        print(
            f"[{ts}] OLLAMA_WARN  Cannot reach Ollama at {OLLAMA_BASE_URL}\n"
            f"             -> In Docker-Compose ensure OLLAMA_BASE_URL=http://<ollama-service>:11434\n"
            f"             -> Server will start anyway; LLM calls will fail until Ollama is up.",
            flush=True,
        )
    except Exception as e:
        print(f"[{ts}] OLLAMA_WARN  Reachability check failed: {e}", flush=True)

# ---------------------------------------------
# ENTRY POINT
# ---------------------------------------------

async def main(host: str, port: int):
    print(f"\nITSM BOT WebSocket Server  v14.1  ->  ws://{host}:{port}")
    print(f"  Ollama : {OLLAMA_BASE_URL}  ({OLLAMA_MODEL})")
    print(f"  Backend: {BACKEND_BASE}")
    print(f"  Max retries: {MAX_RETRIES}")
    print()
    print("  +==============================================================================+")
    print("    v14.1 CHANGES - OLLAMA / DOCKER-COMPOSE FIXES                            ")
    print("  +==============================================================================+")
    print("    OLLAMA_BASE_URL read from env  (Docker service-name resolution)         ")
    print("    OLLAMA_MODEL    read from env  (override without rebuilding image)       ")
    print("    call_llm() logs every error type with actionable hints                  ")
    print("    Startup reachability check warns if Ollama/model not found              ")
    print("    <think>...</think> tags stripped from qwen3 reasoning output              ")
    print("    Model name whitespace-stripped to prevent silent 'not found' errors     ")
    print("    All v14.0 session-management features 100% preserved                   ")
    print("  +==============================================================================+")
    print()

    # Run reachability check before accepting connections
    if not (groq_client and ("/" in OLLAMA_MODEL or "llama" in OLLAMA_MODEL.lower() or "qwen" in OLLAMA_MODEL.lower() or "gemma" in OLLAMA_MODEL.lower() or "kimi" in OLLAMA_MODEL.lower())):
        await _check_ollama_reachable()
    else:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] GROQ_OK    Using Groq model '{OLLAMA_MODEL}'", flush=True)

    async with websockets.serve(handler, host, port):
        await asyncio.Future()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ITSM BOT WebSocket Server")
    ap.add_argument("--host",  default="0.0.0.0")
    ap.add_argument("--port",  type=int, default=8765)
    # FIX 4: Accept exact model tag used in your local Ollama library
    ap.add_argument("--model", default="qwen/qwen3-32b",
                    choices=["qwen/qwen3-32b", "gemma3:4b", "qwen3.5:cloud", "qwen3.5:397b-cloud", "kimi-k2.6:cloud"])
    args = ap.parse_args()
    # FIX 3: Strip whitespace from CLI-supplied model name too
    OLLAMA_MODEL = args.model.strip()
    asyncio.run(main(args.host, args.port))