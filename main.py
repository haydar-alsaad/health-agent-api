"""
Al-Noor Healthcare Agent API - v3.4 (multi-tenant, service-account auth)
Architecture: Supabase-backed via httpx REST, authenticated as a dedicated
service-account user (NOT the service role key).

CHANGES IN v3.4 — BILLING CORRECTNESS + BOOKING GUARD:
  - PARTIAL PAYMENTS NO LONGER LOSE MONEY. record_payment previously updated
    only `status` and left `patient_due_sar` at its original value. Combined
    with /patient filtering outstanding on status == "Outstanding" and paid on
    status == "Paid", a "Partially Paid" invoice matched NEITHER list and
    disappeared from the response: outstanding_total_sar fell to zero and the
    agent told the patient their account was clear while they still owed.
    Now the balance is decremented, "Partially Paid" counts as outstanding,
    and the response returns remaining_sar / fully_paid so the agent can quote
    the real remaining figure.
  - record_payment rejects amounts <= 0 and reports overpayment separately
    rather than silently clamping.
  - book_appointment now returns a structured 409 when the patient already has
    a live appointment with the same doctor on the same day. Every other write
    path had a server guard; this one relied solely on the agent checking
    upcoming_appointments first.

CHANGES IN v3.3.1:
  - The medication dose field is "Dose" in the seed JSONB, not "Dosage" —
    v3.3 returned null for every dose. Widened the field reader to accept any
    plausible spelling.
  - Also surfaced instructions_en/ar, frequency_ar, duration_days and
    refills_total, which were sitting unused in the medications array. The
    Arabic frequency in particular matters: the agent runs bilingual and
    previously had only the English string to work from.

CHANGES IN v3.3 — PRESCRIPTION REFILL DATA:
  refill_prescription needs prescription_id + medication_id + pharmacy_id.
  Only the first was reachable in a usable form:
    - pharmacy_id had NO source. `pharmacies` appeared in no response and
      there was no pharmacy tool, so the agent would gather the medication,
      ask "which pharmacy?", and then stall — it could not turn the answer
      into an ID. This is why refills froze mid-flow.
    - medication_id lived inside the prescriptions[].medications[] JSONB
      array, which the agent had to walk unaided. This is why refills that
      did complete often used the wrong medication.
  Fixes:
    - `pharmacies` added to the reference cache and returned on /patient —
      full catalog with per-pharmacy delivery availability and fee, so the
      agent quotes real numbers instead of memorised constants.
    - `refillable_medications` added to /patient — a FLAT list carrying
      prescription_id, medication_id, name, dosage, refills_remaining and a
      can_refill_now flag. Everything a refill call needs, no nesting.
    - /prescription/refill now resolves the pharmacy name from the cache
      rather than issuing a fresh query.
  No breaking changes: existing response fields are untouched, the two new
  keys are additive, and no endpoint signature changed.

CHANGES IN v3.2:
  - Added GET /whoami — a tenant-routing diagnostic. Resolves a caller_phone
    to its owner_id and explains whether it matched or fell back, bypassing
    (but reporting) the tenant cache. Added after a missing RLS policy on
    demo_users caused EVERY request to silently resolve to the fallback tenant
    for days: /health passed, isolation looked perfect in an audit, and the
    only symptom was that no registered tenant had any agent activity.
    Also reports how many tenants the API can see — 0 means the service
    account cannot read demo_users at all, which is a different bug from
    a phone number simply not being registered.

CHANGES FROM v3.0 — AUTHENTICATION:
  This project lives on Lovable Cloud, where the Supabase service role key is
  not available to us. Instead the API signs in as a dedicated auth user
  (railway-agent@nebelus.ai) that has additive `for all` RLS policies on all 11
  per-tenant tables, granting it cross-tenant read/write. Every PostgREST
  request carries that user's access token.

  - Sign-in on startup via the password grant; token cached in memory.
  - Proactive refresh 5 minutes before expiry using the refresh_token grant,
    falling back to a fresh password grant if the refresh token is rejected.
  - An asyncio.Lock serialises refreshes so N concurrent requests trigger one
    token fetch, not N.
  - Any PostgREST 401/403 forces a re-auth and retries the request ONCE. This
    covers token revocation, clock skew, and Supabase-side session eviction.
  - sb_get no longer swallows auth failures. A 401 after retry raises a 502
    instead of returning [] — an empty list masquerading as "no rows" is what
    made the original misconfiguration look like a 404 for hours.
  - /health reports token state (authenticated, seconds until expiry).

  SUPABASE_SERVICE_ROLE_KEY is no longer used. Leave it unset.

CARRIED FORWARD FROM v3.0:
  - caller_phone → owner_id tenant routing via the demo_users table, with
    DEFAULT_OWNER_ID as the fallback for unregistered callers.
  - sb_* helpers take an explicit `owner`; per-tenant tables without one raise
    rather than silently returning cross-tenant rows.
  - Phone normalization tolerating a missing "+".
  - HTTP/2, in-process reference cache for the shared doctors/clinics catalogs.

TENANCY MODEL:
  Per-tenant tables (scoped by owner_id):
    patients, appointments, prescriptions, lab_results, invoices,
    medical_history, refill_requests, preauth_requests, agent_actions,
    lab_documents, doctor_availability
  Shared tables (no owner_id, one copy for everyone):
    doctors, clinics, pharmacies, medications_catalog, insurance_providers

ENV VARS:
  SUPABASE_URL               (required)  e.g. https://xxxx.supabase.co
  SUPABASE_ANON_KEY          (required)  publishable/anon key — public, safe
  SUPABASE_SERVICE_EMAIL     (required)  railway-agent@nebelus.ai
  SUPABASE_SERVICE_PASSWORD  (required)  that account's password — SECRET
  DEFAULT_OWNER_ID           (required)  UUID of the fallback demo tenant
  SEED_ON_BOOT               (optional, default "false")
"""

import asyncio
import json
import os
import re
import sys
from datetime import date, datetime, timedelta
from time import monotonic as _monotonic
from typing import Optional, Any

import httpx
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# Config
# ============================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_EMAIL = os.environ.get("SUPABASE_SERVICE_EMAIL", "")
SUPABASE_SERVICE_PASSWORD = os.environ.get("SUPABASE_SERVICE_PASSWORD", "")
DEFAULT_OWNER_ID = os.environ.get("DEFAULT_OWNER_ID", "")
SEED_ON_BOOT = os.environ.get("SEED_ON_BOOT", "false").lower() == "true"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

_missing = [
    name for name, val in [
        ("SUPABASE_URL", SUPABASE_URL),
        ("SUPABASE_ANON_KEY", SUPABASE_ANON_KEY),
        ("SUPABASE_SERVICE_EMAIL", SUPABASE_SERVICE_EMAIL),
        ("SUPABASE_SERVICE_PASSWORD", SUPABASE_SERVICE_PASSWORD),
        ("DEFAULT_OWNER_ID", DEFAULT_OWNER_ID),
    ] if not val
]
if _missing:
    print(f"WARNING: missing required env vars: {', '.join(_missing)}. API will fail.")


# ============================================================
# Supabase auth — service-account token management
# ============================================================
# This project is on Lovable Cloud, so the service role key isn't available.
# Instead we sign in as a dedicated auth user that carries additive `for all`
# RLS policies on every per-tenant table, giving it cross-tenant access.
#
# Token lifecycle:
#   startup            -> password grant, cache access + refresh token
#   < 5 min to expiry  -> refresh_token grant (cheap)
#   refresh rejected   -> fall back to a fresh password grant
#   PostgREST 401/403  -> force re-auth, retry the request once
#
# A single asyncio.Lock serialises all of the above so N concurrent requests
# trigger one token fetch rather than N.

_TOKEN_REFRESH_MARGIN = 300.0  # refresh when < 5 min of life remains

_auth_state: dict = {
    "access_token": None,
    "refresh_token": None,
    "expires_at": 0.0,       # monotonic deadline
    "last_error": None,
    "signed_in_at": None,    # wall-clock ISO, for diagnostics
}
_auth_lock: Optional[asyncio.Lock] = None  # created in startup (needs a loop)


async def _auth_request(payload: dict, grant_type: str) -> dict:
    """POST to Supabase's token endpoint. Raises on failure."""
    url = f"{SUPABASE_URL}/auth/v1/token?grant_type={grant_type}"
    r = await http_client.post(
        url,
        headers={
            "apikey": SUPABASE_ANON_KEY,
            "Content-Type": "application/json",
        },
        json=payload,
    )
    r.raise_for_status()
    return r.json()


async def _sign_in_password() -> None:
    """Full sign-in with email + password. Replaces any cached token."""
    data = await _auth_request(
        {"email": SUPABASE_SERVICE_EMAIL, "password": SUPABASE_SERVICE_PASSWORD},
        "password",
    )
    _store_token(data)
    print(f"[auth] signed in as {SUPABASE_SERVICE_EMAIL}")


async def _sign_in_refresh() -> None:
    """Renew using the refresh token. Cheaper than a password grant."""
    rt = _auth_state.get("refresh_token")
    if not rt:
        raise RuntimeError("no refresh token cached")
    data = await _auth_request({"refresh_token": rt}, "refresh_token")
    _store_token(data)
    print("[auth] token refreshed")


def _store_token(data: dict) -> None:
    expires_in = float(data.get("expires_in") or 3600)
    _auth_state["access_token"] = data.get("access_token")
    _auth_state["refresh_token"] = data.get("refresh_token") or _auth_state.get("refresh_token")
    _auth_state["expires_at"] = _monotonic() + expires_in
    _auth_state["last_error"] = None
    _auth_state["signed_in_at"] = datetime.now().astimezone().isoformat()


async def ensure_token(force: bool = False) -> str:
    """Return a valid access token, refreshing or re-signing in as needed.

    `force=True` discards the cached token — used after a PostgREST 401.
    Serialised by _auth_lock so concurrent callers share one fetch.
    """
    global _auth_lock
    if _auth_lock is None:
        _auth_lock = asyncio.Lock()

    tok = _auth_state.get("access_token")
    fresh_enough = tok and (_auth_state["expires_at"] - _monotonic()) > _TOKEN_REFRESH_MARGIN
    if fresh_enough and not force:
        return tok

    async with _auth_lock:
        # Re-check inside the lock: another coroutine may have just refreshed.
        tok = _auth_state.get("access_token")
        fresh_enough = tok and (_auth_state["expires_at"] - _monotonic()) > _TOKEN_REFRESH_MARGIN
        if fresh_enough and not force:
            return tok

        try:
            if force:
                await _sign_in_password()
            else:
                try:
                    await _sign_in_refresh()
                except Exception:
                    await _sign_in_password()
        except Exception as e:
            _auth_state["last_error"] = str(e)[:300]
            print(f"[auth] sign-in FAILED: {e}")
            raise HTTPException(
                status_code=502,
                detail="Database authentication failed — check service account credentials",
            )

        return _auth_state["access_token"]


async def sb_headers(extra: Optional[dict] = None) -> dict:
    """Build request headers with a currently-valid access token."""
    token = await ensure_token()
    h = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    if extra:
        h.update(extra)
    return h


def _auth_stats() -> dict:
    """Diagnostic: token state, exposed via /health. Never leaks the token."""
    exp = _auth_state.get("expires_at") or 0
    return {
        "authenticated": bool(_auth_state.get("access_token")),
        "expires_in_seconds": round(exp - _monotonic(), 1) if exp else None,
        "signed_in_at": _auth_state.get("signed_in_at"),
        "last_error": _auth_state.get("last_error"),
    }


# ============================================================
# Tenancy: which tables carry owner_id
# ============================================================
# Per-tenant tables get `owner_id=eq.<uuid>` injected into every query and
# `owner_id` injected into every inserted row.
#
# Shared tables (doctors, clinics, pharmacies, medications_catalog,
# insurance_providers) have no owner_id — one copy serves all tenants.

TENANT_TABLES = {
    "patients",
    "appointments",
    "prescriptions",
    "lab_results",
    "lab_documents",
    "invoices",
    "medical_history",
    "refill_requests",
    "preauth_requests",
    "agent_actions",
    "doctor_availability",
}


# ============================================================
# App
# ============================================================

app = FastAPI(title="Al-Noor Health Agent API", version="3.4")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared httpx client for connection pooling
http_client: Optional[httpx.AsyncClient] = None


@app.on_event("startup")
async def startup():
    global http_client
    # HTTP/2 lets us multiplex unlimited parallel requests over ONE TCP connection
    # to Supabase. Without it, httpx defaults to HTTP/1.1 which only allows one
    # request per connection at a time — so 9 parallel queries hit the
    # ~6-concurrent-connection limit and 3 queue up for ~300 ms. Requires h2 package
    # (added via httpx[http2] in requirements.txt).
    http_client = httpx.AsyncClient(
        http2=True,
        timeout=30.0,
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
    )

    # Sign in as the service account up front so the first real request doesn't
    # pay for it. Don't crash the process on failure — /health will report the
    # problem and every request retries, so a transient Supabase blip at boot
    # doesn't take the service down permanently.
    try:
        await ensure_token(force=True)
    except Exception as e:
        print(f"[auth] startup sign-in failed (will retry on first request): {e}")

    if SEED_ON_BOOT:
        await seed_if_empty()


@app.on_event("shutdown")
async def shutdown():
    global http_client
    if http_client:
        await http_client.aclose()


# ============================================================
# Phone normalization + tenant resolution
# ============================================================

def normalize_phone(raw: Optional[str]) -> Optional[str]:
    """Canonicalize a phone number to E.164 with a leading '+'.

    Tolerates: missing '+', spaces, dashes, parentheses, leading '00'.
    Returns None for empty/unusable input.

    This exists because URL query strings sometimes drop or mangle the '+'
    (it URL-encodes to a space), and agents occasionally strip it. Normalizing
    on both write and read sides means the lookup matches regardless.
    """
    if not raw:
        return None
    s = re.sub(r"[\s\-()]", "", str(raw).strip())
    if not s:
        return None
    if s.startswith("00"):
        s = "+" + s[2:]
    elif not s.startswith("+"):
        s = "+" + s
    # Must be + followed by digits only
    if not re.fullmatch(r"\+\d{6,20}", s):
        return None
    return s


# Tenant resolution cache. Bindings change rarely (a sales person sets their
# demo number once), so a longer TTL than the reference cache is fine.
_TENANT_CACHE_TTL = 300.0  # 5 minutes
_tenant_cache: dict = {}  # normalized_phone -> {"owner_id": str, "ts": float}


async def resolve_owner(caller_phone: Optional[str]) -> str:
    """Resolve a WhatsApp phone number to the owning demo tenant's owner_id.

    Falls back to DEFAULT_OWNER_ID when:
      - caller_phone is missing (agent not yet updated with the parameter)
      - caller_phone doesn't match any demo_users row (sales person hasn't
        registered their demo number yet)

    The fallback is deliberate: a demo that silently lands in the shared
    default tenant is recoverable; a hard 400 mid-demo in front of a prospect
    is not.
    """
    normalized = normalize_phone(caller_phone)
    if not normalized:
        return DEFAULT_OWNER_ID

    cached = _tenant_cache.get(normalized)
    if cached and (_monotonic() - cached["ts"]) <= _TENANT_CACHE_TTL:
        return cached["owner_id"]

    # demo_users is NOT a per-tenant table — it's the tenant registry itself.
    rows = await _sb_raw_get("demo_users", {
        "whatsapp_number": f"eq.{normalized}",
        "select": "owner_id",
        "limit": "1",
    })
    owner = rows[0]["owner_id"] if rows else DEFAULT_OWNER_ID

    _tenant_cache[normalized] = {"owner_id": owner, "ts": _monotonic()}
    return owner


def _tenant_cache_stats() -> dict:
    """Diagnostic: how many phone→tenant bindings are currently cached."""
    now = _monotonic()
    return {
        "entries": len(_tenant_cache),
        "oldest_age_seconds": (
            round(now - min(v["ts"] for v in _tenant_cache.values()), 1)
            if _tenant_cache else None
        ),
    }


# ============================================================
# Supabase REST helpers (tenant-aware)
# ============================================================

AUTH_FAIL_CODES = (401, 403)


async def _sb_request(method: str, table: str, *, params=None, json_body=None, extra_headers=None):
    """Execute one PostgREST call with a valid token, retrying once on 401/403.

    A 401 here means the token expired, was revoked, or the session was evicted
    server-side. Re-authenticating and retrying is the correct response — the
    request itself is fine. If it fails a second time, the credentials or the
    RLS policies are genuinely wrong and we surface that rather than hiding it.
    """
    url = f"{SUPABASE_URL}/rest/v1/{table}"

    for attempt in (1, 2):
        headers = await sb_headers(extra_headers)
        r = await http_client.request(
            method, url, headers=headers, params=params or {}, json=json_body
        )
        if r.status_code in AUTH_FAIL_CODES and attempt == 1:
            print(f"[auth] {r.status_code} on {method} {table} — re-authenticating and retrying")
            await ensure_token(force=True)
            continue
        r.raise_for_status()
        return r.json()


async def _sb_raw_get(table: str, params: Optional[dict] = None) -> list:
    """Unscoped GET. ONLY for non-tenant tables like demo_users.
    Do not use for anything in TENANT_TABLES.

    Auth failures are NOT swallowed. Returning [] on a 401 is what made a
    misconfigured key look like "patient not found" instead of a broken
    deployment — an empty list is a legitimate answer, a 401 never is.
    """
    try:
        return await _sb_request("GET", table, params=params or {})
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        print(f"sb_get error {table}: {code} {e.response.text[:200]}")
        if code in AUTH_FAIL_CODES:
            raise HTTPException(
                status_code=502,
                detail="Database authentication failed — service account cannot read this table",
            )
        return []
    except HTTPException:
        raise
    except Exception as e:
        print(f"sb_get exception {table}: {e}")
        return []


def _scope_params(table: str, params: Optional[dict], owner: Optional[str]) -> dict:
    """Inject owner_id filter for per-tenant tables.

    Raises loudly if a per-tenant table is queried without an owner. Silent
    cross-tenant reads are the worst possible failure mode here — better to
    500 and see it in the logs than to serve another sales person's demo data.
    """
    p = dict(params or {})
    if table in TENANT_TABLES:
        if not owner:
            raise HTTPException(
                status_code=500,
                detail=f"Internal error: query on per-tenant table '{table}' missing owner scope",
            )
        p["owner_id"] = f"eq.{owner}"
    return p


async def sb_get(table: str, params: Optional[dict] = None, owner: Optional[str] = None) -> list:
    """GET from Supabase REST, scoped to the tenant for per-tenant tables."""
    return await _sb_raw_get(table, _scope_params(table, params, owner))


async def sb_get_one(table: str, params: Optional[dict] = None, owner: Optional[str] = None) -> Optional[dict]:
    """GET single row from Supabase. Returns dict or None."""
    rows = await sb_get(table, params, owner=owner)
    return rows[0] if rows else None


async def sb_insert(table: str, payload, owner: Optional[str] = None) -> Any:
    """INSERT into Supabase, injecting owner_id for per-tenant tables.
    Accepts a single dict or a list of dicts."""
    if table in TENANT_TABLES:
        if not owner:
            raise HTTPException(
                status_code=500,
                detail=f"Internal error: insert into per-tenant table '{table}' missing owner scope",
            )
        if isinstance(payload, list):
            payload = [{**row, "owner_id": owner} for row in payload]
        else:
            payload = {**payload, "owner_id": owner}

    try:
        return await _sb_request("POST", table, json_body=payload)
    except httpx.HTTPStatusError as e:
        print(f"sb_insert error {table}: {e.response.status_code} {e.response.text[:300]}")
        raise HTTPException(status_code=500, detail=f"Insert to {table} failed: {e.response.text[:200]}")


async def sb_update(table: str, params: dict, payload: dict, owner: Optional[str] = None) -> Any:
    """UPDATE rows matching params, scoped to the tenant for per-tenant tables."""
    scoped = _scope_params(table, params, owner)
    try:
        return await _sb_request("PATCH", table, params=scoped, json_body=payload)
    except httpx.HTTPStatusError as e:
        print(f"sb_update error {table}: {e.response.status_code} {e.response.text[:300]}")
        raise HTTPException(status_code=500, detail=f"Update {table} failed: {e.response.text[:200]}")


async def log_agent_action(
    patient_id: Optional[str],
    action_type: str,
    description: str,
    metadata: Optional[dict] = None,
    status: str = "Success",
    owner: Optional[str] = None,
):
    """Insert into agent_actions for the Live Activity Drawer.
    Scoped to the tenant so each sales person sees only their own activity feed."""
    try:
        await sb_insert("agent_actions", {
            "patient_id": patient_id,
            "action_type": action_type,
            "description": description,
            "metadata": metadata or {},
            "status": status,
        }, owner=owner)
    except Exception as e:
        # Audit log failures should not break the parent operation
        print(f"agent_actions log failed: {e}")


# ============================================================
# In-process cache for SHARED reference tables
# ============================================================
# doctors (21 rows) and clinics (3 rows) are SHARED across all tenants — one
# copy of the catalog, no owner_id. So this cache stays global; it is NOT
# per-tenant and does not need invalidating when a tenant resets.
#
# NOTE: doctor_availability is NOT cached here. It moved to per-tenant in the
# multi-tenancy migration (so two sales people can't collide on the same slot),
# and it changes with every booking. It's queried live, scoped by owner.

_REF_CACHE_TTL = 60.0  # seconds
_reference_cache: dict = {
    "doctors": {"data": None, "ts": 0.0},
    "clinics": {"data": None, "ts": 0.0},
    "pharmacies": {"data": None, "ts": 0.0},
}


async def get_doctors_cached() -> list:
    """Return the shared doctors catalog from cache or fetch+cache it."""
    c = _reference_cache["doctors"]
    if c["data"] is None or (_monotonic() - c["ts"]) > _REF_CACHE_TTL:
        c["data"] = await sb_get("doctors", {
            "select": "doctor_id,full_name_en,full_name_ar,specialty_en,specialty_ar,sub_specialty_en,sub_specialty_ar,title_en,title_ar,primary_clinic_id,languages,years_of_experience,qualifications,consultation_fee_sar,followup_fee_sar,bio_en,bio_ar,status"
        })
        c["ts"] = _monotonic()
    return c["data"]


async def get_clinics_cached() -> list:
    """Return the shared clinics catalog from cache or fetch+cache it."""
    c = _reference_cache["clinics"]
    if c["data"] is None or (_monotonic() - c["ts"]) > _REF_CACHE_TTL:
        c["data"] = await sb_get("clinics", {"select": "*"})
        c["ts"] = _monotonic()
    return c["data"]


async def get_pharmacies_cached() -> list:
    """Return the shared pharmacies catalog from cache or fetch+cache it.

    Exposed on /patient because the agent cannot complete a refill without a
    pharmacy_id, and before v3.3 there was no way for it to obtain one — no
    pharmacy tool existed and pharmacies were absent from every response. The
    agent would collect the medication, then stall at "which pharmacy?" with
    nowhere to resolve the answer to an ID.
    """
    c = _reference_cache["pharmacies"]
    if c["data"] is None or (_monotonic() - c["ts"]) > _REF_CACHE_TTL:
        c["data"] = await sb_get("pharmacies", {"select": "*"})
        c["ts"] = _monotonic()
    return c["data"]


def _cache_stats() -> dict:
    """Diagnostic: current cache warmth per shared reference table."""
    now = _monotonic()
    return {
        table: {
            "warm": entry["data"] is not None,
            "row_count": len(entry["data"]) if entry["data"] is not None else 0,
            "age_seconds": round(now - entry["ts"], 1) if entry["ts"] else None,
        }
        for table, entry in _reference_cache.items()
    }


# ============================================================
# Data seeding (bootstrap only — OFF by default)
# ============================================================
# In the multi-tenant world, baseline data lives in Supabase's *_backup tables
# and is cloned per-user by clone_baseline_for_user() on signup. This JSON seed
# path is kept only for bootstrapping a brand-new empty environment; it writes
# everything under DEFAULT_OWNER_ID. Enable with SEED_ON_BOOT=true.

SEED_FILES = [
    # Order matters for foreign keys: independent tables first
    ("insurance_providers", "insurance_providers.json"),
    ("clinics", "clinics.json"),
    ("pharmacies", "pharmacies.json"),
    ("doctors", "doctors.json"),
    ("patients", "patients.json"),
    ("medications_catalog", "medications_catalog.json"),
    ("appointments", "appointments.json"),
    ("lab_results", "lab_results.json"),
    ("prescriptions", "prescriptions.json"),
    ("invoices", "invoices.json"),
    ("medical_history", "medical_history.json"),
    ("doctor_availability", "doctor_availability.json"),
    ("preauth_requests", "preauth_requests.json"),
    ("refill_requests", "refill_requests.json"),
]


def json_to_db_row(table: str, raw: dict) -> dict:
    """Map raw JSON keys (e.g. 'Patient ID') to DB column names ('patient_id')."""
    mappings = {
        "insurance_providers": {
            "Provider ID": "provider_id",
            "Provider Name (EN)": "provider_name_en",
            "Provider Name (AR)": "provider_name_ar",
            "Plan Tier (EN)": "plan_tier_en",
            "Plan Tier (AR)": "plan_tier_ar",
            "Annual Premium Range SAR": "annual_premium_range_sar",
            "Annual Limit SAR": "annual_limit_sar",
            "GP Consultation Coverage": "gp_consultation_coverage",
            "Specialist Consultation Coverage": "specialist_consultation_coverage",
            "Lab Coverage": "lab_coverage",
            "Imaging Coverage": "imaging_coverage",
            "Medication Coverage": "medication_coverage",
            "Pre-Authorization Required For": "pre_authorization_required_for",
            "Network Hospitals": "network_hospitals",
            "Co-pay Notes": "co_pay_notes",
            "ER Coverage": "er_coverage",
        },
        "clinics": {
            "Clinic ID": "clinic_id",
            "Clinic Name (EN)": "clinic_name_en",
            "Clinic Name (AR)": "clinic_name_ar",
            "Type": "type",
            "Address (EN)": "address_en",
            "Address (AR)": "address_ar",
            "Phone": "phone",
            "City (EN)": "city_en",
            "City (AR)": "city_ar",
            "Operating Hours": "operating_hours",
            "Specialties Available": "specialties_available",
            "Pharmacy On Site": "pharmacy_on_site",
            "Pharmacy ID": "pharmacy_id",
            "Lab On Site": "lab_on_site",
            "Imaging On Site": "imaging_on_site",
            "Emergency Department": "emergency_department",
            "Parking": "parking",
            "Bed Capacity": "bed_capacity",
        },
        "pharmacies": {
            "Pharmacy ID": "pharmacy_id",
            "Pharmacy Name (EN)": "pharmacy_name_en",
            "Pharmacy Name (AR)": "pharmacy_name_ar",
            "Type": "type",
            "Linked Clinic ID": "linked_clinic_id",
            "Address (EN)": "address_en",
            "Address (AR)": "address_ar",
            "City": "city",
            "Phone": "phone",
            "Operating Hours": "operating_hours",
            "Home Delivery Available": "home_delivery_available",
            "Home Delivery Fee SAR": "home_delivery_fee_sar",
            "Home Delivery Cities": "home_delivery_cities",
            "Home Delivery Window": "home_delivery_window",
        },
        "doctors": {
            "Doctor ID": "doctor_id",
            "Full Name (EN)": "full_name_en",
            "Full Name (AR)": "full_name_ar",
            "Specialty (EN)": "specialty_en",
            "Specialty (AR)": "specialty_ar",
            "Sub-specialty (EN)": "sub_specialty_en",
            "Sub-specialty (AR)": "sub_specialty_ar",
            "Title (EN)": "title_en",
            "Title (AR)": "title_ar",
            "Languages": "languages",
            "Years of Experience": "years_of_experience",
            "Qualifications": "qualifications",
            "Primary Clinic ID": "primary_clinic_id",
            "Visiting Clinic IDs": "visiting_clinic_ids",
            "Consultation Fee SAR": "consultation_fee_sar",
            "Follow-up Fee SAR": "followup_fee_sar",
            "Bio (EN)": "bio_en",
            "Bio (AR)": "bio_ar",
            "Status": "status",
        },
        "patients": {
            "Patient ID": "patient_id",
            "Full Name (EN)": "full_name_en",
            "Full Name (AR)": "full_name_ar",
            "Date of Birth": "date_of_birth",
            "Age": "age",
            "Gender": "gender",
            "Phone": "phone",
            "Email": "email",
            "Preferred Language": "preferred_language",
            "City (EN)": "city_en",
            "City (AR)": "city_ar",
            "Address (EN)": "address_en",
            "Address (AR)": "address_ar",
            "Insurance Provider ID": "insurance_provider_id",
            "Insurance Policy Number": "insurance_policy_number",
            "Primary Care Doctor ID": "primary_care_doctor_id",
            "Allergies": "allergies",
            "Active Conditions (EN)": "active_conditions_en",
            "Active Conditions (AR)": "active_conditions_ar",
            "Emergency Contact Name": "emergency_contact_name",
            "Emergency Contact Phone": "emergency_contact_phone",
            "Parent/Guardian": "parent_guardian",
            "Patient Status": "patient_status",
            "Registered Since": "registered_since",
            "Demo Notes": "demo_notes",
        },
        "medications_catalog": {
            "Medication ID": "medication_id",
            "Name (EN)": "name_en",
            "Name (AR)": "name_ar",
            "Drug Class (EN)": "drug_class_en",
            "Drug Class (AR)": "drug_class_ar",
            "Indication (EN)": "indication_en",
            "Indication (AR)": "indication_ar",
            "Common Dosages": "common_dosages",
            "Side Effects (EN)": "side_effects_en",
            "Side Effects (AR)": "side_effects_ar",
            "Interactions": "interactions",
            "Requires Prescription": "requires_prescription",
            "Controlled Substance": "controlled_substance",
            "Coverage Tier": "coverage_tier",
        },
        "appointments": {
            "Appointment ID": "appointment_id",
            "Patient ID": "patient_id",
            "Doctor ID": "doctor_id",
            "Clinic ID": "clinic_id",
            "Date": "date",
            "Start Time": "start_time",
            "Duration Minutes": "duration_minutes",
            "Type": "type",
            "Reason for Visit": "reason_for_visit",
            "Status": "status",
            "Notes": "notes",
            "Follow-up Required": "followup_required",
            "Created Date": "created_date",
        },
        "lab_results": {
            "Lab Result ID": "lab_result_id",
            "Patient ID": "patient_id",
            "Ordering Doctor ID": "ordering_doctor_id",
            "Clinic ID": "clinic_id",
            "Test Type": "test_type",
            "Test Name (EN)": "test_name_en",
            "Test Name (AR)": "test_name_ar",
            "Test Code": "test_code",
            "Order Date": "order_date",
            "Result Date": "result_date",
            "Status": "status",
            "Estimated Available": "estimated_available",
            "Linked Appointment ID": "linked_appointment_id",
            "Results": "results",
            "Imaging Findings (EN)": "imaging_findings_en",
            "Imaging Findings (AR)": "imaging_findings_ar",
            "Radiologist": "radiologist",
            "Lab Tech": "lab_tech",
            "Notes (EN)": "notes_en",
            "Notes (AR)": "notes_ar",
        },
        "prescriptions": {
            "Prescription ID": "prescription_id",
            "Patient ID": "patient_id",
            "Prescribing Doctor ID": "prescribing_doctor_id",
            "Clinic ID": "clinic_id",
            "Issued Date": "issued_date",
            "Expiration Date": "expiration_date",
            "Status": "status",
            "Medications": "medications",
            "Last Filled Date": "last_filled_date",
            "Last Filled Pharmacy ID": "last_filled_pharmacy_id",
            "Linked Appointment ID": "linked_appointment_id",
            "Linked Diagnosis (EN)": "linked_diagnosis_en",
            "Linked Diagnosis (AR)": "linked_diagnosis_ar",
        },
        "invoices": {
            "Invoice ID": "invoice_id",
            "Patient ID": "patient_id",
            "Linked Appointment ID": "linked_appointment_id",
            "Linked Lab Result ID": "linked_lab_result_id",
            "Issue Date": "issue_date",
            "Due Date": "due_date",
            "Items": "items",
            "Subtotal SAR": "subtotal_sar",
            "Insurance Provider (EN)": "insurance_provider_en",
            "Insurance Provider (AR)": "insurance_provider_ar",
            "Insurance Covered SAR": "insurance_covered_sar",
            "Patient Due SAR": "patient_due_sar",
            "Status": "status",
            "Payment Method": "payment_method",
            "Payment Date": "payment_date",
            "Notes (EN)": "notes_en",
            "Notes (AR)": "notes_ar",
        },
        "medical_history": {
            "History ID": "history_id",
            "Patient ID": "patient_id",
            "Event Date": "event_date",
            "Event Type": "event_type",
            "Description (EN)": "description_en",
            "Description (AR)": "description_ar",
            "Doctor ID": "doctor_id",
            "Linked Appointment ID": "linked_appointment_id",
        },
        "doctor_availability": {
            "Slot ID": "slot_id",
            "Doctor ID": "doctor_id",
            "Clinic ID": "clinic_id",
            "Date": "date",
            "Day of Week": "day_of_week",
            "Start Time": "start_time",
            "End Time": "end_time",
            "Slot Capacity": "slot_capacity",
            "Booked Count": "booked_count",
            "Status": "status",
        },
        "preauth_requests": {
            "Preauth ID": "preauth_id",
            "Patient ID": "patient_id",
            "Doctor ID": "doctor_id",
            "Procedure Name": "procedure_name",
            "Insurance Provider ID": "insurance_provider_id",
            "Status": "status",
            "Requested At": "requested_at",
            "Requested By": "requested_by",
            "Reviewed At": "reviewed_at",
            "Reviewer Notes": "reviewer_notes",
        },
        "refill_requests": {
            "Prescription ID": "prescription_id",
            "Patient ID": "patient_id",
            "Medication ID": "medication_id",
            "Medication Name (EN)": "medication_name_en",
            "Pharmacy ID": "pharmacy_id",
            "Delivery Method": "delivery_method",
            "Status": "status",
            "Requested At": "requested_at",
            "Requested By": "requested_by",
            "Processed At": "processed_at",
        },
    }
    m = mappings.get(table, {})
    return {m[k]: v for k, v in raw.items() if k in m}


async def seed_if_empty():
    """Bootstrap an empty environment from JSON, all under DEFAULT_OWNER_ID.
    Normally OFF — baseline cloning is Supabase's job in the multi-tenant setup."""
    if not DEFAULT_OWNER_ID:
        print("[seed] DEFAULT_OWNER_ID not set — refusing to seed.")
        return
    try:
        existing = await sb_get(
            "patients", {"select": "patient_id", "limit": "1"}, owner=DEFAULT_OWNER_ID
        )
        if existing:
            print(f"[seed] Default tenant already has data. Skipping seed.")
            return
        print("[seed] Empty default tenant detected. Seeding from JSON files...")
        for table, filename in SEED_FILES:
            path = os.path.join(DATA_DIR, filename)
            if not os.path.exists(path):
                print(f"[seed] {filename} not found, skipping")
                continue
            with open(path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            if not rows:
                continue
            mapped = [json_to_db_row(table, r) for r in rows]
            for i in range(0, len(mapped), 500):
                chunk = mapped[i:i+500]
                try:
                    await sb_insert(table, chunk, owner=DEFAULT_OWNER_ID)
                except Exception as e:
                    print(f"[seed] {table} batch {i} failed: {e}")
            print(f"[seed] {table}: {len(mapped)} rows")
        print("[seed] Done.")
    except Exception as e:
        print(f"[seed] failed: {e}")


# ============================================================
# Helper: enrich rows with related data
# ============================================================

def enrich_appointment(apt: dict, doctors_by_id: dict, clinics_by_id: dict) -> dict:
    """Add doctor + clinic names to an appointment row (pure function, no I/O)."""
    d = doctors_by_id.get(apt.get("doctor_id"), {})
    c = clinics_by_id.get(apt.get("clinic_id"), {})
    return {
        **apt,
        "doctor_name_en": d.get("full_name_en"),
        "doctor_name_ar": d.get("full_name_ar"),
        "doctor_specialty_en": d.get("specialty_en"),
        "doctor_specialty_ar": d.get("specialty_ar"),
        "clinic_name_en": c.get("clinic_name_en"),
        "clinic_name_ar": c.get("clinic_name_ar"),
        "clinic_address_en": c.get("address_en"),
        "clinic_address_ar": c.get("address_ar"),
    }


# ============================================================
# Health check
# ============================================================

@app.get("/")
async def root():
    return {
        "service": "Al-Noor Health Agent API",
        "version": "3.4",
        "multi_tenant": True,
        "auth_mode": "service_account",
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_ANON_KEY),
        "service_account_configured": bool(SUPABASE_SERVICE_EMAIL and SUPABASE_SERVICE_PASSWORD),
        "default_owner_configured": bool(DEFAULT_OWNER_ID),
    }


@app.get("/health")
async def health():
    """Quick health check (used by cron-job.org for warming).
    Exposes reference-cache and tenant-cache warmth for production diagnostics."""
    if _missing:
        return {
            "status": "degraded",
            "reason": f"missing env vars: {', '.join(_missing)}",
            "auth": _auth_stats(),
        }

    checks = {"api": "ok"}
    try:
        rows = await sb_get(
            "patients", {"select": "patient_id", "limit": "1"}, owner=DEFAULT_OWNER_ID
        )
        checks["supabase"] = "ok" if rows else "no_data"
    except Exception as e:
        checks["supabase"] = f"error: {str(e)[:200]}"

    auth = _auth_stats()
    healthy = checks["supabase"] in ("ok", "no_data") and auth["authenticated"]

    return {
        "status": "ok" if healthy else "degraded",
        "version": "3.4",
        "multi_tenant": True,
        "auth_mode": "service_account",
        "default_owner_configured": bool(DEFAULT_OWNER_ID),
        "checks": checks,
        "auth": auth,
        "reference_cache": _cache_stats(),
        "tenant_cache": _tenant_cache_stats(),
    }


# ============================================================
# GET /whoami — tenant routing diagnostic
# ============================================================
# Answers "which demo tenant does this phone number resolve to, and why?"
# without needing a Supabase audit.
#
# This exists because a broken phone->tenant lookup is INVISIBLE from outside:
# every request silently falls back to DEFAULT_OWNER_ID, /health still passes
# (it queries with DEFAULT_OWNER_ID directly), and the data still looks
# correctly isolated because nothing leaks — everything just lands in one
# tenant. That failure cost a full multi-tenant data audit to find once.
#
# Deliberately bypasses _tenant_cache while REPORTING its state, so a stale
# cache entry can be told apart from a genuinely failing lookup.

@app.get("/whoami")
async def whoami(
    caller_phone: Optional[str] = Query(
        None, description="Phone number to resolve, with or without the leading '+'"
    ),
):
    """Resolve a caller_phone to its demo tenant and explain the result."""
    normalized = normalize_phone(caller_phone)

    result: dict = {
        "caller_phone_received": caller_phone,
        "normalized_phone": normalized,
        "default_owner_id": DEFAULT_OWNER_ID or None,
    }

    # Report the cached value WITHOUT using it, so a stale cache is visible.
    cached = _tenant_cache.get(normalized) if normalized else None
    result["cache"] = {
        "present": bool(cached),
        "owner_id": cached["owner_id"] if cached else None,
        "age_seconds": round(_monotonic() - cached["ts"], 1) if cached else None,
    }

    if not normalized:
        result.update({
            "matched": False,
            "fell_back": True,
            "resolved_owner_id": DEFAULT_OWNER_ID or None,
            "reason": "caller_phone missing or not a valid phone number",
        })
        return result

    rows = await _sb_raw_get("demo_users", {
        "whatsapp_number": f"eq.{normalized}",
        "select": "owner_id,email",
        "limit": "1",
    })

    if rows:
        result.update({
            "matched": True,
            "fell_back": False,
            "resolved_owner_id": rows[0]["owner_id"],
            "tenant_email": rows[0].get("email"),
        })
    else:
        # How many tenants CAN we see? If this is 0 while the portal shows
        # registered users, the service account can't read demo_users at all
        # (missing RLS policy) rather than the number simply not being registered.
        all_rows = await _sb_raw_get("demo_users", {"select": "owner_id"})
        result.update({
            "matched": False,
            "fell_back": True,
            "resolved_owner_id": DEFAULT_OWNER_ID or None,
            "visible_tenant_count": len(all_rows),
            "reason": (
                f"no demo_users row with whatsapp_number = '{normalized}' "
                f"({len(all_rows)} tenants visible to the API) — "
                "writes for this caller will land in the fallback tenant"
            ),
        })

    return result


# ============================================================
# READ: /patient (the workhorse — parallelized)
# ============================================================

@app.get("/patient")
async def get_patient(
    patient_id: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
    phone: Optional[str] = Query(None),
    caller_phone: Optional[str] = Query(None, description="Demo tenant routing — WhatsApp sender number"),
):
    """Returns full patient package: profile, insurance, primary doctor,
    appointments, prescriptions, lab results, invoices, medical history.
    All sub-fetches run in parallel via asyncio.gather, scoped to the tenant."""
    owner = await resolve_owner(caller_phone)

    # Step 1: find the patient (within this tenant)
    if patient_id:
        patient = await sb_get_one("patients", {"patient_id": f"eq.{patient_id}"}, owner=owner)
    elif name:
        patient = await sb_get_one("patients", {
            "or": f"(full_name_en.ilike.*{name}*,full_name_ar.ilike.*{name}*)"
        }, owner=owner)
    elif phone:
        # Normalize so a missing '+' still matches. Try canonical form first,
        # then the raw value as a fallback for legacy rows.
        normalized = normalize_phone(phone)
        patient = None
        if normalized:
            patient = await sb_get_one("patients", {"phone": f"eq.{normalized}"}, owner=owner)
        if not patient:
            patient = await sb_get_one("patients", {"phone": f"eq.{phone}"}, owner=owner)
    else:
        raise HTTPException(status_code=400, detail="Provide patient_id, name, or phone")

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    pid = patient["patient_id"]

    # Step 2: fetch ALL related data in parallel, scoped to this tenant.
    # Doctors and clinics are SHARED catalogs served from the global cache.
    today = date.today().isoformat()
    (
        appointments,
        prescriptions,
        lab_results,
        invoices,
        history,
        all_doctors,
        all_clinics,
        all_pharmacies,
        insurance,
        refill_reqs,
        preauth_reqs,
    ) = await asyncio.gather(
        sb_get("appointments", {
            "patient_id": f"eq.{pid}",
            "order": "date.asc,start_time.asc",
        }, owner=owner),
        sb_get("prescriptions", {
            "patient_id": f"eq.{pid}",
            "order": "issued_date.desc",
        }, owner=owner),
        sb_get("lab_results", {
            "patient_id": f"eq.{pid}",
            "order": "order_date.desc",
        }, owner=owner),
        sb_get("invoices", {
            "patient_id": f"eq.{pid}",
            "order": "issue_date.desc",
        }, owner=owner),
        sb_get("medical_history", {
            "patient_id": f"eq.{pid}",
            "order": "event_date.desc",
        }, owner=owner),
        get_doctors_cached(),
        get_clinics_cached(),
        get_pharmacies_cached(),
        sb_get_one("insurance_providers", {
            "provider_id": f"eq.{patient.get('insurance_provider_id', '')}"
        }) if patient.get("insurance_provider_id") else asyncio.sleep(0, result=None),
        sb_get("refill_requests", {
            "patient_id": f"eq.{pid}",
            "order": "requested_at.desc",
        }, owner=owner),
        sb_get("preauth_requests", {
            "patient_id": f"eq.{pid}",
            "order": "requested_at.desc",
        }, owner=owner),
    )

    doctors_by_id = {d["doctor_id"]: d for d in all_doctors}
    clinics_by_id = {c["clinic_id"]: c for c in all_clinics}

    # Resolve primary_doctor from the shared cached catalog
    primary_doctor_id = patient.get("primary_care_doctor_id")
    primary_doctor = doctors_by_id.get(primary_doctor_id) if primary_doctor_id else None

    # Step 3: enrich + segment
    upcoming = []
    past = []
    for apt in appointments:
        enriched = enrich_appointment(apt, doctors_by_id, clinics_by_id)
        if apt["date"] >= today and apt.get("status") not in ("Cancelled", "Completed"):
            upcoming.append(enriched)
        else:
            past.append(enriched)

    past = sorted(past, key=lambda x: x.get("date", ""), reverse=True)[:5]

    active_prescriptions = [p for p in prescriptions if p.get("status") == "Active"]
    released_lab_results = [l for l in lab_results if l.get("status") == "Released"]
    pending_lab_results = [l for l in lab_results if l.get("status") == "Pending"]
    # A partially paid invoice still has money owed on it, so it belongs in
    # outstanding — NOT in a third bucket that neither list matches. Before
    # v3.4 "Partially Paid" fell through both filters and the invoice vanished
    # from the response entirely: outstanding_total_sar dropped to zero and the
    # agent told the patient their account was clear while they still owed.
    outstanding_invoices = [
        i for i in invoices if i.get("status") in ("Outstanding", "Partially Paid")
    ]
    paid_invoices = [i for i in invoices if i.get("status") == "Paid"]

    pending_refill_requests = [
        r for r in (refill_reqs or [])
        if r.get("status") in ("Submitted", "Approved", "In Progress")
    ]
    pending_preauth_requests = [
        p for p in (preauth_reqs or [])
        if p.get("status") in ("Submitted", "Under Review", "Approved")
    ]

    allergies = patient.get("allergies") or []
    allergies_alert = bool(allergies)

    total_paid = sum(float(i.get("patient_due_sar") or 0) for i in paid_invoices)
    total_outstanding = sum(float(i.get("patient_due_sar") or 0) for i in outstanding_invoices)

    # --- Refill support -------------------------------------------------
    # A refill needs prescription_id + medication_id + pharmacy_id. The first
    # two live in the prescription rows (medication_id nested inside the
    # `medications` JSONB array); the third had no source at all before v3.3.
    #
    # Two additions below:
    #   1. `pharmacies` — the shared catalog, so the agent can resolve a
    #      pharmacy the patient names into an ID, and quote that pharmacy's
    #      real delivery fee instead of a memorised constant.
    #   2. `refillable_medications` — a FLAT list of exactly what a refill
    #      call needs, so the agent never has to walk the nested JSONB.
    #      Nesting was the other half of why refills were unreliable.

    pharmacies_out = [
        {
            "pharmacy_id": p.get("pharmacy_id"),
            "pharmacy_name_en": p.get("pharmacy_name_en"),
            "pharmacy_name_ar": p.get("pharmacy_name_ar"),
            "type": p.get("type"),
            "linked_clinic_id": p.get("linked_clinic_id"),
            "city": p.get("city"),
            "phone": p.get("phone"),
            "operating_hours": p.get("operating_hours"),
            "home_delivery_available": p.get("home_delivery_available"),
            "home_delivery_fee_sar": p.get("home_delivery_fee_sar"),
            "home_delivery_cities": p.get("home_delivery_cities"),
            "home_delivery_window": p.get("home_delivery_window"),
        }
        for p in (all_pharmacies or [])
    ]

    def _med_field(m: dict, *keys):
        """Read the first key present in a medication object.

        The `medications` JSONB uses the ORIGINAL Title Case seed keys —
        "Medication ID", "Name (EN)", "Dose", "Frequency (EN)", "Refills
        Remaining" — but a Lovable-normalised row may use snake_case instead.
        Pass every plausible spelling; note that the dose field is "Dose",
        NOT "Dosage" (that mismatch shipped a null in v3.3).
        """
        for k in keys:
            v = m.get(k)
            if v is not None:
                return v
        return None

    refillable_medications = []
    for rx in active_prescriptions:
        for m in (rx.get("medications") or []):
            refills = _med_field(m, "refills_remaining", "Refills Remaining")
            refills_total = _med_field(m, "refills_total", "Refills Total")
            try:
                refills = int(refills) if refills is not None else 0
            except (TypeError, ValueError):
                refills = 0
            refillable_medications.append({
                # Everything refill_prescription needs, pre-flattened:
                "prescription_id": rx.get("prescription_id"),
                "medication_id": _med_field(m, "medication_id", "Medication ID"),
                "name_en": _med_field(m, "name_en", "Name (EN)"),
                "name_ar": _med_field(m, "name_ar", "Name (AR)"),
                "dose": _med_field(m, "dose", "Dose", "dosage", "Dosage"),
                "frequency_en": _med_field(m, "frequency_en", "Frequency (EN)", "frequency", "Frequency"),
                "frequency_ar": _med_field(m, "frequency_ar", "Frequency (AR)"),
                "instructions_en": _med_field(m, "instructions_en", "Instructions (EN)"),
                "instructions_ar": _med_field(m, "instructions_ar", "Instructions (AR)"),
                "duration_days": _med_field(m, "duration_days", "Duration Days"),
                "refills_remaining": refills,
                "refills_total": refills_total,
                "can_refill_now": refills > 0,
                "prescription_expiration_date": rx.get("expiration_date"),
                "last_filled_date": rx.get("last_filled_date"),
                "last_filled_pharmacy_id": rx.get("last_filled_pharmacy_id"),
                "prescribing_doctor_id": rx.get("prescribing_doctor_id"),
            })

    return {
        "patient": patient,
        "insurance": insurance,
        "primary_doctor": primary_doctor,
        "allergies": allergies,
        "allergies_alert": allergies_alert,
        "active_conditions_en": patient.get("active_conditions_en") or [],
        "active_conditions_ar": patient.get("active_conditions_ar") or [],
        "upcoming_appointments": upcoming,
        "recent_past_appointments": past,
        "active_prescriptions": active_prescriptions,
        "all_prescriptions": prescriptions,
        "released_lab_results": released_lab_results,
        "pending_lab_results": pending_lab_results,
        "outstanding_invoices": outstanding_invoices,
        "paid_invoices_recent": paid_invoices[:5],
        "payment_history_summary": {
            "total_paid_sar": total_paid,
            "total_outstanding_sar": total_outstanding,
            "paid_invoice_count": len(paid_invoices),
            "outstanding_invoice_count": len(outstanding_invoices),
        },
        "pending_refill_requests": pending_refill_requests,
        "pending_preauth_requests": pending_preauth_requests,
        "medical_history": history,
        "pharmacies": pharmacies_out,
        "refillable_medications": refillable_medications,
    }


# ============================================================
# READ: /doctor  (shared catalog — no tenant scoping needed)
# ============================================================

@app.get("/doctor")
async def get_doctor(
    doctor_id: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
    specialty: Optional[str] = Query(None),
    clinic_id: Optional[str] = Query(None),
    caller_phone: Optional[str] = Query(None, description="Accepted for consistency; doctors is a shared catalog"),
):
    """Look up doctor info from the shared catalog. Robust to the agent passing
    multiple parameters: if the primary filter returns nothing, falls back
    through the remaining provided filters before giving up."""
    if not any([doctor_id, name, specialty, clinic_id]):
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: doctor_id, name, specialty, clinic_id"
        )

    attempts = []
    if doctor_id:
        attempts.append(("doctor_id", {"doctor_id": f"eq.{doctor_id}", "order": "full_name_en.asc"}))
    if name:
        attempts.append(("name", {
            "or": f"(full_name_en.ilike.*{name}*,full_name_ar.ilike.*{name}*)",
            "order": "full_name_en.asc",
        }))
    if specialty:
        attempts.append(("specialty", {"specialty_en": f"ilike.*{specialty}*", "order": "full_name_en.asc"}))
    if clinic_id:
        attempts.append(("clinic_id", {"primary_clinic_id": f"eq.{clinic_id}", "order": "full_name_en.asc"}))

    for filter_name, params in attempts:
        doctors = await sb_get("doctors", params)
        if doctors:
            return {"doctors": doctors, "count": len(doctors), "matched_by": filter_name}

    return {"doctors": [], "count": 0, "matched_by": None,
            "note": f"No doctor matched the provided filters (tried: {[a[0] for a in attempts]})"}


# ============================================================
# READ: /slots  (doctor_availability is PER-TENANT)
# ============================================================

@app.get("/slots")
async def get_slots(
    doctor_id: Optional[str] = Query(None),
    specialty: Optional[str] = Query(None),
    clinic_id: Optional[str] = Query(None),
    city: Optional[str] = Query(None),
    from_date: Optional[str] = Query(None),
    near_date: Optional[str] = Query(None),
    near_window_days: int = Query(7),
    limit: int = Query(5),
    caller_phone: Optional[str] = Query(None, description="Demo tenant routing — WhatsApp sender number"),
):
    """Available appointment slots for THIS tenant, earliest first.

    Doctors and clinics come from the shared cached catalogs (used for both
    filter resolution and enrichment). doctor_availability is per-tenant, so
    each sales person has their own slot grid — no cross-demo booking collisions.
    """
    owner = await resolve_owner(caller_phone)
    today = date.today().isoformat()

    all_doctors = await get_doctors_cached()
    all_clinics = await get_clinics_cached()
    doctors_by_id = {d["doctor_id"]: d for d in all_doctors}
    clinics_by_id = {c["clinic_id"]: c for c in all_clinics}

    # Resolve specialty/city to doctor_ids in memory (shared catalog)
    doctor_filter_ids: Optional[list] = None
    if specialty or city:
        candidates = all_doctors
        if specialty:
            specialty_lower = specialty.lower()
            candidates = [
                d for d in candidates
                if specialty_lower in (d.get("specialty_en") or "").lower()
            ]
        if city:
            city_lower = city.lower()
            city_clinic_ids = {
                c["clinic_id"] for c in all_clinics
                if city_lower in (c.get("city_en") or "").lower()
            }
            candidates = [
                d for d in candidates
                if d.get("primary_clinic_id") in city_clinic_ids
            ]
        doctor_filter_ids = [d["doctor_id"] for d in candidates]
        if not doctor_filter_ids:
            return {"slots": [], "count": 0}

    params = {
        "status": "eq.Open",
        "select": "*",
        "order": "date.asc,start_time.asc",
        "limit": str(limit),
    }
    if doctor_id:
        params["doctor_id"] = f"eq.{doctor_id}"
    elif doctor_filter_ids:
        params["doctor_id"] = f"in.({','.join(doctor_filter_ids)})"
    if clinic_id:
        params["clinic_id"] = f"eq.{clinic_id}"
    if from_date:
        params["date"] = f"gte.{from_date}"
    else:
        params["date"] = f"gte.{today}"

    if near_date:
        try:
            target = datetime.strptime(near_date, "%Y-%m-%d").date()
            window_start = (target - timedelta(days=near_window_days)).isoformat()
            window_end = (target + timedelta(days=near_window_days)).isoformat()
            params["date"] = f"gte.{window_start}"
            params["and"] = f"(date.lte.{window_end})"
        except ValueError:
            pass  # ignore malformed date

    slots = await sb_get("doctor_availability", params, owner=owner)

    enriched = []
    for s in slots:
        d = doctors_by_id.get(s["doctor_id"], {})
        c = clinics_by_id.get(s["clinic_id"], {})
        enriched.append({
            **s,
            "doctor_name_en": d.get("full_name_en"),
            "doctor_name_ar": d.get("full_name_ar"),
            "doctor_specialty_en": d.get("specialty_en"),
            "doctor_consultation_fee_sar": d.get("consultation_fee_sar"),
            "clinic_name_en": c.get("clinic_name_en"),
            "clinic_name_ar": c.get("clinic_name_ar"),
            "clinic_address_en": c.get("address_en"),
        })

    return {"slots": enriched, "count": len(enriched)}


# ============================================================
# READ: /clinic  (shared catalog)
# ============================================================

@app.get("/clinic")
async def get_clinic(
    clinic_id: Optional[str] = Query(None),
    city: Optional[str] = Query(None),
    caller_phone: Optional[str] = Query(None, description="Accepted for consistency; clinics is a shared catalog"),
):
    params = {}
    if clinic_id:
        params["clinic_id"] = f"eq.{clinic_id}"
    elif city:
        params["city_en"] = f"ilike.*{city}*"
    clinics = await sb_get("clinics", params)
    return {"clinics": clinics, "count": len(clinics)}


# ============================================================
# READ: /medication  (shared catalog)
# ============================================================

@app.get("/medication")
async def get_medication(
    medication_id: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
    caller_phone: Optional[str] = Query(None, description="Accepted for consistency; medications is a shared catalog"),
):
    params = {}
    if medication_id:
        params["medication_id"] = f"eq.{medication_id}"
    elif name:
        params["or"] = f"(name_en.ilike.*{name}*,name_ar.ilike.*{name}*)"
    else:
        raise HTTPException(status_code=400, detail="Provide medication_id or name")
    meds = await sb_get("medications_catalog", params)
    return {"medications": meds, "count": len(meds)}


# ============================================================
# READ: /insurance  (shared catalog)
# ============================================================

@app.get("/insurance")
async def get_insurance(
    provider_id: Optional[str] = Query(None),
    caller_phone: Optional[str] = Query(None, description="Accepted for consistency; insurance is a shared catalog"),
):
    if not provider_id:
        raise HTTPException(status_code=400, detail="Provide provider_id")
    plan = await sb_get_one("insurance_providers", {"provider_id": f"eq.{provider_id}"})
    if not plan:
        raise HTTPException(status_code=404, detail="Insurance plan not found")
    return plan


# ============================================================
# READ: /lab-result/fetch
# ============================================================

@app.get("/lab-result/fetch")
async def fetch_lab_document(
    lab_result_id: str = Query(...),
    caller_phone: Optional[str] = Query(None, description="Demo tenant routing — WhatsApp sender number"),
):
    """Fetch the pre-generated PDF document for a released lab result.
    Returns the download URL and metadata. The agent sends this URL via
    send_whatsapp_media to deliver the PDF to the patient."""
    owner = await resolve_owner(caller_phone)

    lab = await sb_get_one("lab_results", {"lab_result_id": f"eq.{lab_result_id}"}, owner=owner)
    if not lab:
        raise HTTPException(status_code=404, detail="Lab result not found")
    if lab.get("status") != "Released":
        raise HTTPException(
            status_code=409,
            detail=f"Lab result is {lab.get('status', 'not Released')} — no PDF available yet"
        )

    doc = await sb_get_one("lab_documents", {"lab_result_id": f"eq.{lab_result_id}"}, owner=owner)
    if not doc:
        raise HTTPException(
            status_code=404,
            detail="No PDF document found for this lab result"
        )

    return {
        "ok": True,
        "lab_result_id": lab_result_id,
        "patient_id": lab.get("patient_id"),
        "test_name_en": lab.get("test_name_en"),
        "test_name_ar": lab.get("test_name_ar"),
        "result_date": lab.get("result_date"),
        "download_url": doc.get("download_url"),
        "filename": doc.get("filename"),
        "mime_type": doc.get("mime_type", "application/pdf"),
    }


# ============================================================
# WRITE: /appointment/book
# ============================================================

async def _book_impl(
    patient_id: str,
    slot_id: str,
    reason: Optional[str],
    apt_type: str,
    owner: str,
) -> dict:
    """Internal booking implementation, tenant-scoped.
    Split out from the endpoint so /appointment/reschedule can call it with
    an already-resolved owner instead of re-resolving."""
    patient = await sb_get_one("patients", {"patient_id": f"eq.{patient_id}"}, owner=owner)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    slot = await sb_get_one("doctor_availability", {"slot_id": f"eq.{slot_id}"}, owner=owner)
    if not slot:
        raise HTTPException(status_code=404, detail="Slot not found")
    if slot.get("status") != "Open":
        raise HTTPException(status_code=409, detail=f"Slot is {slot.get('status')}")
    if slot.get("booked_count", 0) >= slot.get("slot_capacity", 1):
        raise HTTPException(status_code=409, detail="Slot is full")

    # Duplicate guard: same patient, same doctor, same day, still live.
    # The SI tells the agent to check upcoming_appointments first, but every
    # other write path here has a server-side guard (full slot, duplicate
    # registration) and this one didn't — so a dropped SI check meant a
    # silently double-booked patient.
    same_day = await sb_get("appointments", {
        "patient_id": f"eq.{patient_id}",
        "doctor_id": f"eq.{slot['doctor_id']}",
        "date": f"eq.{slot['date']}",
    }, owner=owner)
    live = [a for a in same_day if a.get("status") not in ("Cancelled", "Completed")]
    if live:
        existing = live[0]
        raise HTTPException(
            status_code=409,
            detail={
                "code": "duplicate_appointment",
                "appointment_id": existing.get("appointment_id"),
                "date": existing.get("date"),
                "start_time": existing.get("start_time"),
                "doctor_id": existing.get("doctor_id"),
                "message": (
                    f"Patient already has appointment {existing.get('appointment_id')} "
                    f"with this doctor on {existing.get('date')} at {existing.get('start_time')}"
                ),
            },
        )

    # Generate the next appointment ID WITHIN THIS TENANT. Each tenant counts
    # independently, so APT-H042 in Alice's demo and APT-H042 in Bob's demo are
    # different appointments — that's intentional, they're separate demos.
    existing = await sb_get(
        "appointments",
        {"select": "appointment_id", "order": "appointment_id.desc", "limit": "1"},
        owner=owner,
    )
    next_num = 1
    if existing:
        try:
            last = existing[0]["appointment_id"]
            next_num = int(last.replace("APT-H", "")) + 1
        except (ValueError, KeyError):
            next_num = 1000
    apt_id = f"APT-H{next_num:03d}"

    new_booked = slot.get("booked_count", 0) + 1
    new_status = "Booked" if new_booked >= slot.get("slot_capacity", 1) else "Open"
    await sb_update(
        "doctor_availability",
        {"slot_id": f"eq.{slot_id}"},
        {"booked_count": new_booked, "status": new_status},
        owner=owner,
    )

    apt_row = {
        "appointment_id": apt_id,
        "patient_id": patient_id,
        "doctor_id": slot["doctor_id"],
        "clinic_id": slot["clinic_id"],
        "date": slot["date"],
        "start_time": slot["start_time"],
        "duration_minutes": 30,
        "type": apt_type,
        "reason_for_visit": reason or "",
        "status": "Scheduled",
        "created_date": date.today().isoformat(),
    }
    await sb_insert("appointments", apt_row, owner=owner)

    # Doctor/clinic names from the shared cached catalogs — no extra queries
    all_doctors = await get_doctors_cached()
    all_clinics = await get_clinics_cached()
    doctors_by_id = {d["doctor_id"]: d for d in all_doctors}
    clinics_by_id = {c["clinic_id"]: c for c in all_clinics}
    doctor = doctors_by_id.get(slot["doctor_id"])
    clinic = clinics_by_id.get(slot["clinic_id"])
    desc = (
        f"Booked {slot['date']} {slot['start_time']} with "
        f"{doctor.get('full_name_en') if doctor else slot['doctor_id']} "
        f"at {clinic.get('clinic_name_en') if clinic else slot['clinic_id']}"
    )
    await log_agent_action(patient_id, "Book Appointment", desc, {
        "appointment_id": apt_id,
        "slot_id": slot_id,
        "doctor_id": slot["doctor_id"],
        "clinic_id": slot["clinic_id"],
    }, owner=owner)

    return {"ok": True, "appointment_id": apt_id, "appointment": apt_row}


@app.post("/appointment/book")
async def book_appointment(
    patient_id: str = Body(...),
    slot_id: str = Body(...),
    reason: Optional[str] = Body(None),
    type: str = Body("Initial Consultation"),
    caller_phone: Optional[str] = Body(None),
):
    """Book a slot, increment booked_count, create appointment row."""
    owner = await resolve_owner(caller_phone)
    return await _book_impl(patient_id, slot_id, reason, type, owner)


# ============================================================
# WRITE: /appointment/cancel
# ============================================================

async def _cancel_impl(appointment_id: str, reason: Optional[str], owner: str) -> dict:
    """Internal cancellation implementation, tenant-scoped."""
    apt = await sb_get_one("appointments", {"appointment_id": f"eq.{appointment_id}"}, owner=owner)
    if not apt:
        raise HTTPException(status_code=404, detail="Appointment not found")
    if apt.get("status") in ("Cancelled", "Completed"):
        raise HTTPException(status_code=409, detail=f"Cannot cancel — status is {apt['status']}")

    await sb_update(
        "appointments",
        {"appointment_id": f"eq.{appointment_id}"},
        {"status": "Cancelled",
         "notes": f"{apt.get('notes') or ''}\nCancelled: {reason or 'No reason given'}"},
        owner=owner,
    )

    # Release the slot (find by doctor+date+time within this tenant)
    slot = await sb_get_one("doctor_availability", {
        "doctor_id": f"eq.{apt['doctor_id']}",
        "date": f"eq.{apt['date']}",
        "start_time": f"eq.{apt['start_time']}",
    }, owner=owner)
    if slot:
        new_booked = max(0, slot.get("booked_count", 1) - 1)
        new_status = "Open" if new_booked < slot.get("slot_capacity", 1) else slot.get("status")
        await sb_update(
            "doctor_availability",
            {"slot_id": f"eq.{slot['slot_id']}"},
            {"booked_count": new_booked, "status": new_status},
            owner=owner,
        )

    await log_agent_action(
        apt["patient_id"], "Cancel Appointment",
        f"Cancelled appointment {appointment_id} on {apt['date']} at {apt['start_time']}",
        {"appointment_id": appointment_id, "reason": reason},
        owner=owner,
    )

    return {"ok": True, "appointment_id": appointment_id, "status": "Cancelled"}


@app.post("/appointment/cancel")
async def cancel_appointment(
    appointment_id: str = Body(...),
    reason: Optional[str] = Body(None),
    caller_phone: Optional[str] = Body(None),
):
    owner = await resolve_owner(caller_phone)
    return await _cancel_impl(appointment_id, reason, owner)


# ============================================================
# WRITE: /appointment/reschedule
# ============================================================

@app.post("/appointment/reschedule")
async def reschedule_appointment(
    appointment_id: str = Body(...),
    new_slot_id: str = Body(...),
    reason: Optional[str] = Body(None),
    caller_phone: Optional[str] = Body(None),
):
    """Move an appointment to a new slot. Books the new slot FIRST — if that
    fails, nothing is cancelled and the patient keeps their original booking."""
    owner = await resolve_owner(caller_phone)

    # 1. look up existing appointment
    apt = await sb_get_one("appointments", {"appointment_id": f"eq.{appointment_id}"}, owner=owner)
    if not apt:
        raise HTTPException(status_code=404, detail="Appointment not found")
    patient_id = apt["patient_id"]
    apt_type = apt.get("type", "Follow-up")

    # 2. book new
    book_result = await _book_impl(
        patient_id=patient_id,
        slot_id=new_slot_id,
        reason=apt.get("reason_for_visit"),
        apt_type=apt_type,
        owner=owner,
    )

    # 3. cancel old (only after new is booked successfully)
    await _cancel_impl(
        appointment_id=appointment_id,
        reason=f"Rescheduled to {book_result['appointment_id']}",
        owner=owner,
    )

    return {
        "ok": True,
        "old_appointment_id": appointment_id,
        "new_appointment_id": book_result["appointment_id"],
    }


# ============================================================
# WRITE: /prescription/refill
# ============================================================

@app.post("/prescription/refill")
async def refill_prescription(
    prescription_id: str = Body(...),
    medication_id: str = Body(...),
    pharmacy_id: str = Body(...),
    delivery_method: str = Body("Pickup"),  # "Pickup" | "Home Delivery"
    caller_phone: Optional[str] = Body(None),
):
    owner = await resolve_owner(caller_phone)

    rx = await sb_get_one("prescriptions", {"prescription_id": f"eq.{prescription_id}"}, owner=owner)
    if not rx:
        raise HTTPException(status_code=404, detail="Prescription not found")
    if rx.get("status") != "Active":
        raise HTTPException(status_code=409, detail=f"Prescription is {rx.get('status')}")

    # Find the specific medication in the JSON array, decrement refills.
    # Tolerates both snake_case and original "Title Case" JSONB keys.
    meds = rx.get("medications", [])
    med_name = None
    found = False
    for m in meds:
        m_id = m.get("medication_id") or m.get("Medication ID")
        if m_id == medication_id:
            refills_remaining = m.get("refills_remaining")
            if refills_remaining is None:
                refills_remaining = m.get("Refills Remaining", 0)
            if refills_remaining <= 0:
                raise HTTPException(status_code=409, detail="No refills remaining")
            if "refills_remaining" in m:
                m["refills_remaining"] = refills_remaining - 1
            else:
                m["Refills Remaining"] = refills_remaining - 1
            med_name = m.get("name_en") or m.get("Name (EN)")
            found = True
            break

    if not found:
        raise HTTPException(status_code=404, detail=f"Medication {medication_id} not in this prescription")

    await sb_update(
        "prescriptions",
        {"prescription_id": f"eq.{prescription_id}"},
        {"medications": meds,
         "last_filled_date": date.today().isoformat(),
         "last_filled_pharmacy_id": pharmacy_id},
        owner=owner,
    )

    refill_row = {
        "prescription_id": prescription_id,
        "patient_id": rx["patient_id"],
        "medication_id": medication_id,
        "medication_name_en": med_name,
        "pharmacy_id": pharmacy_id,
        "delivery_method": delivery_method,
        "status": "Submitted",
        "requested_by": "Agent",
    }
    await sb_insert("refill_requests", refill_row, owner=owner)

    # pharmacies is a SHARED catalog — served from the in-process cache,
    # so this costs nothing on a warm call.
    all_pharmacies = await get_pharmacies_cached()
    pharmacy = next(
        (p for p in all_pharmacies if p.get("pharmacy_id") == pharmacy_id), None
    )
    pharmacy_name = pharmacy.get("pharmacy_name_en") if pharmacy else pharmacy_id

    await log_agent_action(
        rx["patient_id"], "Refill Request",
        f"Refill requested for {med_name} → {pharmacy_name} ({delivery_method})",
        {"prescription_id": prescription_id, "medication_id": medication_id,
         "pharmacy_id": pharmacy_id, "delivery_method": delivery_method},
        owner=owner,
    )

    return {"ok": True, "medication_name": med_name, "pharmacy": pharmacy_name, "delivery_method": delivery_method}


# ============================================================
# WRITE: /invoice/payment
# ============================================================

@app.post("/invoice/payment")
async def record_payment(
    invoice_id: str = Body(...),
    amount_sar: float = Body(...),
    payment_method: str = Body("Credit Card"),
    caller_phone: Optional[str] = Body(None),
):
    owner = await resolve_owner(caller_phone)

    inv = await sb_get_one("invoices", {"invoice_id": f"eq.{invoice_id}"}, owner=owner)
    if not inv:
        raise HTTPException(status_code=404, detail="Invoice not found")
    if inv.get("status") == "Paid":
        raise HTTPException(status_code=409, detail="Invoice already paid")

    if amount_sar <= 0:
        raise HTTPException(status_code=400, detail="Payment amount must be greater than zero")

    due = float(inv.get("patient_due_sar") or 0)

    # Decrement the balance. Before v3.4 only `status` moved and
    # patient_due_sar was left at its original value, so a partial payment
    # left the full amount showing as owed while the invoice simultaneously
    # dropped out of both the outstanding and paid lists.
    remaining = round(max(0.0, due - amount_sar), 2)
    overpaid = round(max(0.0, amount_sar - due), 2)

    if remaining > 0:
        new_status = "Partially Paid"
        note_extra = (
            f"\nPartial payment {amount_sar} SAR on {date.today().isoformat()} "
            f"({payment_method}) — SAR {remaining} remaining"
        )
    else:
        new_status = "Paid"
        note_extra = f"\nPaid in full {amount_sar} SAR on {date.today().isoformat()} ({payment_method})"

    await sb_update(
        "invoices",
        {"invoice_id": f"eq.{invoice_id}"},
        {"status": new_status,
         "patient_due_sar": remaining,
         "payment_method": payment_method,
         "payment_date": date.today().isoformat(),
         "notes_en": (inv.get("notes_en") or "") + note_extra},
        owner=owner,
    )

    desc = (
        f"Payment of SAR {amount_sar} via {payment_method} for invoice {invoice_id}"
        + (f" — SAR {remaining} still outstanding" if remaining > 0 else " — paid in full")
    )
    await log_agent_action(
        inv["patient_id"], "Payment Recorded", desc,
        {"invoice_id": invoice_id, "amount_sar": amount_sar, "method": payment_method,
         "remaining_sar": remaining, "status": new_status},
        owner=owner,
    )

    return {
        "ok": True,
        "invoice_id": invoice_id,
        "status": new_status,
        "amount_paid_sar": amount_sar,
        "remaining_sar": remaining,
        "overpaid_sar": overpaid if overpaid > 0 else None,
        "fully_paid": remaining == 0,
    }


# ============================================================
# WRITE: /profile/update
# ============================================================

ALLOWED_PROFILE_FIELDS = {"phone", "email", "address_en", "address_ar", "city_en", "city_ar"}


@app.post("/profile/update")
async def update_profile(
    patient_id: str = Body(...),
    field: str = Body(...),
    new_value: str = Body(...),
    caller_phone: Optional[str] = Body(None),
):
    if field not in ALLOWED_PROFILE_FIELDS:
        raise HTTPException(status_code=400, detail=f"Field {field} not updatable. Allowed: {ALLOWED_PROFILE_FIELDS}")

    owner = await resolve_owner(caller_phone)

    patient = await sb_get_one("patients", {"patient_id": f"eq.{patient_id}"}, owner=owner)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Canonicalize phone updates so later lookups match
    value_to_store = new_value
    if field == "phone":
        value_to_store = normalize_phone(new_value) or new_value

    old_value = patient.get(field)
    await sb_update(
        "patients", {"patient_id": f"eq.{patient_id}"}, {field: value_to_store}, owner=owner
    )

    await log_agent_action(
        patient_id, "Profile Updated",
        f"{field}: {old_value} → {value_to_store}",
        {"field": field, "old_value": old_value, "new_value": value_to_store},
        owner=owner,
    )

    return {"ok": True, "patient_id": patient_id, "field": field, "new_value": value_to_store}


# ============================================================
# WRITE: /preauth/request
# ============================================================

@app.post("/preauth/request")
async def request_preauth(
    patient_id: str = Body(...),
    procedure_name: str = Body(...),
    doctor_id: Optional[str] = Body(None),
    caller_phone: Optional[str] = Body(None),
):
    owner = await resolve_owner(caller_phone)

    patient = await sb_get_one("patients", {"patient_id": f"eq.{patient_id}"}, owner=owner)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    row = {
        "patient_id": patient_id,
        "doctor_id": doctor_id,
        "procedure_name": procedure_name,
        "insurance_provider_id": patient.get("insurance_provider_id"),
        "status": "Submitted",
        "requested_by": "Agent",
    }
    await sb_insert("preauth_requests", row, owner=owner)

    await log_agent_action(
        patient_id, "Pre-Auth Requested",
        f"Pre-authorization request submitted for {procedure_name}",
        {"procedure": procedure_name, "doctor_id": doctor_id},
        owner=owner,
    )

    return {"ok": True, "procedure": procedure_name, "status": "Submitted",
            "estimated_response_days": "1-5 business days"}


# ============================================================
# WRITE: /lab-result/release
# ============================================================

@app.post("/lab-result/release")
async def release_lab_result(
    lab_result_id: str = Body(...),
    released_by: str = Body("Doctor"),
    caller_phone: Optional[str] = Body(None),
):
    owner = await resolve_owner(caller_phone)

    lab = await sb_get_one("lab_results", {"lab_result_id": f"eq.{lab_result_id}"}, owner=owner)
    if not lab:
        raise HTTPException(status_code=404, detail="Lab result not found")
    if lab.get("status") == "Released":
        raise HTTPException(status_code=409, detail="Already released")

    await sb_update(
        "lab_results",
        {"lab_result_id": f"eq.{lab_result_id}"},
        {"status": "Released",
         "result_date": date.today().isoformat(),
         "released_at": datetime.now().astimezone().isoformat(),
         "released_by": released_by},
        owner=owner,
    )

    await log_agent_action(
        lab["patient_id"], "Lab Result Released",
        f"Released {lab.get('test_name_en')} to patient",
        {"lab_result_id": lab_result_id, "released_by": released_by},
        owner=owner,
    )

    return {"ok": True, "lab_result_id": lab_result_id, "status": "Released"}


# ============================================================
# WRITE: /patient/register
# ============================================================

_REGISTRATION_REASONS = {
    "current_concern",
    "new_primary",
    "follow_up",
    "wellness",
    "other",
}

_INSURANCE_STATUSES = {
    "has_provider",
    "has_insurance_unknown_provider",
    "self_pay",
    "unknown",
}


def _compose_intake_notes(
    registration_date: str,
    reason: Optional[str],
    concern_note: Optional[str],
    insurance_status: Optional[str],
    insurance_provider: Optional[str],
) -> str:
    """Build a clean, human-readable staff-facing summary of the registration.
    The staff portal renders this prominently on Pending Verification patients."""
    parts = [f"Patient self-registered via WhatsApp on {registration_date}."]

    concern_clean = (concern_note or "").strip()
    if reason == "current_concern":
        if concern_clean:
            parts.append(f"Reports: {concern_clean}.")
        else:
            parts.append("Reports a current health concern (details to confirm at intake).")
    elif reason == "follow_up":
        if concern_clean:
            parts.append(f"Follow-up care needed: {concern_clean}.")
        else:
            parts.append("Follow-up care needed (details to confirm at intake).")
    elif reason == "new_primary":
        parts.append("Looking for a new primary clinic — switching providers.")
    elif reason == "wellness":
        parts.append("Reached out for routine wellness/checkup.")
    elif reason == "other":
        if concern_clean:
            parts.append(f"Other reason: {concern_clean}.")
        else:
            parts.append("Reached out for an unspecified reason.")

    provider_clean = (insurance_provider or "").strip()
    if insurance_status == "has_provider" and provider_clean:
        parts.append(f"Has {provider_clean} insurance.")
    elif insurance_status == "has_insurance_unknown_provider":
        parts.append("Has insurance but doesn't know plan details — needs verification.")
    elif insurance_status == "self_pay":
        parts.append("Self-pay (no insurance on file).")

    if reason in ("current_concern", "follow_up"):
        parts.append("Recommend prompt GP follow-up to assess.")
    elif reason == "new_primary":
        parts.append("Recommend GP intro visit once insurance is verified.")
    elif reason == "wellness":
        parts.append("Recommend routine wellness visit at patient's convenience.")

    return " ".join(parts)


@app.post("/patient/register")
async def register_new_patient(
    national_id: str = Body(...),
    email: str = Body(...),
    phone: str = Body(...),
    full_name_en: Optional[str] = Body(None),
    full_name_ar: Optional[str] = Body(None),
    registration_reason: Optional[str] = Body(None),
    registration_concern_note: Optional[str] = Body(None),
    registration_insurance_provider: Optional[str] = Body(None),
    registration_insurance_status: Optional[str] = Body(None),
    caller_phone: Optional[str] = Body(None),
):
    """Register a new patient via the WhatsApp agent, within the caller's tenant.

    Creates a patient row with status 'Pending Verification' and composes a
    staff-facing intake_notes summary. A staff member follows up within 1
    business day to verify and finalize.

    Duplicate detection is PER-TENANT: the same National ID can exist in two
    different sales people's demos without conflict. That's intentional — each
    demo is independent.
    """
    owner = await resolve_owner(caller_phone)

    # === Validation ===
    nid = (national_id or "").strip()
    if not nid.isdigit() or len(nid) != 10:
        raise HTTPException(status_code=400, detail="National ID must be exactly 10 digits")

    first_digit = nid[0]
    if first_digit == "1":
        id_type = "Saudi"
    elif first_digit == "2":
        id_type = "Iqama"
    else:
        raise HTTPException(
            status_code=400,
            detail="National ID must start with 1 (Saudi National ID) or 2 (Iqama)"
        )

    name_en = (full_name_en or "").strip()
    name_ar = (full_name_ar or "").strip()
    if not name_en and not name_ar:
        raise HTTPException(
            status_code=400,
            detail="At least one of full_name_en or full_name_ar is required"
        )

    em = (email or "").strip()
    if "@" not in em or "." not in em.split("@")[-1]:
        raise HTTPException(status_code=400, detail="A valid email address is required")

    # Canonicalize the patient's phone so downstream get_patient_data(phone=...)
    # lookups match regardless of '+' handling.
    ph = normalize_phone(phone)
    if not ph:
        raise HTTPException(status_code=400, detail="A valid phone number is required")

    reason = (registration_reason or "").strip().lower() or None
    if reason and reason not in _REGISTRATION_REASONS:
        raise HTTPException(
            status_code=400,
            detail=f"registration_reason must be one of: {sorted(_REGISTRATION_REASONS)}"
        )

    insurance_status = (registration_insurance_status or "").strip().lower() or None
    if insurance_status and insurance_status not in _INSURANCE_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"registration_insurance_status must be one of: {sorted(_INSURANCE_STATUSES)}"
        )

    concern_note = (registration_concern_note or "").strip() or None
    insurance_provider = (registration_insurance_provider or "").strip() or None

    # === Duplicate check by national_id, WITHIN THIS TENANT ===
    existing = await sb_get_one("patients", {"national_id": f"eq.{nid}"}, owner=owner)
    if existing:
        # Structured 409 so the agent can extract patient_id reliably from
        # detail.patient_id rather than parsing the message string.
        raise HTTPException(
            status_code=409,
            detail={
                "code": "already_registered",
                "patient_id": existing.get("patient_id"),
                "patient_status": existing.get("patient_status"),
                "message": f"A patient with this National ID is already registered (Patient ID: {existing.get('patient_id')})",
            },
        )

    # === Generate next sequential patient_id WITHIN THIS TENANT ===
    all_patients = await sb_get("patients", {"select": "patient_id"}, owner=owner)
    max_num = 0
    for p in all_patients:
        pid = p.get("patient_id", "")
        if pid.startswith("PAT-"):
            try:
                n = int(pid.split("-")[1])
                if n > max_num:
                    max_num = n
            except (ValueError, IndexError):
                pass
    new_patient_id = f"PAT-{max_num + 1:03d}"

    preferred_language = "Arabic" if name_ar else "English"

    today_iso = date.today().isoformat()
    try:
        date_display = date.today().strftime("%B %-d, %Y")
    except ValueError:
        date_display = date.today().strftime("%B %d, %Y").replace(" 0", " ")

    intake_notes = _compose_intake_notes(
        registration_date=date_display,
        reason=reason,
        concern_note=concern_note,
        insurance_status=insurance_status,
        insurance_provider=insurance_provider,
    )

    row = {
        "patient_id": new_patient_id,
        "national_id": nid,
        "id_type": id_type,
        "full_name_en": name_en or None,
        "full_name_ar": name_ar or None,
        "email": em,
        "phone": ph,
        "preferred_language": preferred_language,
        "patient_status": "Pending Verification",
        "registered_since": today_iso,
        "allergies": [],
        "active_conditions_en": [],
        "active_conditions_ar": [],
        "registration_reason": reason,
        "registration_concern_note": concern_note,
        "registration_insurance_provider": insurance_provider,
        "registration_insurance_status": insurance_status,
        "intake_notes": intake_notes,
        "demo_notes": "Self-registered via WhatsApp agent",
    }

    result = await sb_insert("patients", row, owner=owner)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to register patient")

    display_name = name_en or name_ar
    reason_label = (reason or "unspecified").replace("_", " ")
    await log_agent_action(
        new_patient_id,
        "New Patient Registration",
        f"New patient registered via WhatsApp: {display_name} ({id_type} ID, reason: {reason_label})",
        {
            "national_id_last4": nid[-4:],
            "id_type": id_type,
            "email": em,
            "phone": ph,
            "registration_reason": reason,
            "registration_insurance_status": insurance_status,
        },
        owner=owner,
    )

    return {
        "ok": True,
        "patient_id": new_patient_id,
        "patient_status": "Pending Verification",
        "id_type": id_type,
        "full_name_en": name_en or None,
        "full_name_ar": name_ar or None,
        "registration_reason": reason,
        "registration_insurance_status": insurance_status,
        "intake_notes": intake_notes,
        "message": f"Registered as {new_patient_id}. A team member will reach out within 1 business day to verify and finalize.",
    }


# ============================================================
# Dev entrypoint
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
