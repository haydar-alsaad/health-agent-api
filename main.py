"""
Al-Noor Healthcare Agent API - v2.0

Architecture: Supabase-backed via httpx REST + service role key.
Endpoints:
  Read:
    GET /health                      - status + data load counts
    GET /patient                     - workhorse: full patient package (parallelized)
    GET /doctor                      - doctor info
    GET /slots                       - available appointment slots
    GET /clinic                      - clinic info
    GET /medication                  - medication catalog lookup
    GET /insurance                   - insurance plan lookup

  Write:
    POST /appointment/book           - book a slot, create appointment
    POST /appointment/cancel         - cancel appointment, release slot
    POST /appointment/reschedule     - cancel + book in one transaction
    POST /prescription/refill        - create refill request, decrement refills
    POST /invoice/payment            - record payment, mark invoice Paid
    POST /profile/update             - update phone/email/address only
    POST /preauth/request            - create pre-auth request
    POST /lab-result/release         - flip Pending → Released (portal-side)

Lessons from Education applied:
  - asyncio.gather in /patient for parallel Supabase fetches
  - Every write inserts an agent_actions audit row
  - Indexes assumed on filter columns
  - Auto-warm cron-friendly: /patient?patient_id=PAT-002 is cheap & fast
"""
import asyncio
import json
import os
import sys
from datetime import date, datetime, timedelta
from typing import Optional, Any
import httpx
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# Config
# ============================================================
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
SEED_ON_BOOT = os.environ.get("SEED_ON_BOOT", "true").lower() == "true"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("WARNING: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set. API will fail.")

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation",
}

# ============================================================
# App
# ============================================================
app = FastAPI(title="Al-Noor Health Agent API", version="2.0")
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
    http_client = httpx.AsyncClient(
        timeout=30.0,
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
    )
    if SEED_ON_BOOT:
        await seed_if_empty()


@app.on_event("shutdown")
async def shutdown():
    global http_client
    if http_client:
        await http_client.aclose()


# ============================================================
# Supabase REST helpers
# ============================================================
async def sb_get(table: str, params: Optional[dict] = None) -> list:
    """GET from Supabase REST. Returns list of rows."""
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    try:
        r = await http_client.get(url, headers=HEADERS, params=params or {})
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        print(f"sb_get error {table}: {e.response.status_code} {e.response.text[:200]}")
        return []
    except Exception as e:
        print(f"sb_get exception {table}: {e}")
        return []


async def sb_get_one(table: str, params: Optional[dict] = None) -> Optional[dict]:
    """GET single row from Supabase. Returns dict or None."""
    rows = await sb_get(table, params)
    return rows[0] if rows else None


# ============================================================
# In-process cache for rarely-changing reference tables
# ============================================================
# doctors (21 rows) and clinics (3 rows) change very rarely — a doctor
# is added every couple of months at most. Refetching the whole table on
# every /patient call is wasted work that adds ~1 Supabase round-trip
# to the critical path. Cache them in memory with a short TTL so the
# first request after a deploy is normal, but subsequent requests skip
# the network entirely.
#
# Per-replica cache (Railway may run multiple). Each replica warms on
# its first request. Cache lives until process restart or TTL expires.
from time import monotonic as _monotonic

_REF_CACHE_TTL = 60.0  # seconds
_reference_cache: dict[str, dict] = {
    "doctors": {"data": None, "ts": 0.0},
    "clinics": {"data": None, "ts": 0.0},
}


async def get_doctors_cached() -> list:
    """Return the doctors table from cache or fetch+cache it."""
    c = _reference_cache["doctors"]
    if c["data"] is None or (_monotonic() - c["ts"]) > _REF_CACHE_TTL:
        # Only the fields needed for enrichment + doctor info lookups.
        c["data"] = await sb_get("doctors", {
            "select": "doctor_id,full_name_en,full_name_ar,specialty_en,specialty_ar,sub_specialty_en,sub_specialty_ar,title_en,title_ar,primary_clinic_id,languages,years_of_experience,qualifications,consultation_fee_sar,followup_fee_sar,bio_en,bio_ar,status"
        })
        c["ts"] = _monotonic()
    return c["data"]


async def get_clinics_cached() -> list:
    """Return the clinics table from cache or fetch+cache it."""
    c = _reference_cache["clinics"]
    if c["data"] is None or (_monotonic() - c["ts"]) > _REF_CACHE_TTL:
        c["data"] = await sb_get("clinics", {"select": "*"})
        c["ts"] = _monotonic()
    return c["data"]


async def sb_insert(table: str, payload: dict | list) -> Any:
    """INSERT into Supabase. Returns inserted row(s)."""
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    try:
        r = await http_client.post(url, headers=HEADERS, json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        print(f"sb_insert error {table}: {e.response.status_code} {e.response.text[:300]}")
        raise HTTPException(status_code=500, detail=f"Insert to {table} failed: {e.response.text[:200]}")


async def sb_update(table: str, params: dict, payload: dict) -> Any:
    """UPDATE rows in Supabase matching params (using PostgREST filter syntax)."""
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    try:
        r = await http_client.patch(url, headers=HEADERS, params=params, json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        print(f"sb_update error {table}: {e.response.status_code} {e.response.text[:300]}")
        raise HTTPException(status_code=500, detail=f"Update {table} failed: {e.response.text[:200]}")


async def log_agent_action(
    patient_id: Optional[str],
    action_type: str,
    description: str,
    metadata: Optional[dict] = None,
    status: str = "Success",
):
    """Insert into agent_actions for the Live Activity Drawer."""
    try:
        await sb_insert("agent_actions", {
            "patient_id": patient_id,
            "action_type": action_type,
            "description": description,
            "metadata": metadata or {},
            "status": status,
        })
    except Exception as e:
        # Audit log failures should not break the parent operation
        print(f"agent_actions log failed: {e}")


# ============================================================
# Data seeding (run once on boot if tables empty)
# ============================================================
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
    # Mapping rules based on schema in Lovable prompt
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
    """Check if patients table is empty; if so, seed all tables from JSON."""
    try:
        existing = await sb_get("patients", {"select": "patient_id", "limit": "1"})
        if existing:
            print(f"[seed] Database already has data ({len(existing)} patient(s) found). Skipping seed.")
            return
        print("[seed] Empty database detected. Seeding from JSON files...")
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
            # Batch insert in chunks of 500 (Supabase REST limit safety)
            for i in range(0, len(mapped), 500):
                chunk = mapped[i:i+500]
                try:
                    await sb_insert(table, chunk)
                except Exception as e:
                    print(f"[seed] {table} batch {i} failed: {e}")
            print(f"[seed] {table}: {len(mapped)} rows")
        print("[seed] Done.")
    except Exception as e:
        print(f"[seed] failed: {e}")


# ============================================================
# Helper: enrich rows with related data
# ============================================================
async def enrich_appointment(apt: dict, doctors_by_id: dict, clinics_by_id: dict) -> dict:
    """Add doctor + clinic names to an appointment row."""
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
        "version": "2.0",
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_KEY),
    }


@app.get("/health")
async def health():
    """Quick health check (used by cron-job.org for warming)."""
    if not (SUPABASE_URL and SUPABASE_KEY):
        return {"status": "degraded", "reason": "Supabase env not configured"}
    # Lightweight ping — just confirm Supabase is reachable on one table.
    try:
        await sb_get("patients", {"select": "patient_id", "limit": "1"})
        return {"status": "ok"}
    except Exception as e:
        return {"status": "degraded", "reason": str(e)[:200]}


# ============================================================
# READ: /patient (the workhorse — parallelized)
# ============================================================
@app.get("/patient")
async def get_patient(
    patient_id: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
    phone: Optional[str] = Query(None),
):
    """Returns full patient package: profile, insurance, primary doctor,
    appointments, prescriptions, lab results, invoices, medical history.

    All sub-fetches run in parallel via asyncio.gather."""
    # Step 1: find the patient
    if patient_id:
        patient = await sb_get_one("patients", {"patient_id": f"eq.{patient_id}"})
    elif name:
        # Case-insensitive partial match on EN or AR name
        patient = await sb_get_one("patients", {
            "or": f"(full_name_en.ilike.*{name}*,full_name_ar.ilike.*{name}*)"
        })
    elif phone:
        patient = await sb_get_one("patients", {"phone": f"eq.{phone}"})
    else:
        raise HTTPException(status_code=400, detail="Provide patient_id, name, or phone")

    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    pid = patient["patient_id"]

    # Step 2: fetch ALL related data in parallel. Doctors and clinics come from
    # an in-process cache (refreshed every 60s) so we skip the Supabase round-trip
    # on the vast majority of warm calls — the tables change rarely.
    #
    # Lessons from Education (main.py line 322): "These calls have no dependencies
    # on each other, so we batch them with asyncio.gather() to overlap network latency
    # instead of stacking it. On warm Supabase this drops ~800ms-1s vs sequential calls."
    today = date.today().isoformat()
    (
        appointments,
        prescriptions,
        lab_results,
        invoices,
        history,
        all_doctors,
        all_clinics,
        insurance,
        primary_doctor,
        refill_reqs,
        preauth_reqs,
    ) = await asyncio.gather(
        sb_get("appointments", {
            "patient_id": f"eq.{pid}",
            "order": "date.asc,start_time.asc",
        }),
        sb_get("prescriptions", {
            "patient_id": f"eq.{pid}",
            "order": "issued_date.desc",
        }),
        sb_get("lab_results", {
            "patient_id": f"eq.{pid}",
            "order": "order_date.desc",
        }),
        sb_get("invoices", {
            "patient_id": f"eq.{pid}",
            "order": "issue_date.desc",
        }),
        sb_get("medical_history", {
            "patient_id": f"eq.{pid}",
            "order": "event_date.desc",
        }),
        get_doctors_cached(),
        get_clinics_cached(),
        sb_get_one("insurance_providers", {
            "provider_id": f"eq.{patient.get('insurance_provider_id', '')}"
        }) if patient.get("insurance_provider_id") else asyncio.sleep(0, result=None),
        sb_get_one("doctors", {
            "doctor_id": f"eq.{patient.get('primary_care_doctor_id', '')}"
        }) if patient.get("primary_care_doctor_id") else asyncio.sleep(0, result=None),
        sb_get("refill_requests", {
            "patient_id": f"eq.{pid}",
            "order": "requested_at.desc",
        }),
        sb_get("preauth_requests", {
            "patient_id": f"eq.{pid}",
            "order": "requested_at.desc",
        }),
    )

    doctors_by_id = {d["doctor_id"]: d for d in all_doctors}
    clinics_by_id = {c["clinic_id"]: c for c in all_clinics}

    # Step 3: enrich + segment
    upcoming = []
    past = []
    for apt in appointments:
        enriched = await enrich_appointment(apt, doctors_by_id, clinics_by_id)
        if apt["date"] >= today and apt.get("status") not in ("Cancelled", "Completed"):
            upcoming.append(enriched)
        else:
            past.append(enriched)
    # past sorted reverse-chrono; keep most recent 5
    past = sorted(past, key=lambda x: x.get("date", ""), reverse=True)[:5]

    active_prescriptions = [p for p in prescriptions if p.get("status") == "Active"]
    released_lab_results = [l for l in lab_results if l.get("status") == "Released"]
    pending_lab_results = [l for l in lab_results if l.get("status") == "Pending"]
    outstanding_invoices = [i for i in invoices if i.get("status") == "Outstanding"]
    paid_invoices = [i for i in invoices if i.get("status") == "Paid"]

    # Pending refill and preauth requests — these are in-flight workflows the agent
    # should be aware of before submitting duplicates.
    pending_refill_requests = [
        r for r in (refill_reqs or [])
        if r.get("status") in ("Submitted", "Approved", "In Progress")
    ]
    pending_preauth_requests = [
        p for p in (preauth_reqs or [])
        if p.get("status") in ("Submitted", "Under Review", "Approved")
    ]

    # Compute allergies alert
    allergies = patient.get("allergies") or []
    allergies_alert = bool(allergies)

    # Payment history summary
    total_paid = sum(float(i.get("patient_due_sar") or 0) for i in paid_invoices)
    total_outstanding = sum(float(i.get("patient_due_sar") or 0) for i in outstanding_invoices)

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
    }


# ============================================================
# READ: /doctor
# ============================================================
@app.get("/doctor")
async def get_doctor(
    doctor_id: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
    specialty: Optional[str] = Query(None),
    clinic_id: Optional[str] = Query(None),
):
    """
    Look up doctor info. Robust to the agent passing multiple parameters
    (e.g. doctor_id + name): if the primary filter returns nothing, falls
    back through the remaining provided filters before giving up.
    """
    if not any([doctor_id, name, specialty, clinic_id]):
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: doctor_id, name, specialty, clinic_id"
        )

    # Try filters in priority order. If a filter returns results, return them.
    # If empty, fall through to the next provided filter.
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

    last_filter = None
    for filter_name, params in attempts:
        last_filter = filter_name
        doctors = await sb_get("doctors", params)
        if doctors:
            return {"doctors": doctors, "count": len(doctors), "matched_by": filter_name}

    # All provided filters returned empty
    return {"doctors": [], "count": 0, "matched_by": None,
            "note": f"No doctor matched the provided filters (tried: {[a[0] for a in attempts]})"}


# ============================================================
# READ: /slots
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
):
    """Available appointment slots, sorted by earliest first (or by proximity to near_date)."""
    today = date.today().isoformat()

    # If specialty or city given, first resolve to doctor_ids
    doctor_filter_ids: Optional[list[str]] = None
    if specialty or city:
        d_params = {"select": "doctor_id,primary_clinic_id"}
        if specialty:
            d_params["specialty_en"] = f"ilike.*{specialty}*"
        candidate_docs = await sb_get("doctors", d_params)
        if city:
            # Need clinic city — fetch clinics in that city
            clinics_in_city = await sb_get("clinics", {
                "city_en": f"ilike.*{city}*",
                "select": "clinic_id",
            })
            city_clinic_ids = {c["clinic_id"] for c in clinics_in_city}
            candidate_docs = [d for d in candidate_docs if d.get("primary_clinic_id") in city_clinic_ids]
        doctor_filter_ids = [d["doctor_id"] for d in candidate_docs]
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
        # default: today onward
        params["date"] = f"gte.{today}"

    # near_date overrides from_date for proximity sort
    if near_date:
        try:
            target = datetime.strptime(near_date, "%Y-%m-%d").date()
            window_start = (target - timedelta(days=near_window_days)).isoformat()
            window_end = (target + timedelta(days=near_window_days)).isoformat()
            params["date"] = f"gte.{window_start}"
            params["and"] = f"(date.lte.{window_end})"
        except ValueError:
            pass  # ignore malformed date

    slots = await sb_get("doctor_availability", params)

    # Enrich with doctor + clinic names
    all_doctors = await sb_get("doctors", {})
    all_clinics = await sb_get("clinics", {})
    doctors_by_id = {d["doctor_id"]: d for d in all_doctors}
    clinics_by_id = {c["clinic_id"]: c for c in all_clinics}
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
# READ: /clinic
# ============================================================
@app.get("/clinic")
async def get_clinic(
    clinic_id: Optional[str] = Query(None),
    city: Optional[str] = Query(None),
):
    params = {}
    if clinic_id:
        params["clinic_id"] = f"eq.{clinic_id}"
    elif city:
        params["city_en"] = f"ilike.*{city}*"
    clinics = await sb_get("clinics", params)
    return {"clinics": clinics, "count": len(clinics)}


# ============================================================
# READ: /medication
# ============================================================
@app.get("/medication")
async def get_medication(
    medication_id: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
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
# READ: /insurance
# ============================================================
@app.get("/insurance")
async def get_insurance(provider_id: Optional[str] = Query(None)):
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
async def fetch_lab_document(lab_result_id: str = Query(...)):
    """
    Fetch the pre-generated PDF document for a released lab result.
    Returns the download URL and metadata. The agent sends this URL via
    send_whatsapp_media to deliver the PDF to the patient.
    """
    # 1) Verify the lab result exists and is Released
    lab = await sb_get_one("lab_results", {"lab_result_id": f"eq.{lab_result_id}"})
    if not lab:
        raise HTTPException(status_code=404, detail="Lab result not found")

    if lab.get("status") != "Released":
        raise HTTPException(
            status_code=409,
            detail=f"Lab result is {lab.get('status', 'not Released')} — no PDF available yet"
        )

    # 2) Find the associated PDF document
    doc = await sb_get_one("lab_documents", {"lab_result_id": f"eq.{lab_result_id}"})
    if not doc:
        raise HTTPException(
            status_code=404,
            detail="No PDF document found for this lab result"
        )

    # 3) Return the URL and metadata
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
@app.post("/appointment/book")
async def book_appointment(
    patient_id: str = Body(...),
    slot_id: str = Body(...),
    reason: Optional[str] = Body(None),
    type: str = Body("Initial Consultation"),
):
    """Book a slot, increment Booked Count, create appointment row."""
    # 1. Validate patient + slot exist
    patient = await sb_get_one("patients", {"patient_id": f"eq.{patient_id}"})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    slot = await sb_get_one("doctor_availability", {"slot_id": f"eq.{slot_id}"})
    if not slot:
        raise HTTPException(status_code=404, detail="Slot not found")
    if slot.get("status") != "Open":
        raise HTTPException(status_code=409, detail=f"Slot is {slot.get('status')}")
    if slot.get("booked_count", 0) >= slot.get("slot_capacity", 1):
        raise HTTPException(status_code=409, detail="Slot is full")

    # 2. Generate appointment ID
    existing = await sb_get("appointments", {"select": "appointment_id", "order": "appointment_id.desc", "limit": "1"})
    next_num = 1
    if existing:
        try:
            last = existing[0]["appointment_id"]
            next_num = int(last.replace("APT-H", "")) + 1
        except (ValueError, KeyError):
            next_num = 1000
    apt_id = f"APT-H{next_num:03d}"

    # 3. Mark slot Booked
    new_booked = slot.get("booked_count", 0) + 1
    new_status = "Booked" if new_booked >= slot.get("slot_capacity", 1) else "Open"
    await sb_update("doctor_availability",
                    {"slot_id": f"eq.{slot_id}"},
                    {"booked_count": new_booked, "status": new_status})

    # 4. Insert appointment
    apt_row = {
        "appointment_id": apt_id,
        "patient_id": patient_id,
        "doctor_id": slot["doctor_id"],
        "clinic_id": slot["clinic_id"],
        "date": slot["date"],
        "start_time": slot["start_time"],
        "duration_minutes": 30,
        "type": type,
        "reason_for_visit": reason or "",
        "status": "Scheduled",
        "created_date": date.today().isoformat(),
    }
    await sb_insert("appointments", apt_row)

    # 5. Log agent action
    doctor = await sb_get_one("doctors", {"doctor_id": f"eq.{slot['doctor_id']}"})
    clinic = await sb_get_one("clinics", {"clinic_id": f"eq.{slot['clinic_id']}"})
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
    })

    return {"ok": True, "appointment_id": apt_id, "appointment": apt_row}


# ============================================================
# WRITE: /appointment/cancel
# ============================================================
@app.post("/appointment/cancel")
async def cancel_appointment(
    appointment_id: str = Body(...),
    reason: Optional[str] = Body(None),
):
    apt = await sb_get_one("appointments", {"appointment_id": f"eq.{appointment_id}"})
    if not apt:
        raise HTTPException(status_code=404, detail="Appointment not found")
    if apt.get("status") in ("Cancelled", "Completed"):
        raise HTTPException(status_code=409, detail=f"Cannot cancel — status is {apt['status']}")

    # Mark cancelled
    await sb_update("appointments",
                    {"appointment_id": f"eq.{appointment_id}"},
                    {"status": "Cancelled",
                     "notes": f"{apt.get('notes') or ''}\nCancelled: {reason or 'No reason given'}"})

    # Try to release the slot (find by doctor+date+time)
    slot = await sb_get_one("doctor_availability", {
        "doctor_id": f"eq.{apt['doctor_id']}",
        "date": f"eq.{apt['date']}",
        "start_time": f"eq.{apt['start_time']}",
    })
    if slot:
        new_booked = max(0, slot.get("booked_count", 1) - 1)
        new_status = "Open" if new_booked < slot.get("slot_capacity", 1) else slot.get("status")
        await sb_update("doctor_availability",
                        {"slot_id": f"eq.{slot['slot_id']}"},
                        {"booked_count": new_booked, "status": new_status})

    await log_agent_action(apt["patient_id"], "Cancel Appointment",
                           f"Cancelled appointment {appointment_id} on {apt['date']} at {apt['start_time']}",
                           {"appointment_id": appointment_id, "reason": reason})

    return {"ok": True, "appointment_id": appointment_id, "status": "Cancelled"}


# ============================================================
# WRITE: /appointment/reschedule
# ============================================================
@app.post("/appointment/reschedule")
async def reschedule_appointment(
    appointment_id: str = Body(...),
    new_slot_id: str = Body(...),
    reason: Optional[str] = Body(None),
):
    # 1. cancel old
    apt = await sb_get_one("appointments", {"appointment_id": f"eq.{appointment_id}"})
    if not apt:
        raise HTTPException(status_code=404, detail="Appointment not found")
    patient_id = apt["patient_id"]
    apt_type = apt.get("type", "Follow-up")

    # 2. book new
    book_result = await book_appointment(
        patient_id=patient_id,
        slot_id=new_slot_id,
        reason=apt.get("reason_for_visit"),
        type=apt_type,
    )

    # 3. cancel old (only after new is booked successfully)
    await cancel_appointment(appointment_id=appointment_id, reason=f"Rescheduled to {book_result['appointment_id']}")

    return {"ok": True, "old_appointment_id": appointment_id, "new_appointment_id": book_result["appointment_id"]}


# ============================================================
# WRITE: /prescription/refill
# ============================================================
@app.post("/prescription/refill")
async def refill_prescription(
    prescription_id: str = Body(...),
    medication_id: str = Body(...),
    pharmacy_id: str = Body(...),
    delivery_method: str = Body("Pickup"),  # "Pickup" | "Home Delivery"
):
    rx = await sb_get_one("prescriptions", {"prescription_id": f"eq.{prescription_id}"})
    if not rx:
        raise HTTPException(status_code=404, detail="Prescription not found")
    if rx.get("status") != "Active":
        raise HTTPException(status_code=409, detail=f"Prescription is {rx.get('status')}")

    # Find the specific medication in the JSON array, decrement refills
    # Lovable normalized JSONB keys from "Medication ID" → "medication_id", etc.
    # We tolerate both formats for safety.
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
            # Decrement in whichever key the data uses
            if "refills_remaining" in m:
                m["refills_remaining"] = refills_remaining - 1
            else:
                m["Refills Remaining"] = refills_remaining - 1
            med_name = m.get("name_en") or m.get("Name (EN)")
            found = True
            break
    if not found:
        raise HTTPException(status_code=404, detail=f"Medication {medication_id} not in this prescription")

    await sb_update("prescriptions",
                    {"prescription_id": f"eq.{prescription_id}"},
                    {"medications": meds, "last_filled_date": date.today().isoformat(), "last_filled_pharmacy_id": pharmacy_id})

    # Create refill request
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
    await sb_insert("refill_requests", refill_row)

    pharmacy = await sb_get_one("pharmacies", {"pharmacy_id": f"eq.{pharmacy_id}"})
    pharmacy_name = pharmacy.get("pharmacy_name_en") if pharmacy else pharmacy_id

    await log_agent_action(rx["patient_id"], "Refill Request",
                           f"Refill requested for {med_name} → {pharmacy_name} ({delivery_method})",
                           {"prescription_id": prescription_id, "medication_id": medication_id,
                            "pharmacy_id": pharmacy_id, "delivery_method": delivery_method})

    return {"ok": True, "medication_name": med_name, "pharmacy": pharmacy_name, "delivery_method": delivery_method}


# ============================================================
# WRITE: /invoice/payment
# ============================================================
@app.post("/invoice/payment")
async def record_payment(
    invoice_id: str = Body(...),
    amount_sar: float = Body(...),
    payment_method: str = Body("Credit Card"),
):
    inv = await sb_get_one("invoices", {"invoice_id": f"eq.{invoice_id}"})
    if not inv:
        raise HTTPException(status_code=404, detail="Invoice not found")
    if inv.get("status") == "Paid":
        raise HTTPException(status_code=409, detail="Invoice already paid")

    due = float(inv.get("patient_due_sar") or 0)
    if amount_sar < due:
        new_status = "Partially Paid"
        # Track partial via notes
        note_extra = f"\nPartial payment {amount_sar} SAR on {date.today().isoformat()} ({payment_method})"
    else:
        new_status = "Paid"
        note_extra = f"\nPaid in full {amount_sar} SAR on {date.today().isoformat()} ({payment_method})"

    await sb_update("invoices",
                    {"invoice_id": f"eq.{invoice_id}"},
                    {"status": new_status,
                     "payment_method": payment_method,
                     "payment_date": date.today().isoformat(),
                     "notes_en": (inv.get("notes_en") or "") + note_extra})

    await log_agent_action(inv["patient_id"], "Payment Recorded",
                           f"Payment of SAR {amount_sar} via {payment_method} for invoice {invoice_id}",
                           {"invoice_id": invoice_id, "amount_sar": amount_sar, "method": payment_method})

    return {"ok": True, "invoice_id": invoice_id, "status": new_status, "amount_paid_sar": amount_sar}


# ============================================================
# WRITE: /profile/update
# ============================================================
ALLOWED_PROFILE_FIELDS = {"phone", "email", "address_en", "address_ar", "city_en", "city_ar"}


@app.post("/profile/update")
async def update_profile(
    patient_id: str = Body(...),
    field: str = Body(...),
    new_value: str = Body(...),
):
    if field not in ALLOWED_PROFILE_FIELDS:
        raise HTTPException(status_code=400, detail=f"Field {field} not updatable. Allowed: {ALLOWED_PROFILE_FIELDS}")

    patient = await sb_get_one("patients", {"patient_id": f"eq.{patient_id}"})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    old_value = patient.get(field)
    await sb_update("patients", {"patient_id": f"eq.{patient_id}"}, {field: new_value})

    await log_agent_action(patient_id, "Profile Updated",
                           f"{field}: {old_value} → {new_value}",
                           {"field": field, "old_value": old_value, "new_value": new_value})

    return {"ok": True, "patient_id": patient_id, "field": field, "new_value": new_value}


# ============================================================
# WRITE: /preauth/request
# ============================================================
@app.post("/preauth/request")
async def request_preauth(
    patient_id: str = Body(...),
    procedure_name: str = Body(...),
    doctor_id: Optional[str] = Body(None),
):
    patient = await sb_get_one("patients", {"patient_id": f"eq.{patient_id}"})
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
    result = await sb_insert("preauth_requests", row)

    await log_agent_action(patient_id, "Pre-Auth Requested",
                           f"Pre-authorization request submitted for {procedure_name}",
                           {"procedure": procedure_name, "doctor_id": doctor_id})

    return {"ok": True, "procedure": procedure_name, "status": "Submitted",
            "estimated_response_days": "1-5 business days"}


# ============================================================
# WRITE: /lab-result/release (portal-side, but agent can also trigger)
# ============================================================
@app.post("/lab-result/release")
async def release_lab_result(
    lab_result_id: str = Body(...),
    released_by: str = Body("Doctor"),
):
    lab = await sb_get_one("lab_results", {"lab_result_id": f"eq.{lab_result_id}"})
    if not lab:
        raise HTTPException(status_code=404, detail="Lab result not found")
    if lab.get("status") == "Released":
        raise HTTPException(status_code=409, detail="Already released")

    await sb_update("lab_results",
                    {"lab_result_id": f"eq.{lab_result_id}"},
                    {"status": "Released",
                     "result_date": date.today().isoformat(),
                     "released_at": datetime.utcnow().isoformat(),
                     "released_by": released_by})

    await log_agent_action(lab["patient_id"], "Lab Result Released",
                           f"Released {lab.get('test_name_en')} to patient",
                           {"lab_result_id": lab_result_id, "released_by": released_by})

    return {"ok": True, "lab_result_id": lab_result_id, "status": "Released"}


# ============================================================
# WRITE: /patient/register
# ============================================================

# Allowed enum values for the intake fields
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
    """
    Build a clean, human-readable staff-facing summary of the registration.
    The staff portal renders this prominently on Pending Verification patients
    so the staff member calling for verification has full context.
    """
    parts = [f"Patient self-registered via WhatsApp on {registration_date}."]

    # Reason + concern
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

    # Insurance
    provider_clean = (insurance_provider or "").strip()
    if insurance_status == "has_provider" and provider_clean:
        parts.append(f"Has {provider_clean} insurance.")
    elif insurance_status == "has_insurance_unknown_provider":
        parts.append("Has insurance but doesn't know plan details — needs verification.")
    elif insurance_status == "self_pay":
        parts.append("Self-pay (no insurance on file).")
    # If unknown or missing, no insurance line

    # Closer recommendation tailored to context
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
):
    """
    Register a new patient via the WhatsApp agent.
    Creates a patient row with status 'Pending Verification' and composes a
    staff-facing intake_notes summary from the registration context. A staff
    member follows up within 1 business day to verify and finalize.

    Required: national_id (10 digits, starts with 1=Saudi or 2=Iqama),
              email, phone, and at least one of full_name_en / full_name_ar.

    Optional (recommended for richer intake): registration_reason,
              registration_concern_note, registration_insurance_provider,
              registration_insurance_status.
    """

    # === Validation ===

    # National ID: exactly 10 digits, all numeric
    nid = (national_id or "").strip()
    if not nid.isdigit() or len(nid) != 10:
        raise HTTPException(
            status_code=400,
            detail="National ID must be exactly 10 digits"
        )

    # First digit determines id_type
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

    # At least one name
    name_en = (full_name_en or "").strip()
    name_ar = (full_name_ar or "").strip()
    if not name_en and not name_ar:
        raise HTTPException(
            status_code=400,
            detail="At least one of full_name_en or full_name_ar is required"
        )

    # Basic email format
    em = (email or "").strip()
    if "@" not in em or "." not in em.split("@")[-1]:
        raise HTTPException(
            status_code=400,
            detail="A valid email address is required"
        )

    # Phone
    ph = (phone or "").strip()
    if not ph:
        raise HTTPException(
            status_code=400,
            detail="Phone number is required"
        )

    # Optional enum validation
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

    # === Duplicate check by national_id ===

    existing = await sb_get_one("patients", {"national_id": f"eq.{nid}"})
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"A patient with this National ID is already registered (Patient ID: {existing.get('patient_id')})"
        )

    # === Generate next sequential patient_id ===

    all_patients = await sb_get("patients", {"select": "patient_id"})
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

    # === Preferred language: prefer AR if AR name was given, else EN ===
    preferred_language = "Arabic" if name_ar else "English"

    # === Compose intake_notes ===
    today_iso = date.today().isoformat()
    # Friendly date format for the notes (e.g. "June 2, 2026")
    try:
        date_display = date.today().strftime("%B %-d, %Y")
    except ValueError:
        # Windows fallback (just in case)
        date_display = date.today().strftime("%B %d, %Y").replace(" 0", " ")

    intake_notes = _compose_intake_notes(
        registration_date=date_display,
        reason=reason,
        concern_note=concern_note,
        insurance_status=insurance_status,
        insurance_provider=insurance_provider,
    )

    # === Insert ===

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

    result = await sb_insert("patients", row)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to register patient")

    # === Log to agent_actions ===

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
