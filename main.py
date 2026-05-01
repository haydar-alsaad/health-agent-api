"""
Healthcare Agent API - v1.0
6 endpoints + /health:
  GET /health        - status + data load counts
  GET /patient       - workhorse: full patient package
  GET /doctor        - doctor info
  GET /slots         - available appointment slots
  GET /clinic        - clinic info
  GET /medication    - medication catalog lookup
  GET /insurance     - insurance plan lookup
"""
import json
import os
from datetime import date
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Health Agent API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

patients = []
doctors = []
clinics = []
pharmacies = []
insurance_providers = []
medications = []
appointments = []
lab_results = []
prescriptions = []
invoices = []
medical_history = []
doctor_availability = []


def load(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def load_all_data():
    global patients, doctors, clinics, pharmacies, insurance_providers
    global medications, appointments, lab_results, prescriptions, invoices
    global medical_history, doctor_availability

    patients = load("patients.json")
    doctors = load("doctors.json")
    clinics = load("clinics.json")
    pharmacies = load("pharmacies.json")
    insurance_providers = load("insurance_providers.json")
    medications = load("medications_catalog.json")
    appointments = load("appointments.json")
    lab_results = load("lab_results.json")
    prescriptions = load("prescriptions.json")
    invoices = load("invoices.json")
    medical_history = load("medical_history.json")
    doctor_availability = load("doctor_availability.json")


@app.on_event("startup")
def startup():
    load_all_data()


def find_patient(pid):
    return next((p for p in patients if p["Patient ID"] == pid), None)

def find_doctor(did):
    return next((d for d in doctors if d["Doctor ID"] == did), None)

def find_clinic(cid):
    return next((c for c in clinics if c["Clinic ID"] == cid), None)

def find_insurance(iid):
    return next((i for i in insurance_providers if i["Provider ID"] == iid), None)

def find_medication(mid):
    return next((m for m in medications if m["Medication ID"] == mid), None)

def find_pharmacy(pid):
    return next((p for p in pharmacies if p["Pharmacy ID"] == pid), None)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": "1.0",
        "data_loaded": {
            "patients": len(patients),
            "doctors": len(doctors),
            "clinics": len(clinics),
            "pharmacies": len(pharmacies),
            "insurance_providers": len(insurance_providers),
            "medications": len(medications),
            "appointments": len(appointments),
            "lab_results": len(lab_results),
            "prescriptions": len(prescriptions),
            "invoices": len(invoices),
            "medical_history": len(medical_history),
            "doctor_availability_slots": len(doctor_availability),
        }
    }


@app.get("/patient")
def get_patient_data(
    patient_id: str = Query(..., description="Patient ID e.g. PAT-002"),
    include_history: bool = Query(False),
    include_past_appointments: bool = Query(True)
):
    """Workhorse endpoint - everything for one patient in a single call."""
    patient = find_patient(patient_id)
    if not patient:
        raise HTTPException(404, f"Patient {patient_id} not found")

    # Insurance resolved
    insurance = find_insurance(patient["Insurance Provider ID"])
    insurance_resolved = None
    if insurance:
        insurance_resolved = {
            "provider_id": insurance["Provider ID"],
            "provider_name_en": insurance["Provider Name (EN)"],
            "provider_name_ar": insurance["Provider Name (AR)"],
            "plan_tier_en": insurance["Plan Tier (EN)"],
            "plan_tier_ar": insurance["Plan Tier (AR)"],
            "policy_number": patient.get("Insurance Policy Number"),
            "annual_limit_sar": insurance["Annual Limit SAR"],
            "coverage": {
                "gp_consultation": insurance["GP Consultation Coverage"],
                "specialist_consultation": insurance["Specialist Consultation Coverage"],
                "lab": insurance["Lab Coverage"],
                "imaging": insurance["Imaging Coverage"],
                "medication": insurance["Medication Coverage"],
                "er": insurance["ER Coverage"],
            },
            "pre_authorization_required_for": insurance["Pre-Authorization Required For"],
            "co_pay_notes": insurance["Co-pay Notes"],
            "network_clinic_ids": insurance["Network Hospitals"],
        }

    primary_doctor = find_doctor(patient.get("Primary Care Doctor ID")) if patient.get("Primary Care Doctor ID") else None

    # Appointments
    pat_appts = [dict(a) for a in appointments if a["Patient ID"] == patient_id]
    upcoming_appointments = sorted(
        [a for a in pat_appts if a["Status"] == "Scheduled"],
        key=lambda x: (x["Date"], x["Start Time"])
    )
    for a in upcoming_appointments:
        doc = find_doctor(a["Doctor ID"])
        clinic = find_clinic(a["Clinic ID"])
        if doc:
            a["doctor_name_en"] = doc["Full Name (EN)"]
            a["doctor_name_ar"] = doc["Full Name (AR)"]
            a["doctor_specialty_en"] = doc["Specialty (EN)"]
            a["doctor_specialty_ar"] = doc["Specialty (AR)"]
        if clinic:
            a["clinic_name_en"] = clinic["Clinic Name (EN)"]
            a["clinic_name_ar"] = clinic["Clinic Name (AR)"]
            a["clinic_address_en"] = clinic["Address (EN)"]
            a["clinic_address_ar"] = clinic["Address (AR)"]

    recent_appointments = []
    if include_past_appointments:
        recent_appointments = sorted(
            [a for a in pat_appts if a["Status"] == "Completed"],
            key=lambda x: x["Date"], reverse=True
        )[:5]
        for a in recent_appointments:
            doc = find_doctor(a["Doctor ID"])
            if doc:
                a["doctor_name_en"] = doc["Full Name (EN)"]
                a["doctor_name_ar"] = doc["Full Name (AR)"]

    # Active prescriptions with med catalog merged
    pat_rx = [r for r in prescriptions if r["Patient ID"] == patient_id]
    active_prescriptions = []
    for rx in pat_rx:
        if rx["Status"] != "Active":
            continue
        rx_enriched = dict(rx)
        rx_enriched["Medications"] = []
        for med in rx["Medications"]:
            cat = find_medication(med["Medication ID"])
            merged = dict(med)
            if cat:
                merged["indication_en"] = cat["Indication (EN)"]
                merged["indication_ar"] = cat["Indication (AR)"]
                merged["class_en"] = cat["Class (EN)"]
                merged["class_ar"] = cat["Class (AR)"]
                merged["common_side_effects_en"] = cat["Common Side Effects (EN)"]
                merged["common_side_effects_ar"] = cat["Common Side Effects (AR)"]
                merged["interactions_note_en"] = cat["Interactions Note (EN)"]
                merged["interactions_note_ar"] = cat["Interactions Note (AR)"]
                merged["coverage_by_tier"] = cat["Coverage by Tier"]
            rx_enriched["Medications"].append(merged)
        doc = find_doctor(rx["Prescribing Doctor ID"])
        if doc:
            rx_enriched["prescribing_doctor_name_en"] = doc["Full Name (EN)"]
            rx_enriched["prescribing_doctor_name_ar"] = doc["Full Name (AR)"]
        active_prescriptions.append(rx_enriched)

    # Lab results
    pat_labs = [dict(l) for l in lab_results if l["Patient ID"] == patient_id]
    released_lab_results = sorted(
        [l for l in pat_labs if l["Status"] == "Released"],
        key=lambda x: x.get("Result Date") or "", reverse=True
    )
    pending_lab_results = [l for l in pat_labs if l["Status"] == "Pending"]
    for lab in released_lab_results + pending_lab_results:
        doc = find_doctor(lab["Ordering Doctor ID"])
        if doc:
            lab["ordering_doctor_name_en"] = doc["Full Name (EN)"]
            lab["ordering_doctor_name_ar"] = doc["Full Name (AR)"]

    # Invoices
    pat_invoices = [i for i in invoices if i["Patient ID"] == patient_id]
    outstanding_invoices = sorted(
        [i for i in pat_invoices if i["Status"] == "Outstanding"],
        key=lambda x: x["Due Date"]
    )
    paid_invoices = [i for i in pat_invoices if i["Status"] == "Paid"]

    total_paid = sum(i["Patient Due SAR"] for i in paid_invoices)
    total_outstanding = sum(i["Patient Due SAR"] for i in outstanding_invoices)
    last_payment = max(paid_invoices, key=lambda x: x.get("Payment Date") or "") if paid_invoices else None
    payment_history = {
        "total_paid_sar": total_paid,
        "total_outstanding_sar": total_outstanding,
        "outstanding_invoice_count": len(outstanding_invoices),
        "last_payment_date": last_payment["Payment Date"] if last_payment else None,
        "last_payment_method": last_payment["Payment Method"] if last_payment else None,
    }

    history = []
    if include_history:
        history = sorted(
            [h for h in medical_history if h["Patient ID"] == patient_id],
            key=lambda x: x["Event Date"], reverse=True
        )

    return {
        "profile": {
            "patient_id": patient["Patient ID"],
            "name_en": patient["Full Name (EN)"],
            "name_ar": patient["Full Name (AR)"],
            "first_name_en": patient["Full Name (EN)"].split()[0],
            "date_of_birth": patient["Date of Birth"],
            "age": patient["Age"],
            "gender": patient["Gender"],
            "phone": patient["Phone"],
            "email": patient["Email"],
            "preferred_language": patient["Preferred Language"],
            "city_en": patient["City (EN)"],
            "city_ar": patient["City (AR)"],
            "address_en": patient["Address (EN)"],
            "address_ar": patient["Address (AR)"],
            "patient_status": patient["Patient Status"],
            "registered_since": patient["Registered Since"],
            "parent_guardian": patient.get("Parent/Guardian"),
            "emergency_contact": {
                "name": patient["Emergency Contact Name"],
                "phone": patient["Emergency Contact Phone"],
            },
        },
        "active_conditions_en": patient["Active Conditions (EN)"],
        "active_conditions_ar": patient["Active Conditions (AR)"],
        "allergies": patient["Allergies"],
        "allergies_alert": len(patient["Allergies"]) > 0,
        "insurance": insurance_resolved,
        "primary_care_doctor": primary_doctor,
        "upcoming_appointments": upcoming_appointments,
        "recent_appointments": recent_appointments,
        "active_prescriptions": active_prescriptions,
        "released_lab_results": released_lab_results,
        "pending_lab_results": pending_lab_results,
        "outstanding_invoices": outstanding_invoices,
        "payment_history_summary": payment_history,
        "medical_history": history,
    }


@app.get("/doctor")
def get_doctor_info(
    doctor_id: Optional[str] = None,
    specialty: Optional[str] = None,
    name: Optional[str] = None,
    clinic_id: Optional[str] = None,
):
    results = [dict(d) for d in doctors]
    if doctor_id:
        results = [d for d in results if d["Doctor ID"] == doctor_id]
    if specialty:
        s = specialty.lower()
        results = [d for d in results
                   if s in d["Specialty (EN)"].lower()
                   or s in (d.get("Sub-specialty (EN)") or "").lower()]
    if name:
        n = name.lower()
        results = [d for d in results
                   if n in d["Full Name (EN)"].lower() or n in d["Full Name (AR)"]]
    if clinic_id:
        results = [d for d in results
                   if d["Primary Clinic ID"] == clinic_id or clinic_id in d["Visiting Clinic IDs"]]

    for d in results:
        primary_clinic = find_clinic(d["Primary Clinic ID"])
        if primary_clinic:
            d["primary_clinic_name_en"] = primary_clinic["Clinic Name (EN)"]
            d["primary_clinic_name_ar"] = primary_clinic["Clinic Name (AR)"]

    return {"doctors": results, "count": len(results)}


@app.get("/slots")
def get_appointment_slots(
    doctor_id: Optional[str] = None,
    specialty: Optional[str] = None,
    clinic_id: Optional[str] = None,
    city: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    only_open: bool = True,
    limit: int = 20,
):
    results = [dict(s) for s in doctor_availability]

    if only_open:
        results = [s for s in results if s["Status"] == "Open"]
    if doctor_id:
        results = [s for s in results if s["Doctor ID"] == doctor_id]
    if specialty:
        sp = specialty.lower()
        spec_doc_ids = {d["Doctor ID"] for d in doctors
                        if sp in d["Specialty (EN)"].lower()
                        or sp in (d.get("Sub-specialty (EN)") or "").lower()}
        results = [s for s in results if s["Doctor ID"] in spec_doc_ids]
    if clinic_id:
        results = [s for s in results if s["Clinic ID"] == clinic_id]
    if city:
        city_clinic_ids = {c["Clinic ID"] for c in clinics if c["City (EN)"].lower() == city.lower()}
        results = [s for s in results if s["Clinic ID"] in city_clinic_ids]
    if from_date:
        results = [s for s in results if s["Date"] >= from_date]
    if to_date:
        results = [s for s in results if s["Date"] <= to_date]

    results.sort(key=lambda s: (s["Date"], s["Start Time"]))
    results = results[:limit]

    for s in results:
        doc = find_doctor(s["Doctor ID"])
        clinic = find_clinic(s["Clinic ID"])
        if doc:
            s["doctor_name_en"] = doc["Full Name (EN)"]
            s["doctor_name_ar"] = doc["Full Name (AR)"]
            s["doctor_specialty_en"] = doc["Specialty (EN)"]
            s["doctor_specialty_ar"] = doc["Specialty (AR)"]
            s["consultation_fee_sar"] = doc["Consultation Fee SAR"]
        if clinic:
            s["clinic_name_en"] = clinic["Clinic Name (EN)"]
            s["clinic_name_ar"] = clinic["Clinic Name (AR)"]
            s["clinic_city"] = clinic["City (EN)"]

    return {"slots": results, "count": len(results)}


@app.get("/clinic")
def get_clinic_info(
    clinic_id: Optional[str] = None,
    city: Optional[str] = None,
    specialty: Optional[str] = None,
):
    results = [dict(c) for c in clinics]
    if clinic_id:
        results = [c for c in results if c["Clinic ID"] == clinic_id]
    if city:
        results = [c for c in results if c["City (EN)"].lower() == city.lower()]
    if specialty:
        sp = specialty.lower()
        results = [c for c in results
                   if any(sp in s.lower() for s in c["Specialties Available"])]
    return {"clinics": results, "count": len(results)}


@app.get("/medication")
def get_medication_info(
    medication_id: Optional[str] = None,
    name: Optional[str] = None,
):
    results = [dict(m) for m in medications]
    if medication_id:
        results = [m for m in results if m["Medication ID"] == medication_id]
    if name:
        n = name.lower()
        results = [m for m in results
                   if n in m["Generic Name (EN)"].lower()
                   or n in m["Generic Name (AR)"]
                   or any(n in b.lower() for b in m["Brand Names"])]
    return {"medications": results, "count": len(results)}


@app.get("/insurance")
def get_insurance_info(
    provider_id: Optional[str] = None,
    provider_name: Optional[str] = None,
    plan_tier: Optional[str] = None,
):
    results = [dict(i) for i in insurance_providers]
    if provider_id:
        results = [i for i in results if i["Provider ID"] == provider_id]
    if provider_name:
        results = [i for i in results if provider_name.lower() in i["Provider Name (EN)"].lower()]
    if plan_tier:
        results = [i for i in results if plan_tier.lower() in i["Plan Tier (EN)"].lower()]
    return {"insurance_plans": results, "count": len(results)}


@app.get("/")
def root():
    return {
        "service": "Health Agent API",
        "version": "1.0",
        "endpoints": ["/health", "/patient", "/doctor", "/slots", "/clinic", "/medication", "/insurance"]
    }
