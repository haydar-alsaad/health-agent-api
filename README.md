# Al-Noor Health Agent API (v2.0)

FastAPI service backing the Nebelus / T2 Communicate WhatsApp healthcare agent demo for **Al-Noor Medical Network**.

## Architecture

- **Source of truth:** Supabase (managed via Lovable Cloud)
- **API layer:** This FastAPI service running on Railway
- **Realtime:** Supabase Realtime — the staff portal (Lovable) subscribes for live updates when this API writes
- **Auto-warming:** Hit `/health` every 5 minutes via cron-job.org to keep the container warm

## Environment variables (set in Railway)

| Variable | Required | Default | Notes |
|---|---|---|---|
| `SUPABASE_URL` | Yes | — | e.g. `https://xyz.supabase.co` |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | — | from Lovable Cloud → Supabase → Settings → API |
| `SEED_ON_BOOT` | No | `true` | If `true`, seeds tables from `data/*.json` on startup IF the patients table is empty |
| `PORT` | (set by Railway) | 8000 | |

## Endpoints

### Read

| Method | Path | Notes |
|---|---|---|
| GET | `/` | Service info |
| GET | `/health` | Status + table row counts (used for warming) |
| GET | `/patient?patient_id=PAT-002` | Workhorse — full patient package (parallelized) |
| GET | `/doctor?doctor_id=DR-001` (or `?name=` `?specialty=` `?clinic_id=`) | Doctor lookup |
| GET | `/slots?doctor_id=DR-001&from_date=2026-05-15&limit=5` | Available slots |
| GET | `/clinic?clinic_id=CLN-001` (or `?city=Riyadh`) | Clinic lookup |
| GET | `/medication?medication_id=MED-001` (or `?name=Atorvastatin`) | Medication catalog |
| GET | `/insurance?provider_id=INS-001-VIP` | Insurance plan details |

### Write (each logs to `agent_actions` for the Live Activity Drawer)

| Method | Path | Body |
|---|---|---|
| POST | `/appointment/book` | `{patient_id, slot_id, reason?, type?}` |
| POST | `/appointment/cancel` | `{appointment_id, reason?}` |
| POST | `/appointment/reschedule` | `{appointment_id, new_slot_id, reason?}` |
| POST | `/prescription/refill` | `{prescription_id, medication_id, pharmacy_id, delivery_method}` |
| POST | `/invoice/payment` | `{invoice_id, amount_sar, payment_method}` |
| POST | `/profile/update` | `{patient_id, field, new_value}` (field: `phone`, `email`, `address_en/ar`, `city_en/ar`) |
| POST | `/preauth/request` | `{patient_id, procedure_name, doctor_id?}` |
| POST | `/lab-result/release` | `{lab_result_id, released_by}` |

## Local development

```bash
pip install -r requirements.txt
export SUPABASE_URL=https://your-project.supabase.co
export SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
uvicorn main:app --reload
```

Then visit http://localhost:8000/health to verify.

## Deployment to Railway

1. Push to GitHub
2. Connect Railway to the repo
3. In Railway → Variables, set `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`
4. Deploy
5. Set up cron-job.org to hit `https://your-railway-url/health` every 5 minutes

## Data

All seed data lives in `data/` as JSON files. On first boot (with empty Supabase tables), the API seeds them automatically.

| File | Records |
|---|---|
| patients.json | 12 |
| doctors.json | 18 |
| clinics.json | 3 |
| pharmacies.json | 5 |
| insurance_providers.json | 8 |
| medications_catalog.json | 15 |
| appointments.json | 22 |
| lab_results.json | 11 |
| prescriptions.json | 8 |
| invoices.json | 8 |
| medical_history.json | 21 |
| doctor_availability.json | 4127 slots |

**Anchor demo persona:** Omar Al-Salem (PAT-002) — 58yo post-angioplasty cardiac patient with Tawuniya VIP insurance, outstanding invoice INV-H003, released Lipid Panel from April 11.
