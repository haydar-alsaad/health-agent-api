# Health Agent API

FastAPI service backing the Nebelus / T2 Communicate WhatsApp healthcare agent demo.

## Endpoints

- `GET /health` — health check + data load counts
- `GET /patient?patient_id=PAT-002` — full patient package (the workhorse)
- `GET /doctor?doctor_id=DR-001` or `?specialty=Cardiology` — doctor lookup
- `GET /slots?doctor_id=DR-001&from_date=2026-05-15` — appointment slots
- `GET /clinic?city=Riyadh` — clinic lookup
- `GET /medication?name=Atorvastatin` — medication catalog
- `GET /insurance?provider_id=INS-001-VIP` — insurance plan details

## Local development

\`\`\`bash
pip install -r requirements.txt
uvicorn main:app --reload
\`\`\`

Then visit http://localhost:8000/health to verify data loaded.

## Deployment

Designed for Railway. The \`Procfile\` runs uvicorn against \`main:app\`.

## Data

All demo data lives in \`data/\` as JSON files. Bilingual (English + Arabic) where applicable. Anchor demo persona is **Omar Al-Salem (PAT-002)** — post-angioplasty cardiac patient with Tawuniya VIP insurance.

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
