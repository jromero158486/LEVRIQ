"""Layer 7 contract for the Lovable frontend (`/api/v1`, repo jromero158486/screenshot-match).

Pure presentation mapping of the existing services (run_task, documents, facts, conflicts, review, verify_text) into
the frontend's view contract (`src/lib/clinical/types.ts` there). No clinical logic lives here: every fact, block,
conflict and missing item comes from Layers 1-6, and every evidence span is an exact slice of the stored document text.
The original `/api` endpoints are unchanged.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.app.api.schemas import Citation, Claim, VerifyTextRequest, VerifyTextResponse
from backend.app.api.service import NOTICES, UnknownPatient, patient_conflicts, run_task, verify_text
from backend.app.layer3_store.repository import ClinicalStore
from backend.app.layer4_planner.vocabulary import TITLE_DE
from backend.app.layer5_retrieval.engine import Synthesizer, build_timeline
from backend.app.layer6_verification.verify import ClaimVerifier

VERIF = {"supported": "documented", "contradicted": "conflicting", "not_addressed": "unable_to_verify",
         "verification_incomplete": "needs_review"}
APPROVAL = {"accepted": "approved", "rejected": "rejected"}
REVIEW_STATUS = {"accepted": "approved", "rejected": "rejected", "needs_more_info": "escalated"}
DECISION = {"approve": "accepted", "reject": "rejected", "correct": "needs_more_info", "escalate": "needs_more_info"}
SEX = {"weiblich": "female", "w": "female", "female": "female", "männlich": "male", "m": "male", "male": "male"}


class TaskIn(BaseModel):
    patientId: str
    task: str = Field(min_length=3, max_length=2000)
    role: str | None = Field(None, max_length=300)
    locale: str | None = Field(None, description="Response language: en or de; default = language of the request")
    referenceDate: str | None = None


class DecisionIn(BaseModel):
    reviewItemId: str
    factId: str
    decision: str
    comment: str | None = None
    correctedStatement: str | None = None


class VerifyIn(BaseModel):
    patientId: str
    statement: str = Field(min_length=3, max_length=5000)
    judge: str = "nli"



_UI_DE = {"Summary": "Zusammenfassung", "Not documented": "Nicht dokumentiert", "Request not answered": "Anfrage nicht beantwortet",
          "Potential conflicts in the record": "Mögliche Widersprüche in der Dokumentation",
          "Needs clinician review": "Ärztliche Prüfung erforderlich", "Timeline": "Zeitverlauf"}


_TITLE_EN = {v: k for k, v in TITLE_DE.items()}


def _title_de(t: str) -> str:
    m = re.fullmatch(r"(\w+) allergy", t)
    return f"{m.group(1)}-Allergie" if m else TITLE_DE.get(t, t)


def _title_en(t: str) -> str:
    m = re.fullmatch(r"(\w+)-Allergie", t)
    return f"{m.group(1)} allergy" if m else _TITLE_EN.get(t, t)


def _missing_de(msg: str) -> str:
    m = re.match(r"Not documented in the (\d+) available documents \(as of ([\d-]+)\)\.?$", msg)
    return f"In den {m.group(1)} verfügbaren Dokumenten nicht dokumentiert (Stand {m.group(2)})." if m else msg


def _verification(c: Claim) -> str:
    if c.kind == "documented_negative" and c.status == "supported":
        return "documented_negative"
    if c.kind == "not_documented":
        return "not_documented"
    return VERIF.get(c.status, "needs_review")


class _Docs:
    """Per-request cache: document sections (verbatim slices of normalized_text) and passage → section mapping."""

    def __init__(self, store: ClinicalStore, pid: str):
        self.store, self.pid, self.cache = store, pid, {}

    def get(self, document_id: str):
        if document_id not in self.cache:
            found = self.store.document(self.pid, document_id)
            if not found:
                self.cache[document_id] = None
            else:
                d, ps = found
                text = d.normalized_text or ""
                ps = sorted(ps, key=lambda p: p.normalized_start)
                groups: list[list] = []
                for p in ps:
                    head = p.section_title or p.section or "Text"
                    if groups and groups[-1][0] == head:
                        groups[-1][1].append(p)
                    else:
                        groups.append([head, [p]])
                sections, pmap = [], {}
                for i, (head, members) in enumerate(groups):
                    start = members[0].normalized_start
                    end = groups[i + 1][1][0].normalized_start if i + 1 < len(groups) else len(text)
                    sid = f"{document_id}:s{i}"
                    sections.append({"id": sid, "heading": head.rstrip(":"), "text": text[start:end]})
                    for p in members:
                        pmap[p.passage_id] = (sid, start, p)
                self.cache[document_id] = (d, text, sections, pmap)
        return self.cache[document_id]

    def evidence(self, cit: Citation, verification: str, extractor: str) -> dict | None:
        got = self.get(cit.document_id)
        if not got or cit.passage_id not in got[3]:
            return None
        d, text, _, pmap = got
        sid, s0, p = pmap[cit.passage_id]
        ps, pe = p.normalized_start, p.normalized_end
        i = text.find(cit.quote, ps, pe) if cit.quote else -1
        start, end = (i, i + len(cit.quote)) if i >= 0 else (ps, pe)
        return {"sourceId": cit.source_id, "documentId": cit.document_id, "sectionId": sid, "start": start - s0,
                "end": end - s0, "quote": text[start:end],
                "provenance": {"extractor": extractor, "layer": "L6 citation integrity " + ("verified" if cit.verified else "FAILED"),
                               "extractedAt": d.document_date or ""},
                "verification": verification}


def _fact_from_claim(c: Claim, docs: _Docs, category: str) -> dict:
    v = _verification(c)
    extractor = "L5 passage retrieval" if c.kind == "passage" else "L2 extraction"
    ev = [e for e in (docs.evidence(ct, v, extractor) for ct in c.citations) if e]
    notes = [n for n in ([c.original_text and c.original_text != c.text and f"Original: {c.original_text}"]
                         + c.flags + c.notes) if n]
    return {"id": c.claim_id, "statement": c.text, "category": category, "eventDate": c.date or None,
            "documentDate": c.citations[0].document_date if c.citations else None, "verification": v,
            "approval": APPROVAL.get(c.review_status or "", "unreviewed"), "evidence": ev,
            "note": "; ".join(notes) or None}


def _clean(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def _fact_from_facts(f, syn: Synthesizer, docs: _Docs, verifier) -> dict:
    v = "documented_negative" if f.assertion.negated else ("needs_review" if f.review_status == "needs_review" else "documented")
    cits = syn.citations([f])
    for c in cits:
        c.verified = verifier.verify_citation(c)
    ev = [e for e in (docs.evidence(c, v, f.extractor.name) for c in cits) if e]
    return _clean({"id": f.fact_id, "statement": f.label, "category": f.fact_type,
                   "eventDate": f.event_date.value if f.event_date else None, "documentDate": f.document_date,
                   "verification": v, "approval": "unreviewed", "evidence": ev})


def register_frontend_api(app: FastAPI, store: ClinicalStore, llm=None) -> None:
    def known(pid: str):
        if not store.documents(pid):
            raise HTTPException(404, "Patient not found or not authorized")

    def patient(p: dict) -> dict:
        demo = {f.details.attribute: f.details.value for f in store.facts(p["patient_id"], types=["demographic"])}
        reviews = store.review_items(p["patient_id"])
        return _clean({"id": p["patient_id"], "displayName": p["display_name"] or p["patient_id"], "mrn": p["patient_id"],
                       "birthDate": demo.get("birth_date", ""), "sex": SEX.get(str(demo.get("sex", "")).lower(), "unknown"),
                       "referenceDate": p["last_document_date"] or "", "documentCount": p["document_count"],
                       "openReviewCount": sum(r["status"] in ("pending", "needs_more_info") for r in reviews)
                       + len(patient_conflicts(store, p["patient_id"]))})

    @app.get("/api/v1/patients")
    def v1_patients():
        return [patient(p) for p in store.patients()]

    @app.get("/api/v1/patients/{pid}")
    def v1_patient(pid: str):
        for p in store.patients():
            if p["patient_id"] == pid:
                return patient(p)
        raise HTTPException(404, "Patient not found or not authorized")

    @app.get("/api/v1/patients/{pid}/documents")
    def v1_documents(pid: str):
        known(pid)
        docs = _Docs(store, pid)
        out = []
        for d in store.documents(pid):
            _, _, sections, _ = docs.get(d.document_id)
            out.append(_clean({"id": d.document_id, "patientId": pid, "title": d.title, "type": d.document_type or "Dokument",
                               "documentDate": d.document_date or "", "institution": d.hospital, "language": "de",
                               "sections": sections}))
        return out

    @app.get("/api/v1/patients/{pid}/longitudinal")
    def v1_longitudinal(pid: str):
        known(pid)
        docs, syn = _Docs(store, pid), Synthesizer(store, pid)
        as_of = max(d.document_date for d in store.documents(pid) if d.document_date)
        verifier = ClaimVerifier(store, pid, as_of, [])
        facts: dict[str, dict] = {}
        timeline = []
        for i, e in enumerate(build_timeline(store, pid, as_of, syn)):
            timeline.append({"id": f"t{i}", "date": e.date, "dateKind": "document" if e.kind == "document" else "event",
                             "label": e.label, "category": e.kind})
        dx = []
        for f in store.facts(pid, types=["diagnosis", "procedure"]):
            if f.assertion.question:
                continue
            facts[f.fact_id] = _fact_from_facts(f, syn, docs, verifier)
            dx.append(_clean({"id": f.fact_id, "label": f.label + (" (verneint)" if f.assertion.negated else ""),
                              "code": f.codings[0].code if f.codings else None, "kind": f.fact_type,
                              "date": (f.event_date.value if f.event_date else f.document_date) or "", "factId": f.fact_id,
                              "verification": facts[f.fact_id]["verification"]}))
        meds = [f for f in store.facts(pid, types=["medication"]) if f.details.statement == "medication_list"]
        episodes, rows = {}, {}
        for f in meds:
            d = next((x for x in store.documents(pid) if x.document_id == f.document_id), None)
            episodes.setdefault(f.document_id, {"id": f.document_id, "label": d.title if d else f.document_id,
                                                "date": f.document_date or ""})
            facts[f.fact_id] = _fact_from_facts(f, syn, docs, verifier)
            m = f.details
            sub = (m.ingredient or m.drug_text).strip()
            dose = " ".join(x for x in [f"{m.strength_value:g} {m.strength_unit}" if m.strength_value is not None else "",
                                        (m.schedule.text if m.schedule and m.schedule.text else "")] if x) or m.drug_text
            rows.setdefault(sub.casefold(), {"substance": sub, "cells": {}})["cells"][f.document_id] = {
                "dose": dose, "factId": f.fact_id, "status": "documented"}
        medications = {"id": "meds", "kind": "medication_comparison", "title": "Medication lists", "priority": "normal",
                       "episodes": sorted(episodes.values(), key=lambda e: e["date"]),
                       "rows": sorted(rows.values(), key=lambda r: r["substance"].casefold())}
        return {"patientId": pid, "timeline": timeline, "medications": medications, "diagnoses": dx, "facts": facts}

    @app.post("/api/v1/clinical-tasks")
    def v1_task(req: TaskIn):
        text = f"{req.role.strip()}: {req.task.strip()}" if req.role and req.role.strip() else req.task
        try:
            r = run_task(store, req.patientId, text, llm)
        except UnknownPatient:
            raise HTTPException(404, "Patient not found or not authorized")
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        docs = _Docs(store, req.patientId)
        facts: dict[str, dict] = {}
        blocks: list[dict] = []
        summary = []
        lang = (req.locale or "")[:2].lower()
        de = (lang if lang in ("en", "de") else r.plan.presentation.language) == "de"
        ui = (lambda t: _UI_DE.get(t, t)) if de else (lambda t: t)
        loc = _title_de if de else _title_en
        for n, s in enumerate(r.sections):
            for c in s.claims:
                facts[c.claim_id] = _fact_from_claim(c, docs, s.concept)
            if s.claims and len(summary) < 5 and s.outcome != "not_documented":
                latest = s.claims[0] if s.claims[0].kind == "passage" else \
                    max(s.claims, key=lambda c: (c.date or (c.citations[0].document_date if c.citations else "") or ""))
                summary.append({"text": f"{loc(s.title)}: {latest.text}", "factIds": [latest.claim_id]})  # best passage / most recent
            if not s.claims:  # reported in the missing-information block instead
                continue
            blocks.append({"id": s.section_id, "kind": "fact_list", "title": loc(s.title),
                           "priority": "high" if n < 3 else "normal", "taskRelevance": "relevant",
                           "factIds": [c.claim_id for c in s.claims]})
        notes = [{"text": NOTICES[n]["de" if de else "en"], "factIds": []} for n in r.notices if n in NOTICES]
        summary = notes + summary
        if summary:
            blocks.insert(0, {"id": "summary", "kind": "summary",
                              "title": ui("Request not answered" if r.scope == "out_of_scope" else "Summary"), "priority": "high",
                              "taskRelevance": "relevant", "sentences": summary})
        if r.missing:
            cur = store.documents(req.patientId)
            blocks.append({"id": "missing", "kind": "missing_information", "title": ui("Not documented"), "priority": "high",
                           "taskRelevance": "relevant", "items": [
                               {"id": m.step_id, "label": loc(m.title), "detail": _missing_de(m.message) if de else m.message,
                                "searchedDocumentTypes": sorted({d.document_type or "Dokument" for d in cur}),
                                "searchedDocuments": [{"id": d.document_id, "title": d.title} for d in cur
                                                      if (d.document_date or "") <= r.as_of],
                                "searchedTimeRange": {"from": min(d.document_date for d in cur if d.document_date), "to": r.as_of}}
                               for m in r.missing]})
        if r.conflicts:
            items = []
            for k in sorted(r.conflicts, key=lambda k: not k.relevant_to_task):
                fids = [cid for cid, c in ((c.claim_id, c) for s in r.sections for c in s.claims) if k.conflict_id in c.conflict_ids]
                for i, ct in enumerate(k.citations):
                    fid = f"{k.conflict_id}:ev{i}"
                    ev = docs.evidence(ct, "conflicting", "L6 conflict detection")
                    facts[fid] = {"id": fid, "statement": ct.quote, "category": "conflict_evidence",
                                  "documentDate": ct.document_date, "verification": "conflicting",
                                  "approval": APPROVAL.get(k.review_status or "", "unreviewed"),
                                  "evidence": [ev] if ev else []}
                    fids.append(fid)
                items.append({"id": k.conflict_id, "topic": k.title, "factIds": fids, "explanation": k.explanation})
            blocks.append({"id": "conflicts", "kind": "conflicts", "title": ui("Potential conflicts in the record"),
                           "priority": "high" if any(k.relevant_to_task for k in r.conflicts) else "normal",
                           **({"taskRelevance": "relevant"} if any(k.relevant_to_task for k in r.conflicts) else {}),
                           "conflicts": items})
        review = [fid for fid, f in facts.items() if f["verification"] in ("needs_review", "conflicting", "unable_to_verify")
                  and not fid.count(":ev")]
        if review:
            blocks.append({"id": "review", "kind": "review_needed", "title": ui("Needs clinician review"), "priority": "normal",
                           "factIds": review})
        if r.timeline:
            blocks.append({"id": "timeline", "kind": "timeline", "title": ui("Timeline"), "priority": "low",
                           "taskRelevance": "supporting",
                           "events": [{"id": f"t{i}", "date": e.date, "dateKind": "document" if e.kind == "document" else "event",
                                       "label": e.label, "category": e.kind} for i, e in enumerate(r.timeline)]})
        p = r.provenance
        intent = (("out of scope: no patient data shown" if r.scope == "out_of_scope" else
                   '; '.join(loc(s.title) for s in r.plan.steps)) + f" — planner: {p.planner}"
                  f"{' (' + p.planner_model + ')' if p.planner_model else ''}; retrieval: {', '.join(p.retrieval_used) or 'structured only'}"
                  f"{'; ' + '; '.join(p.retrieval_warnings) if p.retrieval_warnings else ''}; as of {r.as_of}; "
                  f"{r.timings_ms.get('total', 0) / 1000:.1f} s")
        return _clean({"id": r.request_id, "patientId": r.patient_id, "task": req.task, "role": req.role,
                       "createdAt": datetime.now(timezone.utc).isoformat(), "origin": "live", "interpretedIntent": intent,
                       "blocks": blocks, "facts": facts})

    @app.get("/api/v1/patients/{pid}/review-items")
    def v1_review_items(pid: str):
        known(pid)
        docs, syn = _Docs(store, pid), Synthesizer(store, pid)
        as_of = max(d.document_date for d in store.documents(pid) if d.document_date)
        verifier = ClaimVerifier(store, pid, as_of, [])
        by_id = {f.fact_id: f for f in store.facts(pid)}
        status = {r["item_key"]: r for r in store.review_items(pid)}
        out = []
        for k in patient_conflicts(store, pid):
            cits = syn.citations([by_id[i] for i in k.fact_ids if i in by_id])
            for c in cits:
                c.verified = verifier.verify_citation(c)
            ev = [e for e in (docs.evidence(c, "conflicting", "L6 conflict detection") for c in cits) if e]
            rec = status.get(k.conflict_id, {})
            st = rec.get("status", "pending")
            out.append({"id": f"{pid}|{k.conflict_id}", "patientId": pid, "reason": k.explanation,
                        "createdAt": status.get(k.conflict_id, {}).get("updated_at") or datetime.now(timezone.utc).isoformat(),
                        "fact": {"id": k.conflict_id, "statement": k.title, "category": k.kind, "verification": "conflicting",
                                 "approval": APPROVAL.get(st, "unreviewed"), "evidence": ev},
                        # reviewed vs pending for the queue; escalations keep approval "unreviewed" (not approved)
                        "reviewStatus": REVIEW_STATUS.get(st, "pending"),
                        "reviewComment": next((h["comment"] for h in reversed(rec.get("history", [])) if h.get("comment")), None)})
        return out

    @app.post("/api/v1/review-items/{item_id}/decision")
    def v1_decision(item_id: str, sub: DecisionIn):
        pid, _, key = item_id.partition("|")
        if not key:
            raise HTTPException(422, "Unknown review item")
        known(pid)
        if sub.decision not in DECISION:
            raise HTTPException(422, f"decision must be one of {sorted(DECISION)}")
        comment = "; ".join(x for x in [sub.comment, sub.correctedStatement and f"Correction: {sub.correctedStatement}",
                                        sub.decision == "escalate" and "escalated"] if x) or None
        store.upsert_review_item(pid, key, "conflict", "", {})
        store.set_review_status(pid, key, DECISION[sub.decision], "demo-clinician (no authentication)", comment)
        return {"accepted": True}

    @app.post("/api/v1/verify-statement", response_model=VerifyTextResponse)
    def v1_verify(req: VerifyIn):
        try:
            return verify_text(store, VerifyTextRequest(patient_id=req.patientId, text=req.statement,
                                                        judge=req.judge if req.judge in ("nli", "none", "groq") else "nli"), llm)
        except UnknownPatient:
            raise HTTPException(404, "Patient not found or not authorized")
