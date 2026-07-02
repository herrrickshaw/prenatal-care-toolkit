"""Unified command-line interface for the prenatal-care-toolkit.

    pctk deid-dicom    in.dcm out.dcm        # Problem 1: header de-id
    pctk deid-image    in.png out.png        # Problem 1: pixel-text redaction
    pctk deid-frame    in.dcm out.png        # Problem 1: DICOM frame -> redacted img
    pctk srb           births.csv            # Problem 2: SRB anomaly report
    pctk compliance    forms.csv regs.csv    # Problem 3: PCPNDT audit
    pctk biometry      --bpd 46 --fl 30 ...  # Problem 4: GA / EFW
    pctk backends                            # list available OCR backends
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from typing import List, Optional


def _parse_date(s: Optional[str]):
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%Y%m%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(str(s).strip(), fmt).date()
        except ValueError:
            continue
    return None


# --------------------------------------------------------------------------- #
# Problem 1
# --------------------------------------------------------------------------- #
def cmd_deid_dicom(args) -> int:
    from .deid import deidentify_file, DeidOptions

    opts = DeidOptions(
        date_policy=args.date_policy,
        remove_private_tags=not args.keep_private,
        keep_patient_sex=args.keep_patient_sex,
    )
    deidentify_file(args.infile, args.outfile, opts)
    print(f"[deid-dicom] wrote {args.outfile}  ({len(opts.audit_log)} actions)")
    if args.verbose:
        for line in opts.audit_log:
            print("   -", line)
    return 0


def cmd_deid_image(args) -> int:
    from .deid import redact_image_file, RedactOptions

    opts = RedactOptions(
        method=args.method,
        margin_only=not args.whole_image,
        ocr_backend=args.ocr_backend,
        also_redact_bands=args.also_redact_bands,
    )
    redact_image_file(args.infile, args.outfile, opts)
    print(f"[deid-image] wrote {args.outfile}")
    for line in opts.audit_log:
        print("   -", line)
    return 0


def cmd_deid_frame(args) -> int:
    from .deid.pixel_text import redact_dicom_frame, RedactOptions

    opts = RedactOptions(method=args.method, ocr_backend=args.ocr_backend)
    redact_dicom_frame(args.infile, args.outfile, frame=args.frame, opts=opts)
    print(f"[deid-frame] wrote {args.outfile}")
    for line in opts.audit_log:
        print("   -", line)
    return 0


# --------------------------------------------------------------------------- #
# Problem 2
# --------------------------------------------------------------------------- #
def cmd_srb(args) -> int:
    from .srb import (load_births, compute_srb, flag_anomalies, SRBConfig,
                      from_aggregated_counts, adapt, SOURCE_NOTES)

    cfg = SRBConfig(
        sex_col=args.sex_col,
        count_col=args.count_col,
        group_cols=args.group or ["district"],
        min_births=args.min_births,
        alpha=args.alpha,
    )
    df = load_births(args.infile)

    # Choose ingestion path: named public-source adapter, aggregated wide
    # counts, or the default per-birth (long) layout.
    if args.source:
        print(f"[srb] adapter '{args.source}': {SOURCE_NOTES.get(args.source, '')}")
        kwargs = {}
        if args.group:
            kwargs["group_cols"] = args.group
        table = adapt(args.source, df, **kwargs)
    elif args.males_col and args.females_col:
        table = from_aggregated_counts(
            df, cfg.group_cols, args.males_col, args.females_col)
    else:
        table = compute_srb(df, cfg)

    # Ratio-only sources carry no denominators -> no significance test.
    if "counts_available" in table.columns and not table["counts_available"].any():
        out = table.sort_values("srb_f_per_1000_m").reset_index(drop=True)
        print(f"[srb] ratio-only source: {len(out)} units (no counts -> no "
              f"significance test). Lowest female ratios first:")
        cols = [c for c in out.columns if c != "counts_available"]
        with_pd_display(out[cols], args.top)
        if args.out:
            out.to_csv(args.out, index=False)
            print(f"[srb] full table -> {args.out}")
        return 0

    flagged = flag_anomalies(table, cfg)
    metric_cols = ["females", "males", "total", "pct_female",
                   "srb_f_per_1000_m", "expected_females", "female_deficit",
                   "p_value", "significant"]
    # Dimension columns = whatever isn't a computed metric.
    dim_cols = [c for c in flagged.columns if c not in metric_cols]
    cols = dim_cols + [c for c in metric_cols if c in flagged.columns
                       and c != "expected_females"]
    sig = flagged[flagged["significant"]] if "significant" in flagged else flagged
    print(f"[srb] {len(flagged)} units >= {cfg.min_births} births; "
          f"{len(sig)} flagged at alpha={cfg.alpha}")
    with_pd_display(flagged[cols], args.top)
    if args.out:
        flagged.to_csv(args.out, index=False)
        print(f"[srb] full table -> {args.out}")
    return 0


def with_pd_display(df, top: int) -> None:
    import pandas as pd
    with pd.option_context("display.max_rows", top,
                           "display.width", 160,
                           "display.float_format", lambda v: f"{v:.3g}"):
        print(df.head(top).to_string(index=False))


# --------------------------------------------------------------------------- #
# Problem 3
# --------------------------------------------------------------------------- #
def cmd_compliance(args) -> int:
    import csv
    from .compliance import FormF, MachineRegistration, audit_records

    forms: List[FormF] = []
    with open(args.forms, newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            ga = row.get("gestational_age_weeks") or ""
            forms.append(FormF(
                patient_ref=row.get("patient_ref", ""),
                facility_id=row.get("facility_id", ""),
                machine_id=row.get("machine_id", ""),
                scan_date=_parse_date(row.get("scan_date")),
                referring_doctor=row.get("referring_doctor", ""),
                indication=row.get("indication", ""),
                gestational_age_weeks=float(ga) if ga else None,
                performed_by=row.get("performed_by", ""),
                record_id=row.get("record_id", f"row{i}"),
            ))

    regs: List[MachineRegistration] = []
    with open(args.registrations, newline="") as fh:
        for row in csv.DictReader(fh):
            regs.append(MachineRegistration(
                machine_id=row.get("machine_id", ""),
                facility_id=row.get("facility_id", ""),
                registered_from=_parse_date(row.get("registered_from")) or date.min,
                registered_to=_parse_date(row.get("registered_to")),
            ))

    result = audit_records(forms, regs)
    c = result["counts"]
    print(f"[compliance] {c['records']} records, {c['findings']} findings "
          f"(high={c['high']} medium={c['medium']} low={c['low']})")
    print("  facility completeness:")
    for fac, s in result["facility_summary"].items():
        print(f"    {fac:14} total={s['total']:4d} "
              f"complete={s['completeness_pct']:5.1f}%  "
              f"unregistered={s['unregistered']}  "
              f"expired={s['expired_retention']}")
    if args.verbose:
        for f in result["findings"]:
            print(f"    [{f.severity:6}] {f.record_id:8} {f.rule:22} {f.detail}")
    return 0


# --------------------------------------------------------------------------- #
# Problem 4
# --------------------------------------------------------------------------- #
def cmd_ingest_formf(args) -> int:
    from .compliance import ingest_form_f, validate_form_f

    form, raw = ingest_form_f(
        args.infile,
        facility_id=args.facility_id,
        ocr_backend=args.ocr_backend,
    )
    print(f"[ingest-formf] {args.infile}")
    fields = ["record_id", "patient_ref", "facility_id", "machine_id",
              "scan_date", "referring_doctor", "indication",
              "gestational_age_weeks", "performed_by"]
    for f in fields:
        print(f"   {f:22} = {getattr(form, f) or '<not found>'}")
    miss = form.missing_fields()
    if miss:
        print(f"   ! missing required fields: {', '.join(miss)}")
    findings = validate_form_f(form)
    for fn in findings:
        print(f"   [{fn.severity}] {fn.rule}: {fn.detail}")
    if args.append_csv:
        _append_form_csv(args.append_csv, form)
        print(f"   -> appended to {args.append_csv}")
    return 0


def _append_form_csv(path: str, form) -> None:
    import csv
    import os
    header = ["record_id", "facility_id", "machine_id", "scan_date",
              "referring_doctor", "indication", "gestational_age_weeks",
              "performed_by", "patient_ref"]
    exists = os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if not exists:
            w.writerow(header)
        w.writerow([getattr(form, h) if getattr(form, h) is not None else ""
                    for h in header])


def cmd_biometry(args) -> int:
    from .biometry import ga_from_measurements, estimated_fetal_weight_hadlock

    res = ga_from_measurements(args.bpd, args.hc, args.ac, args.fl)
    print("[biometry] gestational age")
    for m, v in res["per_measure"].items():
        print(f"   {m.upper():4} -> {v:.2f} w")
    print(f"   composite GA: {res['ga_weeks']:.2f} w  ({res['ga_str']})")
    if args.ac and args.fl:
        efw = estimated_fetal_weight_hadlock(args.ac, args.fl, args.bpd, args.hc)
        print(f"   estimated fetal weight (Hadlock): {efw:.0f} g")
    return 0


def cmd_planes_demo(args) -> int:
    import tempfile
    from .planes import make_synthetic_planes, load_fetal_planes_db, PlaneClassifier

    root = args.data or tempfile.mkdtemp(prefix="pctk_planes_")
    print(f"[planes-demo] generating synthetic FETAL_PLANES_DB-shaped data in {root}")
    make_synthetic_planes(root, n_per_class=args.n_per_class, seed=0)
    df = load_fetal_planes_db(root)
    clf = PlaneClassifier()
    info = clf.train(df)
    res = clf.evaluate(df)
    print(f"[planes-demo] trained on {info['n_train']} imgs, "
          f"{info['n_features']} features, {len(info['classes'])} classes")
    print(f"[planes-demo] test accuracy: {res['accuracy']:.3f} "
          f"(n={res['n_test']})")
    print(res["report"])
    if args.out:
        clf.save(args.out)
        print(f"[planes-demo] model -> {args.out}")
    return 0


def _plane_clf(backend: str, epochs: int = 8):
    if backend == "torch":
        from .planes.torch_cnn import CNNPlaneClassifier, CNNConfig
        return CNNPlaneClassifier(CNNConfig(epochs=epochs))
    from .planes import PlaneClassifier
    return PlaneClassifier()


def _load_plane_clf(backend: str, path: str):
    if backend == "torch":
        from .planes.torch_cnn import CNNPlaneClassifier
        return CNNPlaneClassifier.load(path)
    from .planes import PlaneClassifier
    return PlaneClassifier.load(path)


def cmd_planes_train(args) -> int:
    from .planes import load_fetal_planes_db

    df = load_fetal_planes_db(args.data)
    clf = _plane_clf(args.backend, args.epochs)
    info = clf.train(df)
    print(f"[planes-train:{args.backend}] {info['n_train']} imgs, "
          f"{len(info['classes'])} classes: {info['classes']}")
    if "test" in set(df.get("split", [])):
        res = clf.evaluate(df)
        print(f"[planes-train] held-out accuracy: {res['accuracy']:.3f}")
    clf.save(args.out)
    print(f"[planes-train] model -> {args.out}")
    return 0


def cmd_planes_eval(args) -> int:
    from .planes import load_fetal_planes_db

    clf = _load_plane_clf(args.backend, args.model)
    df = load_fetal_planes_db(args.data)
    res = clf.evaluate(df, split=args.split)
    print(f"[planes-eval] accuracy: {res['accuracy']:.3f} (n={res['n_test']})")
    print(res["report"])
    return 0


def cmd_planes_predict(args) -> int:
    clf = _load_plane_clf(args.backend, args.model)
    out = clf.predict(args.image)
    print(f"[planes-predict] {args.image}")
    print(f"   plane: {out['label']}"
          + (f"  (confidence {out['confidence']:.2f})"
             if "confidence" in out else ""))
    if "proba" in out:
        top = sorted(out["proba"].items(), key=lambda kv: -kv[1])[:3]
        for cls, p in top:
            print(f"     {cls:16} {p:.2f}")
    return 0


def cmd_health_demo(args) -> int:
    import tempfile
    from .health import (make_synthetic_health, load_fetal_health,
                         FetalHealthClassifier)

    csv = args.data or os.path.join(tempfile.mkdtemp(prefix="pctk_health_"),
                                    "ctg.csv")
    print(f"[health-demo] generating synthetic CTG data -> {csv}")
    make_synthetic_health(csv, n=args.n)
    X, y = load_fetal_health(csv)
    clf = FetalHealthClassifier()
    res = clf.fit_eval(X, y)
    print(f"[health-demo] train={res['n_train']} test={res['n_test']}  "
          f"accuracy={res['accuracy']:.3f}  macro-F1={res['macro_f1']:.3f}")
    print(res["report"])
    print("[health-demo] top features:")
    for name, imp in clf.feature_importances(top=6):
        print(f"   {imp:.3f}  {name}")
    if args.out:
        clf.save(args.out)
        print(f"[health-demo] model -> {args.out}")
    return 0


def cmd_health_train(args) -> int:
    from .health import load_fetal_health, FetalHealthClassifier

    X, y = load_fetal_health(args.data)
    clf = FetalHealthClassifier()
    res = clf.fit_eval(X, y)
    print(f"[health-train] accuracy={res['accuracy']:.3f}  "
          f"macro-F1={res['macro_f1']:.3f}")
    print(res["report"])
    clf.save(args.out)
    print(f"[health-train] model -> {args.out}")
    return 0


def cmd_health_predict(args) -> int:
    import json
    from .health import FetalHealthClassifier

    clf = FetalHealthClassifier.load(args.model)
    with open(args.json) as fh:
        row = json.load(fh)
    out = clf.predict(row)
    print(f"[health-predict] status: {out['status']} (class {out['label']})")
    if "proba" in out:
        for k, v in sorted(out["proba"].items(), key=lambda kv: -kv[1]):
            print(f"   {k:14} {v:.2f}")
    return 0


def cmd_backends(args) -> int:
    from .deid.ocr_backends import available_backends
    found = available_backends()
    if found:
        print("[backends] OCR available:", ", ".join(found))
    else:
        print("[backends] no OCR engine installed; pixel redaction uses the "
              "margin-band fallback.\n  install one of:\n"
              "    pip install easyocr        (no system binary)\n"
              "    pip install pytesseract    (needs the tesseract binary)")
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pctk",
                                description="Prenatal-care toolkit (de-id, SRB, "
                                            "PCPNDT compliance, biometry).")
    p.add_argument("--version", action="store_true", help="print version and exit")
    sub = p.add_subparsers(dest="command")

    d1 = sub.add_parser("deid-dicom", help="de-identify DICOM header")
    d1.add_argument("infile"); d1.add_argument("outfile")
    d1.add_argument("--date-policy", choices=["keep", "blank", "shift"],
                    default="shift")
    d1.add_argument("--keep-private", action="store_true")
    d1.add_argument("--keep-patient-sex", action="store_true")
    d1.add_argument("-v", "--verbose", action="store_true")
    d1.set_defaults(func=cmd_deid_dicom)

    d2 = sub.add_parser("deid-image", help="redact burned-in text in an image")
    d2.add_argument("infile"); d2.add_argument("outfile")
    d2.add_argument("--method", choices=["blackout", "blur", "pixelate"],
                    default="blackout")
    d2.add_argument("--whole-image", action="store_true",
                    help="redact detected text anywhere, not only margins")
    d2.add_argument("--ocr-backend", choices=["easyocr", "pytesseract"],
                    default=None)
    d2.add_argument("--also-redact-bands", action="store_true",
                    help="belt-and-suspenders: also blank margin overlay bands")
    d2.set_defaults(func=cmd_deid_image)

    d3 = sub.add_parser("deid-frame", help="DICOM frame -> redacted image")
    d3.add_argument("infile"); d3.add_argument("outfile")
    d3.add_argument("--frame", type=int, default=0)
    d3.add_argument("--method", choices=["blackout", "blur", "pixelate"],
                    default="blackout")
    d3.add_argument("--ocr-backend", choices=["easyocr", "pytesseract"],
                    default=None)
    d3.set_defaults(func=cmd_deid_frame)

    s = sub.add_parser("srb", help="sex-ratio-at-birth anomaly report")
    s.add_argument("infile")
    s.add_argument("--sex-col", default="sex")
    s.add_argument("--count-col", default=None)
    s.add_argument("--source",
                   choices=["crs", "hmis", "ratio", "census", "nfhs", "child06"],
                   default=None,
                   help="map a real public dataset via its adapter")
    s.add_argument("--males-col", default=None,
                   help="aggregated input: male-count column")
    s.add_argument("--females-col", default=None,
                   help="aggregated input: female-count column")
    s.add_argument("--group", nargs="+", default=None)
    s.add_argument("--min-births", type=int, default=30)
    s.add_argument("--alpha", type=float, default=0.01)
    s.add_argument("--top", type=int, default=20)
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_srb)

    c = sub.add_parser("compliance", help="PCPNDT Form-F audit")
    c.add_argument("forms"); c.add_argument("registrations")
    c.add_argument("-v", "--verbose", action="store_true")
    c.set_defaults(func=cmd_compliance)

    ig = sub.add_parser("ingest-formf",
                        help="extract a Form F from a PDF/image into a record")
    ig.add_argument("infile")
    ig.add_argument("--facility-id", default=None)
    ig.add_argument("--ocr-backend", choices=["easyocr", "pytesseract"],
                    default=None)
    ig.add_argument("--append-csv", default=None,
                    help="append the extracted record to this forms CSV")
    ig.set_defaults(func=cmd_ingest_formf)

    b = sub.add_parser("biometry", help="gestational age / fetal weight")
    b.add_argument("--bpd", type=float, default=None)
    b.add_argument("--hc", type=float, default=None)
    b.add_argument("--ac", type=float, default=None)
    b.add_argument("--fl", type=float, default=None)
    b.set_defaults(func=cmd_biometry)

    sub.add_parser("backends", help="list available OCR backends").set_defaults(
        func=cmd_backends)

    # -- fetal-plane classifier (sex-neutral) -- #
    pd_ = sub.add_parser("planes-demo",
                         help="synthetic end-to-end train+eval of the plane model")
    pd_.add_argument("--data", default=None,
                     help="dir to write synthetic data (default: temp)")
    pd_.add_argument("--n-per-class", type=int, default=40)
    pd_.add_argument("--out", default=None, help="save trained model here")
    pd_.set_defaults(func=cmd_planes_demo)

    pt = sub.add_parser("planes-train", help="train plane model on FETAL_PLANES_DB")
    pt.add_argument("data", help="dataset root (Images/ + *_data.csv)")
    pt.add_argument("--out", default="plane_model.joblib")
    pt.add_argument("--backend", choices=["sklearn", "torch"], default="sklearn")
    pt.add_argument("--epochs", type=int, default=8, help="torch backend only")
    pt.set_defaults(func=cmd_planes_train)

    pe = sub.add_parser("planes-eval", help="evaluate a saved plane model")
    pe.add_argument("data"); pe.add_argument("--model", required=True)
    pe.add_argument("--split", default="test")
    pe.add_argument("--backend", choices=["sklearn", "torch"], default="sklearn")
    pe.set_defaults(func=cmd_planes_eval)

    pp = sub.add_parser("planes-predict", help="classify one ultrasound frame")
    pp.add_argument("image"); pp.add_argument("--model", required=True)
    pp.add_argument("--backend", choices=["sklearn", "torch"], default="sklearn")
    pp.set_defaults(func=cmd_planes_predict)

    # -- fetal-health (CTG) classifier (sex-neutral) -- #
    hd = sub.add_parser("health-demo",
                        help="synthetic end-to-end train+eval of the CTG model")
    hd.add_argument("--data", default=None, help="CSV to write synthetic data")
    hd.add_argument("--n", type=int, default=1200)
    hd.add_argument("--out", default=None)
    hd.set_defaults(func=cmd_health_demo)

    ht = sub.add_parser("health-train", help="train CTG model on a fetal-health CSV")
    ht.add_argument("data"); ht.add_argument("--out", default="health_model.joblib")
    ht.set_defaults(func=cmd_health_train)

    hp = sub.add_parser("health-predict", help="classify one CTG record (JSON)")
    hp.add_argument("json"); hp.add_argument("--model", required=True)
    hp.set_defaults(func=cmd_health_predict)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "version", False):
        from . import __version__
        print(f"pctk {__version__}")
        return 0
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except ImportError as exc:
        print(f"[pctk] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
