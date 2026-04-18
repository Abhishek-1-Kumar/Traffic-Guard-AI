"""
TrafficGuard AI — Streamlit App
================================
GitHub + Streamlit Cloud deployment ready.

Model classes:
  model1_vehicle_detection.pt → ['Bicycle','Bus','Bhotbhoti','Car','CNG','Easybike',
                                  'Leguna','Motorbike','MPV','Pedestrian','Pickup',
                                  'PowerTiller','Rickshaw','ShoppingVan','Truck','Van',
                                  'Wheelbarrow']
                                  → filtered to: Car  +  Motorbike
  model2_seatbelt.pt          → ['Seatbelt', 'NoSeatbelt']
  model3_helmet.pt            → ['With Helmet', 'Without Helmet']
  model4_license_plate.pt     → ['LisencePlate']

Pipeline:
  Image ──► model1 (vehicle detection, multi-class → filter Car / Motorbike)
              ├─ Motorbike ──► model3 (helmet: 'With Helmet' / 'Without Helmet')
              │                   Without Helmet ──► model4 (plate) ──► OCR ──► Violation
              └─ Car ──────► model2 (seatbelt)
                              NoSeatbelt ──► model4 (plate) ──► OCR ──► Violation

Run:  streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import time
import re

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="TrafficGuard AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600&display=swap');

:root {
    --bg:    #080b10;
    --surf:  #0d1117;
    --card:  #161b22;
    --bord:  #21262d;
    --acc:   #f0a500;
    --red:   #ff4444;
    --grn:   #00cc66;
    --text:  #e6edf3;
    --mute:  #7d8590;
    --head:  'Rajdhani', sans-serif;
    --mono:  'Share Tech Mono', monospace;
    --body:  'Exo 2', sans-serif;
}
html, body, [class*="css"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--body) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 3rem !important; max-width: 1400px !important; }

/* ─ Hero ─ */
.hero {
    background: linear-gradient(135deg,#0d1117 0%,#161b22 50%,#1a1200 100%);
    border: 1px solid #f0a50030;
    border-radius: 14px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content:'';
    position:absolute; inset:0;
    background: repeating-linear-gradient(
        90deg,transparent,transparent 70px,#f0a5000a 70px,#f0a5000a 71px);
    pointer-events:none;
}
.hero-title {
    font-family: var(--head);
    font-size: 2.8rem; font-weight: 700;
    letter-spacing: 4px; color: var(--acc);
    text-transform: uppercase; margin: 0; line-height: 1;
}
.hero-sub {
    font-family: var(--mono);
    font-size: 0.72rem; color: var(--mute);
    letter-spacing: 2px; margin-top: 0.5rem;
}
.hero-pip { display:flex; gap:8px; margin-top:1.1rem; flex-wrap:wrap; }
.pip {
    font-family:var(--mono); font-size:0.62rem;
    letter-spacing:1px; padding:3px 10px; border-radius:3px; border:1px solid;
}
.pip-v { color:#7ee787; border-color:#7ee78744; background:#7ee78710; }
.pip-h { color:#f0a500; border-color:#f0a50044; background:#f0a50010; }
.pip-s { color:#79c0ff; border-color:#79c0ff44; background:#79c0ff10; }
.pip-p { color:#d2a8ff; border-color:#d2a8ff44; background:#d2a8ff10; }

/* ─ Section label ─ */
.slabel {
    font-family:var(--mono); font-size:0.65rem;
    letter-spacing:3px; color:var(--acc);
    text-transform:uppercase;
    border-left:3px solid var(--acc); padding-left:10px;
    margin: 1.6rem 0 0.8rem;
}

/* ─ Upload zone ─ */
[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 2px dashed #f0a50040 !important;
    border-radius: 10px !important;
}

/* ─ Metric cards ─ */
.mrow { display:flex; gap:12px; flex-wrap:wrap; margin:1rem 0 1.4rem; }
.mcard {
    flex:1; min-width:110px;
    background:var(--card); border:1px solid var(--bord);
    border-radius:8px; padding:14px 16px; position:relative; overflow:hidden;
}
.mcard::after {
    content:''; position:absolute; top:0;left:0;right:0; height:2px;
}
.mc-red::after  { background:var(--red); }
.mc-grn::after  { background:var(--grn); }
.mc-acc::after  { background:var(--acc); }
.mc-blue::after { background:#79c0ff; }
.mcard-num {
    font-family:var(--head); font-size:2rem; font-weight:700; line-height:1;
}
.mc-red  .mcard-num { color:var(--red); }
.mc-grn  .mcard-num { color:var(--grn); }
.mc-acc  .mcard-num { color:var(--acc); }
.mc-blue .mcard-num { color:#79c0ff;    }
.mcard-lbl {
    font-family:var(--mono); font-size:0.6rem;
    letter-spacing:1.5px; color:var(--mute); text-transform:uppercase; margin-top:4px;
}

/* ─ Violation card ─ */
.vcard {
    background:var(--card); border:1px solid #ff444430;
    border-left:4px solid var(--red); border-radius:8px;
    padding:16px 18px; margin-bottom:14px;
}
.vcard-head {
    font-family:var(--head); font-size:1rem; font-weight:600;
    color:var(--red); letter-spacing:1px; margin-bottom:8px;
}
.vcard-plate {
    font-family:var(--mono); font-size:1.5rem;
    color:var(--acc); letter-spacing:4px;
    background:#f0a50012; border:1px solid #f0a50033;
    border-radius:6px; padding:6px 16px;
    display:inline-block; margin:4px 0 8px;
}
.vcard-meta { font-family:var(--mono); font-size:0.65rem; color:var(--mute); letter-spacing:1px; }
.conf-badge {
    display:inline-block; font-family:var(--mono);
    font-size:0.62rem; padding:2px 8px; border-radius:3px;
    letter-spacing:1px; margin-left:6px;
}
.cb-red  { background:#ff444420; color:var(--red);  border:1px solid #ff444440; }
.cb-acc  { background:#f0a50015; color:var(--acc);  border:1px solid #f0a50033; }
.cb-grn  { background:#00cc6615; color:var(--grn);  border:1px solid #00cc6630; }

/* ─ Compliant card ─ */
.okcard {
    background:var(--card); border:1px solid #00cc6625;
    border-left:4px solid var(--grn); border-radius:8px;
    padding:14px 18px; margin-bottom:10px;
}
.okcard-head { font-family:var(--head); font-size:0.9rem; color:var(--grn); letter-spacing:1px; }

/* ─ Sidebar ─ */
[data-testid="stSidebar"] { background: var(--surf) !important; border-right: 1px solid var(--bord) !important; }
.sb-title {
    font-family:var(--head); font-size:1.1rem; font-weight:700;
    color:var(--acc); letter-spacing:2px; text-transform:uppercase; margin-bottom:1rem;
}
.sb-section {
    font-family:var(--mono); font-size:0.6rem;
    letter-spacing:2px; color:var(--mute);
    text-transform:uppercase; margin: 1.2rem 0 0.5rem;
}
.model-row {
    font-family:var(--mono); font-size:0.65rem; color:var(--text);
    background:var(--card); border:1px solid var(--bord);
    border-radius:5px; padding:7px 10px; margin-bottom:6px; letter-spacing:.5px;
}
.model-dot {
    display:inline-block; width:7px;height:7px;
    border-radius:50%; margin-right:7px; vertical-align:middle;
}

/* ─ Status log ─ */
.status-log {
    background:#0d1117; border:1px solid #21262d;
    border-radius:8px; padding:12px 16px;
    font-family:'Share Tech Mono',monospace;
}
.status-line { font-size:0.68rem; letter-spacing:1px; color:#7d8590; padding:3px 0; }
.status-line span { color:#f0a500; }

/* ─ Button ─ */
div.stButton > button {
    background: #f0a500 !important; color: #000 !important;
    font-family: var(--head) !important; font-weight: 700 !important;
    font-size: 1rem !important; letter-spacing: 2px !important;
    text-transform: uppercase !important; border: none !important;
    border-radius: 6px !important; padding: 10px 28px !important;
    width: 100%;
}
div.stButton > button:hover { opacity:.85 !important; }
hr { border-color: var(--bord) !important; margin: 1.6rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def safe_crop(img: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
    H, W = img.shape[:2]
    return img[max(0,int(y1)):min(H,int(y2)), max(0,int(x1)):min(W,int(x2))]

def run_model(model, img: np.ndarray, conf_thresh: float):
    results = model.predict(img, conf=conf_thresh, verbose=False)[0]
    out = []
    for box in results.boxes:
        cid   = int(box.cls[0])
        cname = model.names[cid]
        conf  = float(box.conf[0])
        xyxy  = box.xyxy[0].cpu().numpy().tolist()
        out.append({"class": cname, "conf": round(conf, 3), "box": xyxy})
    return out

def extend_moto_crop(img, box, top_ext_pct=60, side_pct=8, bottom_pct=10):
    """
    Extend motorcycle bounding box UPWARD to capture driver + helmet.
    The rider sits above the bike frame, so the helmet is above y1.
    A small bottom pad retains the license plate area.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    new_y1 = y1 - h * top_ext_pct / 100   # ← extend UP for helmet/driver
    new_y2 = y2 + h * bottom_pct / 100    # small downward pad for plate
    new_x1 = x1 - w * side_pct / 100
    new_x2 = x2 + w * side_pct / 100
    return safe_crop(img, new_x1, new_y1, new_x2, new_y2)

def extend_moto_box(box, img_shape, top_ext_pct=60, side_pct=8, bottom_pct=10):
    """Return the extended box coords (clamped) for annotation drawing."""
    H, W = img_shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    nx1 = max(0, x1 - w * side_pct / 100)
    ny1 = max(0, y1 - h * top_ext_pct / 100)
    nx2 = min(W, x2 + w * side_pct / 100)
    ny2 = min(H, y2 + h * bottom_pct / 100)
    return [nx1, ny1, nx2, ny2]

def extend_car_crop(img, box, side_pct=10, top_pct=18, bottom_pct=22):
    """
    Extend car bounding box with generous padding on all sides.
    Increased top (18%) and bottom (22%) to ensure the full car body
    and license plate (often at rear/bottom edge) are captured.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    new_x1 = x1 - w * side_pct / 100
    new_y1 = y1 - h * top_pct / 100
    new_x2 = x2 + w * side_pct / 100
    new_y2 = y2 + h * bottom_pct / 100
    return safe_crop(img, new_x1, new_y1, new_x2, new_y2)

def extend_car_box(box, img_shape, side_pct=10, top_pct=18, bottom_pct=22):
    """Return extended car box coords (clamped) for annotation drawing."""
    H, W = img_shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    nx1 = max(0, x1 - w * side_pct / 100)
    ny1 = max(0, y1 - h * top_pct / 100)
    nx2 = min(W, x2 + w * side_pct / 100)
    ny2 = min(H, y2 + h * bottom_pct / 100)
    return [nx1, ny1, nx2, ny2]

def clean_plate_text(raw: str) -> str:
    txt = re.sub(r"[^A-Z0-9\- ]", "", raw.upper()).strip()
    return re.sub(r"\s+", " ", txt) or "UNREADABLE"

def ocr_plate(reader, crop: np.ndarray):
    """Returns (plate_text, ocr_confidence)"""
    if crop is None or crop.size == 0:
        return "UNREADABLE", 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scale = max(1, 180 // max(gray.shape[0], 1))
    if scale > 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    gray  = clahe.apply(gray)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    res   = reader.readtext(th, detail=1, paragraph=False,
                            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ")
    if not res:
        res = reader.readtext(crop, detail=1, paragraph=False)
    if not res:
        return "UNREADABLE", 0.0
    res_sorted = sorted(res, key=lambda r: r[0][0][0])
    text = " ".join(r[1] for r in res_sorted)
    conf = float(np.mean([r[2] for r in res_sorted]))
    return clean_plate_text(text), round(conf, 3)

def conf_badge_class(val):
    if val >= 0.65: return "cb-grn"
    if val >= 0.40: return "cb-acc"
    return "cb-red"

def draw_boxes(img: np.ndarray, detections: list, color_map: dict) -> np.ndarray:
    out = img.copy()
    for d in detections:
        x1,y1,x2,y2 = [int(v) for v in d["box"]]
        color = color_map.get(d["class"], (180, 180, 180))
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        label = f"{d['class']}  {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
        cv2.putText(out, label, (x1+3, y1-4),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (0,0,0), 1)
    return out

def np_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# ══════════════════════════════════════════════════════════════════════════════
# CACHED MODEL + OCR LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_models_cached():
    from ultralytics import YOLO
    return {
        # Multi-class vehicle detector (17 classes) — we filter Car + Motorbike
        "vehicle":  YOLO("model1_vehicle_detection (1).pt"),
        "seatbelt": YOLO("model2_seatbelt.pt"),
        # New helmet model: classes → ['With Helmet', 'Without Helmet']
        "helmet":   YOLO("model3_helmet (1).pt"),
        "plate":    YOLO("model4_license_plate.pt"),
    }

@st.cache_resource(show_spinner=False)
def load_ocr_cached():
    import easyocr
    return easyocr.Reader(["en"], gpu=False)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(img_bgr, models, reader, conf, moto_ext_pct, log_cb):
    violations     = []
    clean_vehicles = []
    anno_boxes     = []   # for annotated output image

    # ── Vehicle classes the new model uses ───────────────────────────────
    # Full set: Bicycle, Bus, Bhotbhoti, Car, CNG, Easybike, Leguna,
    #           Motorbike, MPV, Pedestrian, Pickup, PowerTiller, Rickshaw,
    #           ShoppingVan, Truck, Van, Wheelbarrow
    # We care about:  'Car'  and  'Motorbike'
    CAR_CLASS   = "Car"
    MOTO_CLASS  = "Motorbike"

    # ── New helmet model classes ──────────────────────────────────────────
    # 'With Helmet'  /  'Without Helmet'  (spaces, not camelCase)
    HELMET_YES  = "With Helmet"
    HELMET_NO   = "Without Helmet"

    # ── Step 1 : vehicle detection ────────────────────────────────────────
    log_cb("Running vehicle detection (model1)…", 10)
    veh_dets = run_model(models["vehicle"], img_bgr, conf["vehicle"])
    motos = [d for d in veh_dets if d["class"] == MOTO_CLASS]
    cars  = [d for d in veh_dets if d["class"] == CAR_CLASS]
    log_cb(f"Detected → {len(motos)} motorbike(s), {len(cars)} car(s)", 20)

    # ── Step 2a : motorbikes ──────────────────────────────────────────────
    for idx, moto in enumerate(motos):
        log_cb(f"Motorbike {idx+1}/{len(motos)} → helmet model (model3)…", 30)
        crop = extend_moto_crop(img_bgr, moto["box"],
                                top_ext_pct=moto_ext_pct, side_pct=8, bottom_pct=10)
        ext_box = extend_moto_box(moto["box"], img_bgr.shape,
                                  top_ext_pct=moto_ext_pct, side_pct=8, bottom_pct=10)
        if crop.size == 0: continue

        h_dets    = run_model(models["helmet"], crop, conf["helmet"])
        h_classes = [d["class"] for d in h_dets]

        has_helmet = HELMET_YES in h_classes
        no_helmet  = HELMET_NO  in h_classes

        base = {"type": "Motorbike", "veh_conf": moto["conf"], "box": moto["box"]}

        if no_helmet:
            log_cb("  ⚠ Without Helmet → plate detection (model4)…", 50)

            # New helmet model has no plate class → use only model4 for plates
            p_dets    = run_model(models["plate"], crop, conf["plate"])
            plates    = [d for d in p_dets if d["class"] == "LisencePlate"]
            plate_crop_bgr, plate_model_conf = None, 0.0
            if plates:
                best = max(plates, key=lambda d: d["conf"])
                plate_crop_bgr   = safe_crop(crop, *best["box"])
                plate_model_conf = best["conf"]

            plate_text, ocr_conf = ocr_plate(reader, plate_crop_bgr)

            no_h_det = next(d for d in h_dets if d["class"] == HELMET_NO)
            violations.append({
                **base,
                "violation":        "No Helmet",
                "viol_conf":        no_h_det["conf"],
                "plate_text":       plate_text,
                "ocr_conf":         ocr_conf,
                "plate_model_conf": plate_model_conf,
                "crop_bgr":         crop,
                "plate_crop_bgr":   plate_crop_bgr,
            })
            anno_boxes.append({"class":"NO HELMET","conf":moto["conf"],"box":ext_box})

        elif has_helmet:
            clean_vehicles.append({**base, "status": "✅ Helmet Present"})
            anno_boxes.append({"class":"HELMET OK","conf":moto["conf"],"box":ext_box})
        else:
            clean_vehicles.append({**base, "status": "❓ Helmet Uncertain"})
            anno_boxes.append({"class":"UNCERTAIN","conf":moto["conf"],"box":ext_box})

    # ── Step 2b : cars ────────────────────────────────────────────────────
    for idx, car in enumerate(cars):
        log_cb(f"Car {idx+1}/{len(cars)} → seatbelt model (model2)…", 60)
        crop = extend_car_crop(img_bgr, car["box"], side_pct=10, top_pct=18, bottom_pct=22)
        ext_box = extend_car_box(car["box"], img_bgr.shape, side_pct=10, top_pct=18, bottom_pct=22)
        if crop.size == 0: continue

        sb_dets   = run_model(models["seatbelt"], crop, conf["seatbelt"])
        sb_classes = [d["class"] for d in sb_dets]

        has_belt = "Seatbelt"   in sb_classes
        no_belt  = "NoSeatbelt" in sb_classes

        base = {"type": "Car", "veh_conf": car["conf"], "box": car["box"]}

        if no_belt:
            log_cb("  ⚠ NoSeatbelt → plate detection (model4)…", 75)

            p_dets   = run_model(models["plate"], crop, conf["plate"])
            plates   = [d for d in p_dets if d["class"] == "LisencePlate"]
            plate_crop_bgr, plate_model_conf = None, 0.0
            if plates:
                best = max(plates, key=lambda d: d["conf"])
                plate_crop_bgr   = safe_crop(crop, *best["box"])
                plate_model_conf = best["conf"]

            plate_text, ocr_conf = ocr_plate(reader, plate_crop_bgr)
            no_sb_det = next(d for d in sb_dets if d["class"] == "NoSeatbelt")

            violations.append({
                **base,
                "violation":        "No Seatbelt",
                "viol_conf":        no_sb_det["conf"],
                "plate_text":       plate_text,
                "ocr_conf":         ocr_conf,
                "plate_model_conf": plate_model_conf,
                "crop_bgr":         crop,
                "plate_crop_bgr":   plate_crop_bgr,
            })
            anno_boxes.append({"class":"NO SEATBELT","conf":car["conf"],"box":ext_box})

        elif has_belt:
            clean_vehicles.append({**base, "status": "✅ Seatbelt Present"})
            anno_boxes.append({"class":"SEATBELT OK","conf":car["conf"],"box":ext_box})
        else:
            clean_vehicles.append({**base, "status": "❓ Seatbelt Uncertain"})
            anno_boxes.append({"class":"UNCERTAIN","conf":car["conf"],"box":ext_box})

    # ── Annotate main image ───────────────────────────────────────────────
    COLOR_MAP = {
        "NO HELMET":   (0,  60, 255),
        "NO SEATBELT": (0,  60, 255),
        "HELMET OK":   (0, 200,  80),
        "SEATBELT OK": (0, 200,  80),
        "UNCERTAIN":   (0, 165, 255),
    }
    annotated = draw_boxes(img_bgr, anno_boxes, COLOR_MAP)
    log_cb("Pipeline complete.", 100)
    return violations, clean_vehicles, annotated


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sb-title">⚙ CONFIG</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Confidence Thresholds</div>', unsafe_allow_html=True)
    c_v = st.slider("Vehicle (model1)",  0.20, 0.90, 0.40, 0.05)
    c_h = st.slider("Helmet  (model3)",  0.20, 0.90, 0.40, 0.05)
    c_s = st.slider("Seatbelt (model2)", 0.20, 0.90, 0.40, 0.05)
    c_p = st.slider("Plate   (model4)",  0.15, 0.90, 0.35, 0.05)

    st.markdown('<div class="sb-section">Motorcycle Crop</div>', unsafe_allow_html=True)
    moto_ext = st.slider(
        "Height extension ABOVE box (%)", 20, 120, 60, 5,
        help="Adds extra height ABOVE the detected motorcycle box to capture the rider's helmet and head."
    )

    st.markdown('<div class="sb-section">Loaded Models</div>', unsafe_allow_html=True)
    for dot, name, classes in [
        ("#7ee787", "model1_vehicle_detection (1)",
         "Car · Motorbike (+ 15 other classes filtered out)"),
        ("#f0a500", "model3_helmet (1)",
         "With Helmet · Without Helmet"),
        ("#79c0ff", "model2_seatbelt",
         "Seatbelt · NoSeatbelt"),
        ("#d2a8ff", "model4_license_plate",
         "LisencePlate"),
    ]:
        st.markdown(
            f'<div class="model-row">'
            f'<span class="model-dot" style="background:{dot}"></span>'
            f'<b>{name}.pt</b><br>'
            f'<span style="color:#7d8590;font-size:0.58rem">{classes}</span>'
            f'</div>',
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.6rem;'
        'color:#7d8590;letter-spacing:1px">TrafficGuard AI · EE655 Project</div>',
        unsafe_allow_html=True)

conf_dict = {"vehicle": c_v, "helmet": c_h, "seatbelt": c_s, "plate": c_p}


# ══════════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <div class="hero-title">🚦 TrafficGuard AI</div>
  <div class="hero-sub">AUTOMATED TRAFFIC VIOLATION DETECTION SYSTEM · EE655</div>
  <div class="hero-pip">
    <span class="pip pip-v">MODEL 1 · VEHICLE DETECTION</span>
    <span class="pip pip-h">MODEL 3 · HELMET + PLATE</span>
    <span class="pip pip-s">MODEL 2 · SEATBELT</span>
    <span class="pip pip-p">MODEL 4 · LICENSE PLATE</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FILE UPLOAD
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="slabel">📤 Upload Traffic Image</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "", type=["jpg", "jpeg", "png", "bmp", "webp"],
    label_visibility="collapsed"
)

if uploaded is None:
    st.markdown("""
    <div style="text-align:center;padding:4rem;
    font-family:'Share Tech Mono',monospace;font-size:0.75rem;
    color:#7d8590;letter-spacing:2px;
    border:1px dashed #21262d;border-radius:10px;margin-top:1rem">
        UPLOAD A TRAFFIC IMAGE TO BEGIN ANALYSIS
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Decode image ───────────────────────────────────────────────────────────────
file_bytes = np.frombuffer(uploaded.read(), np.uint8)
img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
H, W       = img_bgr.shape[:2]

# ── Preview + run button ───────────────────────────────────────────────────────
col_prev, col_ctrl = st.columns([1.7, 1], gap="large")
with col_prev:
    st.markdown('<div class="slabel">🖼 Input Image</div>', unsafe_allow_html=True)
    st.image(np_to_pil(img_bgr), use_container_width=True)

with col_ctrl:
    st.markdown('<div class="slabel">📋 Image Details</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
    color:#7d8590;line-height:2.4">
    FILENAME &nbsp;&nbsp;→ <span style="color:#e6edf3">{uploaded.name}</span><br>
    RESOLUTION → <span style="color:#e6edf3">{W} × {H} px</span><br>
    SIZE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ <span style="color:#e6edf3">{len(file_bytes)//1024} KB</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div style="margin-top:1.8rem"></div>', unsafe_allow_html=True)
    run_btn = st.button("▶  RUN ANALYSIS")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

if not run_btn:
    st.stop()

# ── Status display ─────────────────────────────────────────────────────────────
progress   = st.progress(0)
status_box = st.empty()
log_lines  = []

def log_cb(msg: str, pct: int):
    log_lines.append(msg)
    lines_html = "".join(
        f'<div class="status-line">›&nbsp;<span>{l}</span></div>'
        for l in log_lines[-7:]
    )
    status_box.markdown(
        f'<div class="status-log">{lines_html}</div>',
        unsafe_allow_html=True)
    progress.progress(pct)

# ── Load models ────────────────────────────────────────────────────────────────
log_cb("Loading YOLO models…", 3)
try:
    models = load_models_cached()
    log_cb("Loading EasyOCR engine…", 6)
    reader = load_ocr_cached()
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# ── Run pipeline ───────────────────────────────────────────────────────────────
t0 = time.time()
violations, clean_vehicles, annotated = run_pipeline(
    img_bgr, models, reader, conf_dict, moto_ext, log_cb)
elapsed = round(time.time() - t0, 2)

status_box.empty()
progress.empty()

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")

# ── Summary metrics ────────────────────────────────────────────────────────────
st.markdown('<div class="slabel">📊 Analysis Summary</div>', unsafe_allow_html=True)

n_total = len(violations) + len(clean_vehicles)
n_nh    = sum(1 for v in violations if "Helmet"   in v["violation"])
n_ns    = sum(1 for v in violations if "Seatbelt" in v["violation"])

st.markdown(f"""
<div class="mrow">
  <div class="mcard mc-acc">
    <div class="mcard-num">{n_total}</div>
    <div class="mcard-lbl">Vehicles Detected</div>
  </div>
  <div class="mcard mc-red">
    <div class="mcard-num">{len(violations)}</div>
    <div class="mcard-lbl">Violations Found</div>
  </div>
  <div class="mcard mc-grn">
    <div class="mcard-num">{len(clean_vehicles)}</div>
    <div class="mcard-lbl">Compliant</div>
  </div>
  <div class="mcard mc-blue">
    <div class="mcard-num">{n_nh}</div>
    <div class="mcard-lbl">No Helmet</div>
  </div>
  <div class="mcard mc-red">
    <div class="mcard-num">{n_ns}</div>
    <div class="mcard-lbl">No Seatbelt</div>
  </div>
  <div class="mcard mc-acc">
    <div class="mcard-num">{elapsed}s</div>
    <div class="mcard-lbl">Inference Time</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Annotated image ────────────────────────────────────────────────────────────
st.markdown('<div class="slabel">🖼 Annotated Detection Result</div>', unsafe_allow_html=True)
st.image(np_to_pil(annotated), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# VIOLATION RECORDS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="slabel">🚨 Violation Records</div>', unsafe_allow_html=True)

if not violations:
    st.markdown("""
    <div class="okcard">
      <div class="okcard-head">✅ No violations detected — all vehicles are compliant.</div>
    </div>""", unsafe_allow_html=True)
else:
    for i, v in enumerate(violations):
        icon = "🏍️" if v["type"] == "Motorbike" else "🚗"
        vc, pc, oc = v["viol_conf"], v["plate_model_conf"], v["ocr_conf"]

        st.markdown(f"""
        <div class="vcard">
          <div class="vcard-head">
            {icon} VIOLATION #{i+1} — {v['violation'].upper()}
            <span class="conf-badge {conf_badge_class(vc)}">VIOL CONF {vc:.3f}</span>
            <span class="conf-badge cb-acc">VEH CONF {v['veh_conf']:.3f}</span>
          </div>
          <div style="margin-bottom:4px;font-family:'Share Tech Mono',monospace;
          font-size:0.65rem;color:#7d8590;letter-spacing:1px">LICENSE PLATE</div>
          <div class="vcard-plate">{v['plate_text']}</div>
          <div class="vcard-meta" style="margin-top:8px">
            PLATE MODEL CONF:
              <span class="conf-badge {conf_badge_class(pc)}">{pc:.3f}</span>
            &nbsp;·&nbsp;
            OCR CONF:
              <span class="conf-badge {conf_badge_class(oc)}">{oc:.3f}</span>
            &nbsp;·&nbsp;
            TYPE: {v['type'].upper()}
          </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="small")
        with c1:
            st.caption(f"Extended vehicle crop — {v['type']} (driver/helmet included)")
            # Draw the extended box outline on the crop itself for clarity
            crop_vis = v["crop_bgr"].copy()
            ch, cw = crop_vis.shape[:2]
            cv2.rectangle(crop_vis, (2, 2), (cw-2, ch-2), (0, 200, 255), 2)
            st.image(np_to_pil(crop_vis), use_container_width=True)
        with c2:
            if v["plate_crop_bgr"] is not None and v["plate_crop_bgr"].size > 0:
                st.caption(f"License plate crop · Plate: {v['plate_text']} · OCR conf: {oc:.3f}")
                st.image(np_to_pil(v["plate_crop_bgr"]), use_container_width=True)
            else:
                st.caption("License plate not detected")
                st.markdown("""
                <div style="background:#161b22;border:1px dashed #21262d;border-radius:6px;
                padding:2.5rem;text-align:center;font-family:'Share Tech Mono',monospace;
                font-size:0.65rem;color:#7d8590;letter-spacing:1px">
                PLATE NOT DETECTED<br>
                <span style="font-size:0.55rem">Try lowering the plate confidence threshold</span>
                </div>""", unsafe_allow_html=True)

        if i < len(violations) - 1:
            st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# COMPLIANT VEHICLES
# ══════════════════════════════════════════════════════════════════════════════

if clean_vehicles:
    st.markdown('<div class="slabel">✅ Compliant Vehicles</div>', unsafe_allow_html=True)
    for cv_ in clean_vehicles:
        icon = "🏍️" if cv_["type"] == "Motorbike" else "🚗"
        st.markdown(f"""
        <div class="okcard">
          <div class="okcard-head">
            {icon} {cv_['type']} — {cv_['status']}
            <span class="conf-badge cb-grn" style="margin-left:8px">
              CONF {cv_['veh_conf']:.3f}
            </span>
          </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD REPORT
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="slabel">💾 Export Report</div>', unsafe_allow_html=True)

report = [
    "=" * 55,
    "   TrafficGuard AI — Violation Report",
    "=" * 55,
    f"  File       : {uploaded.name}",
    f"  Resolution : {W} x {H} px",
    f"  Inference  : {elapsed} s",
    f"  Vehicles   : {n_total}",
    f"  Violations : {len(violations)}",
    f"  Compliant  : {len(clean_vehicles)}",
    "",
]
for i, v in enumerate(violations):
    report += [
        f"  {'─'*50}",
        f"  VIOLATION #{i+1}",
        f"    Type              : {v['type']}",
        f"    Offence           : {v['violation']}",
        f"    Violation Conf    : {v['viol_conf']:.3f}",
        f"    Vehicle Conf      : {v['veh_conf']:.3f}",
        f"    License Plate     : {v['plate_text']}",
        f"    Plate Model Conf  : {v['plate_model_conf']:.3f}",
        f"    OCR Confidence    : {v['ocr_conf']:.3f}",
        "",
    ]
if not violations:
    report.append("  No violations detected.")
report.append("=" * 55)

st.download_button(
    label="⬇  Download Violation Report (.txt)",
    data="\n".join(report),
    file_name=f"violations_{uploaded.name.rsplit('.', 1)[0]}.txt",
    mime="text/plain",
)
