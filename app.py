"""
TrafficGuard AI — Streamlit App
================================
Model files (place in the same directory as app.py):
  model1_vehicle_detection (1).pt → 17-class detector, we use: 'Car', 'Motorbike'
  model2_seatbelt.pt              → ['Seatbelt', 'NoSeatbelt']
  model3_helmet (1).pt            → ['With Helmet', 'Without Helmet']
  model4_license_plate.pt         → ['NumberPlate']

OCR engines:
  EasyOCR  — primary engine  (installed via: pip install easyocr)
  RapidOCR — secondary engine (installed via: pip install rapidocr-onnxruntime)
  Both engines are run over 9 pre-processed plate variants; the highest-
  confidence result across all variants and engines is used.

DEPLOYMENT FIX — invalid load key \\x0d (corrupted .pt files):
  Binary .pt files get corrupted when Git treats them as text and
  converts LF to CRLF. Fix by adding .gitattributes to your repo root:

      *.pt  binary

  OR use Git LFS:
      git lfs track "*.pt"
      git add .gitattributes
      git add *.pt
      git commit -m "track models with LFS"

Run:  streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import re

# ── Page config ───────────────────────────────────────────────────────────────
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
    --bg:#080b10; --surf:#0d1117; --card:#161b22; --bord:#21262d;
    --acc:#f0a500; --red:#ff4444; --grn:#00cc66;
    --text:#e6edf3; --mute:#7d8590;
    --head:'Rajdhani',sans-serif; --mono:'Share Tech Mono',monospace;
    --body:'Exo 2',sans-serif;
}
html,body,[class*="css"]{background:var(--bg)!important;color:var(--text)!important;font-family:var(--body)!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:1.5rem 2rem 3rem!important;max-width:1400px!important;}
.hero{background:linear-gradient(135deg,#0d1117 0%,#161b22 50%,#1a1200 100%);
      border:1px solid #f0a50030;border-radius:14px;padding:2rem 2.5rem;
      margin-bottom:2rem;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;inset:0;
  background:repeating-linear-gradient(90deg,transparent,transparent 70px,#f0a5000a 70px,#f0a5000a 71px);
  pointer-events:none;}
.hero-title{font-family:var(--head);font-size:2.8rem;font-weight:700;
  letter-spacing:4px;color:var(--acc);text-transform:uppercase;margin:0;line-height:1;}
.hero-sub{font-family:var(--mono);font-size:0.72rem;color:var(--mute);
  letter-spacing:2px;margin-top:0.5rem;}
.hero-pip{display:flex;gap:8px;margin-top:1.1rem;flex-wrap:wrap;}
.pip{font-family:var(--mono);font-size:0.62rem;letter-spacing:1px;
  padding:3px 10px;border-radius:3px;border:1px solid;}
.pip-v{color:#7ee787;border-color:#7ee78744;background:#7ee78710;}
.pip-h{color:#f0a500;border-color:#f0a50044;background:#f0a50010;}
.pip-s{color:#79c0ff;border-color:#79c0ff44;background:#79c0ff10;}
.pip-p{color:#d2a8ff;border-color:#d2a8ff44;background:#d2a8ff10;}
.slabel{font-family:var(--mono);font-size:0.65rem;letter-spacing:3px;color:var(--acc);
  text-transform:uppercase;border-left:3px solid var(--acc);padding-left:10px;
  margin:1.6rem 0 0.8rem;}
[data-testid="stFileUploader"]{background:var(--card)!important;
  border:2px dashed #f0a50040!important;border-radius:10px!important;}
.mrow{display:flex;gap:12px;flex-wrap:wrap;margin:1rem 0 1.4rem;}
.mcard{flex:1;min-width:110px;background:var(--card);border:1px solid var(--bord);
  border-radius:8px;padding:14px 16px;position:relative;overflow:hidden;}
.mcard::after{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
.mc-red::after{background:var(--red);} .mc-grn::after{background:var(--grn);}
.mc-acc::after{background:var(--acc);} .mc-blue::after{background:#79c0ff;}
.mcard-num{font-family:var(--head);font-size:2rem;font-weight:700;line-height:1;}
.mc-red .mcard-num{color:var(--red);} .mc-grn .mcard-num{color:var(--grn);}
.mc-acc .mcard-num{color:var(--acc);} .mc-blue .mcard-num{color:#79c0ff;}
.mcard-lbl{font-family:var(--mono);font-size:0.6rem;letter-spacing:1.5px;
  color:var(--mute);text-transform:uppercase;margin-top:4px;}
.vcard{background:var(--card);border:1px solid #ff444430;border-left:4px solid var(--red);
  border-radius:8px;padding:16px 18px;margin-bottom:14px;}
.vcard-head{font-family:var(--head);font-size:1rem;font-weight:600;
  color:var(--red);letter-spacing:1px;margin-bottom:8px;}
.conf-badge{display:inline-block;font-family:var(--mono);font-size:0.62rem;
  padding:2px 8px;border-radius:3px;letter-spacing:1px;margin-left:6px;}
.cb-red{background:#ff444420;color:var(--red);border:1px solid #ff444440;}
.cb-acc{background:#f0a50015;color:var(--acc);border:1px solid #f0a50033;}
.cb-grn{background:#00cc6615;color:var(--grn);border:1px solid #00cc6630;}
.okcard{background:var(--card);border:1px solid #00cc6625;
  border-left:4px solid var(--grn);border-radius:8px;padding:14px 18px;margin-bottom:10px;}
.okcard-head{font-family:var(--head);font-size:0.9rem;color:var(--grn);letter-spacing:1px;}
[data-testid="stSidebar"]{background:var(--surf)!important;border-right:1px solid var(--bord)!important;}
.sb-title{font-family:var(--head);font-size:1.1rem;font-weight:700;
  color:var(--acc);letter-spacing:2px;text-transform:uppercase;margin-bottom:1rem;}
.sb-section{font-family:var(--mono);font-size:0.6rem;letter-spacing:2px;
  color:var(--mute);text-transform:uppercase;margin:1.2rem 0 0.5rem;}
.model-row{font-family:var(--mono);font-size:0.65rem;color:var(--text);
  background:var(--card);border:1px solid var(--bord);
  border-radius:5px;padding:7px 10px;margin-bottom:6px;letter-spacing:.5px;}
.model-dot{display:inline-block;width:7px;height:7px;
  border-radius:50%;margin-right:7px;vertical-align:middle;}
.status-log{background:#0d1117;border:1px solid #21262d;
  border-radius:8px;padding:12px 16px;font-family:'Share Tech Mono',monospace;}
.status-line{font-size:0.68rem;letter-spacing:1px;color:#7d8590;padding:3px 0;}
.status-line span{color:#f0a500;}
div.stButton>button{background:#f0a500!important;color:#000!important;
  font-family:var(--head)!important;font-weight:700!important;font-size:1rem!important;
  letter-spacing:2px!important;text-transform:uppercase!important;border:none!important;
  border-radius:6px!important;padding:10px 28px!important;width:100%;}
div.stButton>button:hover{opacity:.85!important;}
hr{border-color:var(--bord)!important;margin:1.6rem 0!important;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY HELPERS
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
    """Extend UP to capture driver + helmet; small pad down for plate."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return safe_crop(img,
        x1 - w * side_pct / 100,
        y1 - h * top_ext_pct / 100,
        x2 + w * side_pct / 100,
        y2 + h * bottom_pct / 100)


def extend_moto_box(box, img_shape, top_ext_pct=60, side_pct=8, bottom_pct=10):
    H, W = img_shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return [max(0, x1 - w*side_pct/100), max(0, y1 - h*top_ext_pct/100),
            min(W, x2 + w*side_pct/100), min(H, y2 + h*bottom_pct/100)]


def extend_car_crop(img, box, side_pct=10, top_pct=18, bottom_pct=22):
    """Generous padding — top for windshield, bottom for rear plate."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return safe_crop(img,
        x1 - w*side_pct/100,
        y1 - h*top_pct/100,
        x2 + w*side_pct/100,
        y2 + h*bottom_pct/100)


def extend_car_box(box, img_shape, side_pct=10, top_pct=18, bottom_pct=22):
    H, W = img_shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return [max(0, x1 - w*side_pct/100), max(0, y1 - h*top_pct/100),
            min(W, x2 + w*side_pct/100), min(H, y2 + h*bottom_pct/100)]


# ══════════════════════════════════════════════════════════════════════════════
# IMPROVED PLATE PRE-PROCESSING + OCR
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_plate(crop: np.ndarray) -> list:
    """
    Generate multiple pre-processed variants of a plate crop for best OCR results.

    Variants:
      1. CLAHE + Otsu binarisation         (clean / even lighting)
      2. Inverted Otsu                      (dark-background / white-text plates)
      3. Adaptive threshold (Gaussian)      (uneven illumination / shadows)
      4. Sharpen → CLAHE → Otsu            (motion-blur recovery)
      5. Morphological dilation of (1)      (thin / broken strokes)
      6. Deskew → CLAHE → Otsu             (slight camera angle, if detected)
      7. Bilateral filter → CLAHE → Otsu   (noise removal while keeping edges)
      8. Gamma-corrected (bright) → Otsu   (under-exposed plates)
      9. Gamma-corrected (dark)  → Otsu    (over-exposed / washed-out plates)
    """
    if crop is None or crop.size == 0:
        return []

    # ── Upscale so OCR has enough pixels ──────────────────────────────────
    h, w  = crop.shape[:2]
    # Target at least 48 px tall; scale up proportionally; cap at ×8
    scale = min(8, max(1, int(np.ceil(48 / max(h, 1)))))
    # Also ensure width is at least 200 px
    if w * scale < 200:
        scale = min(8, max(scale, int(np.ceil(200 / max(w, 1)))))
    big   = cv2.resize(crop, (w * scale, h * scale),
                       interpolation=cv2.INTER_CUBIC)

    gray  = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray  = clahe.apply(gray)

    variants = []

    # 1. CLAHE + Otsu
    _, otsu = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    # 2. Inverted Otsu  (white-on-dark plates)
    variants.append(cv2.bitwise_not(otsu))

    # 3. Adaptive threshold  (uneven lighting)
    ada = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10)
    variants.append(ada)

    # 4. Sharpen → Otsu  (motion-blur)
    k_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = np.clip(cv2.filter2D(gray, -1, k_sharp), 0, 255).astype(np.uint8)
    _, sharp_otsu = cv2.threshold(sharpened, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(sharp_otsu)

    # 5. Dilate  (thicken broken strokes)
    k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    variants.append(cv2.dilate(otsu, k_dil, iterations=1))

    # 6. Bilateral filter → CLAHE → Otsu  (noise-robust)
    try:
        bilat = cv2.bilateralFilter(gray, 9, 75, 75)
        bilat = clahe.apply(bilat)
        _, bilat_otsu = cv2.threshold(bilat, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(bilat_otsu)
    except Exception:
        pass

    # 7. Gamma bright (γ = 0.5 → brighten dark plate)
    try:
        gamma_bright = np.power(gray / 255.0, 0.5) * 255
        gamma_bright = gamma_bright.astype(np.uint8)
        _, gb_otsu = cv2.threshold(gamma_bright, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(gb_otsu)
    except Exception:
        pass

    # 8. Gamma dark  (γ = 2.0 → darken washed-out plate)
    try:
        gamma_dark = np.power(gray / 255.0, 2.0) * 255
        gamma_dark = gamma_dark.astype(np.uint8)
        _, gd_otsu = cv2.threshold(gamma_dark, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(gd_otsu)
    except Exception:
        pass

    # 9. Deskew if dominant angle > 1°
    try:
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40,
                                minLineLength=gray.shape[1] // 4,
                                maxLineGap=10)
        if lines is not None:
            angles = []
            for ln in lines:
                x1_, y1_, x2_, y2_ = ln[0]
                ang = np.degrees(np.arctan2(y2_ - y1_, x2_ - x1_))
                if abs(ang) < 30:
                    angles.append(ang)
            if angles:
                median_angle = float(np.median(angles))
                if abs(median_angle) > 1.0:
                    cx, cy = gray.shape[1] // 2, gray.shape[0] // 2
                    M = cv2.getRotationMatrix2D((cx, cy), median_angle, 1.0)
                    deskewed = cv2.warpAffine(
                        gray, M, (gray.shape[1], gray.shape[0]),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE)
                    deskewed = clahe.apply(deskewed)
                    _, dsk_otsu = cv2.threshold(deskewed, 0, 255,
                                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    variants.append(dsk_otsu)
    except Exception:
        pass

    return variants


def clean_plate_text(raw: str) -> str:
    txt    = re.sub(r"[^A-Z0-9\- ]", "", raw.upper()).strip()
    txt    = re.sub(r"\s+", " ", txt)
    tokens = [t for t in txt.split() if len(t) >= 1]
    return " ".join(tokens) if tokens else "UNREADABLE"


def ocr_plate(reader_tuple, crop: np.ndarray):
    """
    Run both EasyOCR and RapidOCR (if available) over all pre-processed plate
    variants. Returns (plate_text, ocr_confidence) for the best result across
    both engines.

    reader_tuple: (easyocr.Reader, RapidOCR | None)
    """
    if crop is None or crop.size == 0:
        return "UNREADABLE", 0.0

    reader, rapid = reader_tuple

    variants = preprocess_plate(crop)
    if not variants:
        return "UNREADABLE", 0.0

    ALLOWLIST  = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- "
    best_text  = "UNREADABLE"
    best_conf  = 0.0

    # ── EasyOCR pass ──────────────────────────────────────────────────────
    for v_img in variants:
        try:
            res = reader.readtext(
                v_img, detail=1, paragraph=False,
                allowlist=ALLOWLIST,
                text_threshold=0.45,
                low_text=0.25,
                link_threshold=0.4)
        except Exception:
            continue

        if not res:
            continue

        res_sorted = sorted(res, key=lambda r: r[0][0][0])
        text  = " ".join(r[1] for r in res_sorted)
        conf  = float(np.mean([r[2] for r in res_sorted]))
        clean = clean_plate_text(text)

        if clean and clean != "UNREADABLE" and conf > best_conf:
            best_conf = conf
            best_text = clean

    # ── EasyOCR raw-colour fallback ────────────────────────────────────────
    if best_text == "UNREADABLE":
        try:
            res = reader.readtext(crop, detail=1, paragraph=False)
            if res:
                res_s = sorted(res, key=lambda r: r[0][0][0])
                t = clean_plate_text(" ".join(r[1] for r in res_s))
                c = float(np.mean([r[2] for r in res_s]))
                if t and t != "UNREADABLE":
                    best_text, best_conf = t, c
        except Exception:
            pass

    # ── RapidOCR pass (secondary engine) ──────────────────────────────────
    if rapid is not None:
        for v_img in ([crop] + variants):
            try:
                # RapidOCR accepts numpy arrays (BGR or gray)
                result, _ = rapid(v_img)
                if result:
                    texts = [r[1] for r in result]
                    confs = [float(r[2]) for r in result]
                    combined = clean_plate_text(" ".join(texts))
                    avg_conf = float(np.mean(confs)) if confs else 0.0
                    if combined and combined != "UNREADABLE" and avg_conf > best_conf:
                        best_conf = avg_conf
                        best_text = combined
            except Exception:
                continue

    return best_text, round(best_conf, 3)


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING + DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def conf_badge_class(val):
    if val >= 0.65: return "cb-grn"
    if val >= 0.40: return "cb-acc"
    return "cb-red"


def draw_boxes(img: np.ndarray, detections: list, color_map: dict) -> np.ndarray:
    out = img.copy()
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        color = color_map.get(d["class"], (180, 180, 180))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d['class']}  {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
        cv2.putText(out, label, (x1+3, y1-4),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 1)
    return out


def np_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def upscale_plate_for_display(crop_bgr: np.ndarray,
                               target_h: int = 120):
    """Upscale tiny plate crops so they are clearly visible on-screen.
    Accepts BGR (3-ch), grayscale (2-D), or any numpy image."""
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return crop_bgr
    scale  = min(8, max(1, int(np.ceil(target_h / h))))
    interp = cv2.INTER_LANCZOS4 if scale >= 4 else cv2.INTER_CUBIC
    return cv2.resize(crop_bgr, (w*scale, h*scale), interpolation=interp)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL + OCR LOADING (cached across reruns)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_models_cached():
    from ultralytics import YOLO
    import os, glob

    def find_model(pattern_candidates):
        """Find the first existing file matching any of the given glob patterns."""
        for pat in pattern_candidates:
            hits = glob.glob(pat)
            if hits:
                return hits[0]
        return pattern_candidates[-1]   # fallback (will raise clear error)

    return {
        "vehicle":  YOLO(find_model([
            "model1_vehicle_detection (1).pt",
            "model1_vehicle_detection.pt",
        ])),
        "seatbelt": YOLO(find_model([
            "model2_seatbelt.pt",
        ])),
        "helmet":   YOLO(find_model([
            "model3_helmet (1).pt",
            "model3_helmet.pt",
        ])),
        "plate":    YOLO(find_model([
            "model4_license_plate.pt",
        ])),
    }


@st.cache_resource(show_spinner=False)
def load_ocr_cached():
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False)
    # Try to also load RapidOCR as secondary engine
    try:
        from rapidocr_onnxruntime import RapidOCR
        rapid = RapidOCR()
    except Exception:
        rapid = None
    return reader, rapid


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(img_bgr, models, reader_tuple, conf, moto_ext_pct, log_cb):
    violations     = []
    clean_vehicles = []
    anno_boxes     = []

    CAR_CLASS   = "Car"
    MOTO_CLASS  = "Motorbike"
    HELMET_YES  = "With Helmet"
    HELMET_NO   = "Without Helmet"
    PLATE_CLASS = "NumberPlate"

    # ── Step 1 : vehicle detection ────────────────────────────────────────
    log_cb("Running vehicle detection (model1)…", 10)
    veh_dets = run_model(models["vehicle"], img_bgr, conf["vehicle"])
    motos = [d for d in veh_dets if d["class"] == MOTO_CLASS]
    cars  = [d for d in veh_dets if d["class"] == CAR_CLASS]
    log_cb(f"Detected → {len(motos)} motorbike(s), {len(cars)} car(s)", 20)

    # ── Step 2a : motorbikes ──────────────────────────────────────────────
    for idx, moto in enumerate(motos):
        log_cb(f"Motorbike {idx+1}/{len(motos)} → helmet check…", 30)
        crop    = extend_moto_crop(img_bgr, moto["box"],
                                   top_ext_pct=moto_ext_pct,
                                   side_pct=8, bottom_pct=10)
        ext_box = extend_moto_box(moto["box"], img_bgr.shape,
                                  top_ext_pct=moto_ext_pct,
                                  side_pct=8, bottom_pct=10)
        if crop.size == 0:
            continue

        h_dets    = run_model(models["helmet"], crop, conf["helmet"])
        h_classes = [d["class"] for d in h_dets]
        has_helmet = HELMET_YES in h_classes
        no_helmet  = HELMET_NO  in h_classes

        base = {"type": "Motorbike", "veh_conf": moto["conf"], "box": moto["box"]}

        if no_helmet:
            log_cb("  ⚠ Without Helmet → plate detection…", 50)
            p_dets = run_model(models["plate"], crop, conf["plate"])
            plates = [d for d in p_dets if d["class"] == PLATE_CLASS]

            plate_crop_bgr, plate_model_conf = None, 0.0
            if plates:
                best = max(plates, key=lambda d: d["conf"])
                plate_crop_bgr   = safe_crop(crop, *best["box"])
                plate_model_conf = best["conf"]

            plate_text, ocr_conf = ocr_plate(reader_tuple, plate_crop_bgr)
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
            anno_boxes.append({"class": "NO HELMET",
                                "conf": moto["conf"], "box": ext_box})

        elif has_helmet:
            clean_vehicles.append({**base, "status": "Helmet Present"})
            anno_boxes.append({"class": "HELMET OK",
                                "conf": moto["conf"], "box": ext_box})
        else:
            clean_vehicles.append({**base, "status": "Helmet Uncertain"})
            anno_boxes.append({"class": "UNCERTAIN",
                                "conf": moto["conf"], "box": ext_box})

    # ── Step 2b : cars ────────────────────────────────────────────────────
    for idx, car in enumerate(cars):
        log_cb(f"Car {idx+1}/{len(cars)} → seatbelt check…", 60)
        crop    = extend_car_crop(img_bgr, car["box"],
                                  side_pct=10, top_pct=18, bottom_pct=22)
        ext_box = extend_car_box(car["box"], img_bgr.shape,
                                 side_pct=10, top_pct=18, bottom_pct=22)
        if crop.size == 0:
            continue

        sb_dets    = run_model(models["seatbelt"], crop, conf["seatbelt"])
        sb_classes = [d["class"] for d in sb_dets]
        has_belt = "Seatbelt"   in sb_classes
        no_belt  = "NoSeatbelt" in sb_classes

        base = {"type": "Car", "veh_conf": car["conf"], "box": car["box"]}

        if no_belt:
            log_cb("  ⚠ NoSeatbelt → plate detection…", 75)
            p_dets = run_model(models["plate"], crop, conf["plate"])
            plates = [d for d in p_dets if d["class"] == PLATE_CLASS]

            plate_crop_bgr, plate_model_conf = None, 0.0
            if plates:
                best = max(plates, key=lambda d: d["conf"])
                plate_crop_bgr   = safe_crop(crop, *best["box"])
                plate_model_conf = best["conf"]

            plate_text, ocr_conf = ocr_plate(reader_tuple, plate_crop_bgr)
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
            anno_boxes.append({"class": "NO SEATBELT",
                                "conf": car["conf"], "box": ext_box})

        elif has_belt:
            clean_vehicles.append({**base, "status": "Seatbelt Present"})
            anno_boxes.append({"class": "SEATBELT OK",
                                "conf": car["conf"], "box": ext_box})
        else:
            clean_vehicles.append({**base, "status": "Seatbelt Uncertain"})
            anno_boxes.append({"class": "UNCERTAIN",
                                "conf": car["conf"], "box": ext_box})

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

    st.markdown('<div class="sb-section">Confidence Thresholds</div>',
                unsafe_allow_html=True)
    c_v = st.slider("Vehicle (model1)",  0.20, 0.90, 0.40, 0.05)
    c_h = st.slider("Helmet  (model3)",  0.20, 0.90, 0.40, 0.05)
    c_s = st.slider("Seatbelt (model2)", 0.20, 0.90, 0.40, 0.05)
    c_p = st.slider("Plate   (model4)",  0.10, 0.90, 0.25, 0.05)

    st.markdown('<div class="sb-section">Motorcycle Crop</div>',
                unsafe_allow_html=True)
    moto_ext = st.slider(
        "Height extension ABOVE box (%)", 20, 150, 60, 5,
        help="Extends the motorbike crop upward to include the rider's helmet."
    )

    st.markdown('<div class="sb-section">Loaded Models</div>',
                unsafe_allow_html=True)
    for dot, name, classes in [
        ("#7ee787", "model1_vehicle_detection (1)",
         "Car · Motorbike  (17-class, others filtered)"),
        ("#f0a500", "model3_helmet (1)",
         "With Helmet · Without Helmet"),
        ("#79c0ff", "model2_seatbelt",
         "Seatbelt · NoSeatbelt"),
        ("#d2a8ff", "model4_license_plate",
         "NumberPlate"),
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
# HERO + UPLOAD
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <div class="hero-title">🚦 TrafficGuard AI</div>
  <div class="hero-sub">AUTOMATED TRAFFIC VIOLATION DETECTION SYSTEM · EE655</div>
  <div class="hero-pip">
    <span class="pip pip-v">MODEL 1 · VEHICLE DETECTION</span>
    <span class="pip pip-h">MODEL 3 · HELMET</span>
    <span class="pip pip-s">MODEL 2 · SEATBELT</span>
    <span class="pip pip-p">MODEL 4 · NUMBER PLATE</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="slabel">📤 Upload Traffic Image</div>',
            unsafe_allow_html=True)
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

file_bytes = np.frombuffer(uploaded.read(), np.uint8)
img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
H, W       = img_bgr.shape[:2]

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

if not run_btn:
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

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

log_cb("Loading YOLO models…", 3)
try:
    models = load_models_cached()
    log_cb("Loading OCR engines (EasyOCR + RapidOCR)…", 6)
    reader_tuple = load_ocr_cached()
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.markdown("""
    <div style="background:#161b22;border:1px solid #ff444440;border-radius:8px;
    padding:1rem 1.4rem;font-family:'Share Tech Mono',monospace;font-size:0.7rem;
    color:#7d8590;line-height:2.2;">
    <b style="color:#ff4444">LIKELY CAUSE: .pt files corrupted by Git CRLF conversion.</b><br>
    Add to your repo root a file called <b style="color:#e6edf3">.gitattributes</b>
    containing:<br>
    <span style="color:#f0a500">&nbsp;&nbsp;*.pt binary</span><br><br>
    Then re-add and re-commit all .pt files. Alternatively use Git LFS:<br>
    <span style="color:#f0a500">
    &nbsp;&nbsp;git lfs track "*.pt"<br>
    &nbsp;&nbsp;git add .gitattributes *.pt<br>
    &nbsp;&nbsp;git commit -m "fix: track models as LFS binary"
    </span>
    </div>""", unsafe_allow_html=True)
    st.stop()

t0 = time.time()
violations, clean_vehicles, annotated = run_pipeline(
    img_bgr, models, reader_tuple, conf_dict, moto_ext, log_cb)
elapsed = round(time.time() - t0, 2)

status_box.empty()
progress.empty()

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown('<div class="slabel">📊 Analysis Summary</div>', unsafe_allow_html=True)

n_total = len(violations) + len(clean_vehicles)
n_nh    = sum(1 for v in violations if "Helmet"   in v["violation"])
n_ns    = sum(1 for v in violations if "Seatbelt" in v["violation"])

st.markdown(f"""
<div class="mrow">
  <div class="mcard mc-acc"><div class="mcard-num">{n_total}</div>
    <div class="mcard-lbl">Vehicles Detected</div></div>
  <div class="mcard mc-red"><div class="mcard-num">{len(violations)}</div>
    <div class="mcard-lbl">Violations Found</div></div>
  <div class="mcard mc-grn"><div class="mcard-num">{len(clean_vehicles)}</div>
    <div class="mcard-lbl">Compliant</div></div>
  <div class="mcard mc-blue"><div class="mcard-num">{n_nh}</div>
    <div class="mcard-lbl">No Helmet</div></div>
  <div class="mcard mc-red"><div class="mcard-num">{n_ns}</div>
    <div class="mcard-lbl">No Seatbelt</div></div>
  <div class="mcard mc-acc"><div class="mcard-num">{elapsed}s</div>
    <div class="mcard-lbl">Inference Time</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="slabel">🖼 Annotated Detection Result</div>',
            unsafe_allow_html=True)
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
        icon  = "🏍️" if v["type"] == "Motorbike" else "🚗"
        vc, pc, oc = v["viol_conf"], v["plate_model_conf"], v["ocr_conf"]
        has_plate      = v["plate_crop_bgr"] is not None and v["plate_crop_bgr"].size > 0
        plate_readable = has_plate and v["plate_text"] not in ("UNREADABLE", "")

        st.markdown(f"""
        <div class="vcard">
          <div class="vcard-head">
            {icon} VIOLATION #{i+1} — {v['violation'].upper()}
            <span class="conf-badge {conf_badge_class(vc)}">VIOL CONF {vc:.3f}</span>
            <span class="conf-badge cb-acc">VEH CONF {v['veh_conf']:.3f}</span>
          </div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;
          color:#7d8590;letter-spacing:1px">TYPE: {v['type'].upper()}</div>
        </div>
        """, unsafe_allow_html=True)

        # Vehicle crop — capped so it never overflows the screen
        st.markdown('<div class="slabel" style="margin-top:0.6rem">🚗 Vehicle Crop</div>',
                    unsafe_allow_html=True)
        crop_vis = v["crop_bgr"].copy()
        ch, cw   = crop_vis.shape[:2]
        cv2.rectangle(crop_vis, (2, 2), (cw-2, ch-2), (0, 200, 255), 2)
        MAX_H, MAX_W = 320, 560
        scale = min(MAX_H / max(ch, 1), MAX_W / max(cw, 1), 1.0)
        if scale < 1.0:
            crop_vis = cv2.resize(crop_vis,
                                  (int(cw * scale), int(ch * scale)),
                                  interpolation=cv2.INTER_AREA)
        st.image(np_to_pil(crop_vis))

        # ── License Plate section (always shown) ─────────────────────────────
        st.markdown(
            '<div class="slabel" style="margin-top:0.8rem">'
            '🔍 License Plate &amp; OCR</div>',
            unsafe_allow_html=True)

        # Build base64 plate image for embedding in HTML card
        def plate_to_b64(crop_bgr):
            import base64, io
            disp = upscale_plate_for_display(crop_bgr.copy(), target_h=80)
            pb_h, pb_w = disp.shape[:2]
            cv2.rectangle(disp, (0, 0), (pb_w-1, pb_h-1), (0, 200, 255), 3)
            pil_img = np_to_pil(disp)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()

        if has_plate:
            quality_label = "READABLE ✅" if plate_readable else "LOW QUALITY ⚠"
            quality_color = "#00cc66" if plate_readable else "#f0a500"
            b64_img = plate_to_b64(v["plate_crop_bgr"])
            plate_img_html = f'<img src="data:image/png;base64,{b64_img}" style="max-height:80px;max-width:100%;border-radius:4px;border:2px solid #00c8ff44;display:block;margin:0 auto;">'
        else:
            quality_label = "NOT DETECTED"
            quality_color = "#7d8590"
            plate_img_html = '<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.65rem;color:#7d8590;text-align:center;padding:1rem 0">NO CROP AVAILABLE</div>'

        if has_plate and plate_readable:
            st.markdown(f"""
            <div style="background:#0d1117;border:2px solid #f0a500;border-radius:12px;
            padding:1.4rem 1.6rem;margin-top:0.4rem;display:flex;gap:1.4rem;align-items:center;flex-wrap:wrap">
              <div style="flex:0 0 auto;min-width:120px;text-align:center">
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;
                color:#7d8590;letter-spacing:2px;margin-bottom:6px">PLATE CROP</div>
                {plate_img_html}
                <div style="margin-top:6px">
                  <span class="conf-badge {conf_badge_class(pc)}">MODEL {pc:.3f}</span>
                </div>
              </div>
              <div style="flex:1;min-width:160px">
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.58rem;
                color:#7d8590;letter-spacing:2px;margin-bottom:0.5rem">EXTRACTED PLATE NUMBER</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:2rem;
                font-weight:700;color:#f0a500;letter-spacing:6px;word-break:break-all;
                text-shadow:0 0 20px #f0a50066">
                  {v['plate_text']}
                </div>
                <div style="margin-top:0.7rem;display:flex;gap:8px;flex-wrap:wrap;align-items:center">
                  <span style="font-family:'Share Tech Mono',monospace;font-size:0.58rem;
                  color:{quality_color};letter-spacing:1px">{quality_label}</span>
                  <span class="conf-badge {conf_badge_class(oc)}">OCR CONF {oc:.3f}</span>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        elif has_plate and not plate_readable:
            st.markdown(f"""
            <div style="background:#0d1117;border:2px solid #ff4444;border-radius:12px;
            padding:1.4rem 1.6rem;margin-top:0.4rem;display:flex;gap:1.4rem;align-items:center;flex-wrap:wrap">
              <div style="flex:0 0 auto;min-width:120px;text-align:center">
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;
                color:#7d8590;letter-spacing:2px;margin-bottom:6px">PLATE CROP</div>
                {plate_img_html}
                <div style="margin-top:6px">
                  <span class="conf-badge {conf_badge_class(pc)}">MODEL {pc:.3f}</span>
                </div>
              </div>
              <div style="flex:1;min-width:160px">
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.58rem;
                color:#7d8590;letter-spacing:2px;margin-bottom:0.5rem">PLATE DETECTED — OCR FAILED</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:1.8rem;
                font-weight:700;color:#ff4444;letter-spacing:4px">UNREADABLE</div>
                <div style="margin-top:0.7rem;display:flex;gap:8px;flex-wrap:wrap;align-items:center">
                  <span style="font-family:'Share Tech Mono',monospace;font-size:0.58rem;
                  color:{quality_color};letter-spacing:1px">{quality_label}</span>
                  <span class="conf-badge cb-red">OCR CONF {oc:.3f}</span>
                </div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;
                color:#7d8590;margin-top:0.5rem">Image too blurry / low-res for OCR</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:#0d1117;border:2px solid #21262d;border-radius:12px;
            padding:1.4rem 1.6rem;margin-top:0.4rem;display:flex;gap:1.4rem;align-items:center">
              <div style="flex:0 0 auto;min-width:120px;text-align:center">
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;
                color:#7d8590;letter-spacing:2px;margin-bottom:6px">PLATE CROP</div>
                <div style="background:#161b22;border:1px dashed #21262d;border-radius:4px;
                padding:1rem;font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#7d8590">
                  NO CROP
                </div>
              </div>
              <div style="flex:1;min-width:160px">
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.58rem;
                color:#7d8590;letter-spacing:2px;margin-bottom:0.5rem">NO PLATE DETECTED</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:1rem;color:#7d8590">—</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;
                color:#7d8590;margin-top:0.5rem">Try lowering the plate confidence threshold</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Preprocessed variants (collapsed expander) ────────────────────────
        if has_plate:
            with st.expander("🔬 Preprocessed variants sent to OCR", expanded=False):
                variant_labels = [
                    "CLAHE + Otsu", "Inverted Otsu", "Adaptive Thresh",
                    "Sharpen + Otsu", "Dilated", "Bilateral + Otsu",
                    "Gamma bright", "Gamma dark", "Deskewed",
                ]
                pp_variants = preprocess_plate(v["plate_crop_bgr"])
                cols_pp = st.columns(min(3, len(pp_variants)))
                for vi, vv in enumerate(pp_variants):
                    lbl = variant_labels[vi] if vi < len(variant_labels) else f"Variant {vi+1}"
                    vv_disp = upscale_plate_for_display(vv, target_h=60)
                    with cols_pp[vi % 3]:
                        st.image(vv_disp, caption=lbl, width=160)

        if i < len(violations) - 1:
            st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# COMPLIANT VEHICLES
# ══════════════════════════════════════════════════════════════════════════════

if clean_vehicles:
    st.markdown('<div class="slabel">✅ Compliant Vehicles</div>',
                unsafe_allow_html=True)
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
