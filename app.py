import os, io, re, time, pickle, base64, tempfile, traceback
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import rasterio

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024

ROW_START, ROW_END = 999, 2999
COL_START, COL_END = 999, 2999
PIXEL_KM2    = (30 ** 2) / 1e6
CLASS_LABELS = [1, 2, 3, 4]
CLASS_NAMES  = ["Water", "Vegetation", "Urban", "Desert"]
COLOR_MAP = {
    0: [0,   0,   0  ],
    1: [0,   77,  204],
    2: [26,  179, 26 ],
    3: [230, 26,  26 ],
    4: [242, 217, 26 ],
}
SCALE_NEEDED = {"SVM": True, "KNN": True, "Random": False, "Decision": False, "Naive": False}
MODELS_DIR   = "/content/drive/MyDrive/ECE435_Project/data/processed"
DEMO_FOLDER  = "/content/drive/MyDrive/ECE435_Project/data/raw"

def load_models():
    with open(os.path.join(MODELS_DIR, "best_model_name.txt")) as f:
        best_name = f.read().strip()
    model_key = best_name.split()[0].lower()
    with open(os.path.join(MODELS_DIR, f"model_{model_key}.pkl"), "rb") as f:
        clf = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return clf, scaler, best_name

def parse_mtl(mtl_text):
    def get(key):
        m = re.search(rf"{key}\s*=\s*([\d.E+\-]+)", mtl_text)
        if not m: raise KeyError(f"Key {key} not found in MTL")
        return float(m.group(1))
    sun_elev = get("SUN_ELEVATION")
    sin_elev = np.sin(np.deg2rad(sun_elev))
    coeffs = [(get(f"REFLECTANCE_MULT_BAND_{b}"), get(f"REFLECTANCE_ADD_BAND_{b}")) for b in range(1, 8)]
    return coeffs, sin_elev

def load_bands(band_files):
    def band_num(p):
        m = re.search(r"B(\d+)\.TIF$", os.path.basename(p), re.IGNORECASE)
        return int(m.group(1)) if m else 99
    bands = []
    for bf in sorted(band_files, key=band_num):
        with rasterio.open(bf) as src:
            arr = src.read(1).astype(np.float32)
            bands.append(arr[ROW_START:ROW_END, COL_START:COL_END])
    return np.stack(bands, axis=-1)

def calibrate(raw, coeffs, sin_elev):
    toa = np.zeros_like(raw)
    for i, (M, A) in enumerate(coeffs):
        toa[:,:,i] = np.clip((M * raw[:,:,i] + A) / sin_elev, 0, 1)
    return toa

def compute_indices(toa):
    Green, Red, NIR, SWIR1 = toa[:,:,2], toa[:,:,3], toa[:,:,4], toa[:,:,5]
    def sd(a, b): return np.where((a+b)==0, 0.0, (a-b)/(a+b+1e-10)).astype(np.float32)
    return np.concatenate([toa, sd(NIR,Red)[:,:,None], sd(Green,NIR)[:,:,None], sd(SWIR1,NIR)[:,:,None]], axis=-1)

def classify(stack, clf, scaler, best_name):
    H, W, C = stack.shape
    X = np.nan_to_num(stack.reshape(-1, C), nan=0.0, posinf=1.0, neginf=0.0)
    if SCALE_NEEDED.get(best_name.split()[0], False):
        Xs = np.empty_like(X)
        for i in range(0, len(X), 500000): Xs[i:i+500000] = scaler.transform(X[i:i+500000])
        X = Xs
    preds = np.zeros(len(X), dtype=np.uint8)
    for i in range(0, len(X), 200000): preds[i:i+200000] = clf.predict(X[i:i+200000])
    return preds.reshape(H, W)

def build_rgb(pred):
    rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
    for lbl, c in COLOR_MAP.items(): rgb[pred==lbl] = c
    return Image.fromarray(rgb)

def compute_stats(pred):
    total = pred.size
    return [{"label": l, "class": n, "pixels": int((pred==l).sum()),
             "area_km2": round(int((pred==l).sum()) * PIXEL_KM2, 2),
             "percent": round(int((pred==l).sum()) / total * 100, 2)}
            for l, n in zip(CLASS_LABELS, CLASS_NAMES)]

def img_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

with open("/content/index2.html") as f:
    HTML_PAGE = f.read()

@app.route("/")
def index(): return HTML_PAGE

@app.route("/check_models")
def check_models():
    try:
        _, _, name = load_models()
        return jsonify({"ready": True, "model": name})
    except Exception as e:
        return jsonify({"ready": False, "error": str(e)})

@app.route("/predict", methods=["POST"])
def predict():
    t0 = time.time()
    try:
        band_files, mtl_path = [], None
        tmpdir = tempfile.mkdtemp()
        for key, f in request.files.items():
            dest = os.path.join(tmpdir, f.filename)
            f.save(dest)
            if re.search(r"B[1-7]\.TIF$", f.filename, re.IGNORECASE): band_files.append(dest)
            elif "MTL" in f.filename.upper(): mtl_path = dest
        if not mtl_path: return jsonify({"error": "MTL.txt not found"}), 400
        if len(band_files) != 7: return jsonify({"error": f"Need 7 band files, got {len(band_files)}"}), 400
        raw = load_bands(band_files)
        with open(mtl_path) as f: mtl = f.read()
        coeffs, sin_elev = parse_mtl(mtl)
        toa   = calibrate(raw, coeffs, sin_elev); del raw
        stack = compute_indices(toa)
        clf, scaler, best_name = load_models()
        pred  = classify(stack, clf, scaler, best_name)
        pred[np.all(toa==0, axis=2)] = 0
        return jsonify({"status": "ok", "model": best_name,
                        "time_sec": round(time.time()-t0, 1),
                        "map_b64": img_b64(build_rgb(pred)),
                        "stats": compute_stats(pred)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/demo_predict")
def demo_predict():
    t0 = time.time()
    try:
        all_files  = os.listdir(DEMO_FOLDER)
        band_files = sorted(
            [os.path.join(DEMO_FOLDER, fn) for fn in all_files
             if re.search(r"B[1-7]\.TIF$", fn, re.IGNORECASE)],
            key=lambda p: int(re.search(r"B(\d+)\.TIF$", p, re.IGNORECASE).group(1))
        )
        mtl_paths = [os.path.join(DEMO_FOLDER, fn) for fn in all_files
                     if "MTL" in fn.upper() and fn.upper().endswith(".TXT")]

        if len(band_files) != 7:
            return jsonify({"error": f"Demo folder needs 7 band TIFs, found {len(band_files)}"}), 400
        if not mtl_paths:
            return jsonify({"error": "MTL.txt not found in demo folder"}), 400

        mtl_path = mtl_paths[0]
        raw      = load_bands(band_files)
        with open(mtl_path) as f: mtl = f.read()
        coeffs, sin_elev = parse_mtl(mtl)
        toa   = calibrate(raw, coeffs, sin_elev); del raw
        stack = compute_indices(toa)
        clf, scaler, best_name = load_models()
        pred  = classify(stack, clf, scaler, best_name)
        pred[np.all(toa == 0, axis=2)] = 0
        return jsonify({"status": "ok", "model": best_name,
                        "time_sec": round(time.time() - t0, 1),
                        "map_b64":  img_b64(build_rgb(pred)),
                        "stats":    compute_stats(pred)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
