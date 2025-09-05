import os
import re
from flask import Flask, render_template, request, redirect, url_for, flash
from gensim.downloader import BASE_DIR
from werkzeug.utils import secure_filename
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from numpy.linalg import norm
import PyPDF2

# Config
#
# resume-matcher/
# ├─ app.py
# ├─ requirements.txt
# ├─ cv_job_maching.model           # đặt file model bạn đã save ở đây
# ├─ uploads/
# │  ├─ jds/                        # JD đã upload sẽ được lưu tại đây
# │  └─ resumes/                    # Resume đã upload sẽ được lưu tại đây
# └─ templates/
#    ├─ index.html
#    └─ result.html
#
#

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
JD_DIR = os.path.join(UPLOAD_DIR, 'jds')
RESUME_DIR = os.path.join(UPLOAD_DIR, 'resumes')
MODEL_PATH = os.path.join(BASE_DIR, 'cv_job_maching.model')
ALLOWED_JD_EXT = {'txt', 'pdf'}
ALLOWED_RESUME_EXT = {'pdf', 'txt'}

os.makedirs(JD_DIR, exist_ok=True)
os.makedirs(RESUME_DIR, exist_ok=True)

# -----------------------------
# App & Model
# -----------------------------
# Start copy here
app = Flask(__name__)
app.secret_key = "replace-this-with-a-random-secret-key"  # để flash message

# Load Doc2Vec model once at startup
try:
    model = Doc2Vec.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"[ERROR] Cannot load model at {MODEL_PATH}: {e}")

# -----------------------------
# Helpers
# -----------------------------
def allowed_file(filename, allowed_exts):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_exts

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub('[^a-z]', ' ', text)  # keep only a-z
    text = re.sub(r'\d+', '', text)     # remove digits
    text = ' '.join(text.split())
    return text

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        content = []
        for i in range(len(reader.pages)):
            page = reader.pages[i]
            # page.extract_text() may return None for some PDFs
            txt = page.extract_text() or ""
            content.append(txt)
        return "\n".join(content)
    except Exception as e:
        print(f"[ERROR] Failed to read PDF {pdf_path}: {e}")
        return ""

def read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"[ERROR] Failed to read TXT {path}: {e}")
        return ""

def read_any_text(path: str) -> str:
    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(path)
    else:
        return read_text_file(path)

def infer_vector(text: str):
    if model is None:
        return None
    tokens = text.split()
    return model.infer_vector(tokens)

def cosine_similarity(vec1, vec2) -> float:
    a = np.array(vec1, dtype=float)
    b = np.array(vec2, dtype=float)
    denom = (norm(a) * norm(b))
    if denom == 0:
        return 0.0
    # same formula bạn dùng, nhân 100 để ra %
    return float(100.0 * np.dot(a, b) / denom)

def list_jds():
    files = []
    for fname in sorted(os.listdir(JD_DIR)):
        path = os.path.join(JD_DIR, fname)
        if os.path.isfile(path) and allowed_file(fname, ALLOWED_JD_EXT):
            files.append(fname)
    return files

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def index():
    jds = list_jds()
    model_loaded = (model is not None)
    return render_template("index.html", jds=jds, model_loaded=model_loaded)

@app.route("/upload_jd", methods=["POST"])
def upload_jd():
    if "jd_file" not in request.files:
        flash("No file part for JD.")
        return redirect(url_for("index"))
    file = request.files["jd_file"]
    if file.filename == "":
        flash("No JD selected.")
        return redirect(url_for("index"))
    if file and allowed_file(file.filename, ALLOWED_JD_EXT):
        filename = secure_filename(file.filename)
        save_path = os.path.join(JD_DIR, filename)
        file.save(save_path)
        flash(f"Uploaded JD: {filename}")
    else:
        flash("Invalid JD file type. Only .txt or .pdf are accepted.")
    return redirect(url_for("index"))

@app.route("/match", methods=["POST"])
def match():
    if model is None:
        flash("Model not loaded. Please ensure cv_job_maching.model is present.")
        return redirect(url_for("index"))

    selected_jd = request.form.get("selected_jd", "").strip()
    if not selected_jd:
        flash("Please select a JD.")
        return redirect(url_for("index"))

    # Read JD text
    jd_path = os.path.join(JD_DIR, selected_jd)
    if not os.path.exists(jd_path):
        flash("Selected JD not found.")
        return redirect(url_for("index"))

    jd_text = read_any_text(jd_path)
    jd_text_clean = preprocess_text(jd_text)

    # Handle resume file
    if "resume_file" not in request.files:
        flash("No resume file uploaded.")
        return redirect(url_for("index"))

    resume_file = request.files["resume_file"]
    if resume_file.filename == "":
        flash("No resume selected.")
        return redirect(url_for("index"))

    if not allowed_file(resume_file.filename, ALLOWED_RESUME_EXT):
        flash("Invalid resume file type. Only .pdf or .txt are accepted.")
        return redirect(url_for("index"))

    resume_name = secure_filename(resume_file.filename)
    resume_path = os.path.join(RESUME_DIR, resume_name)
    resume_file.save(resume_path)

    resume_text = read_any_text(resume_path)
    resume_text_clean = preprocess_text(resume_text)

    # Infer vectors & compute similarity
    v_jd = infer_vector(jd_text_clean)
    v_cv = infer_vector(resume_text_clean)

    if v_jd is None or v_cv is None:
        flash("Failed to infer vectors. Check model file.")
        return redirect(url_for("index"))

    score = cosine_similarity(v_cv, v_jd)
    score_rounded = round(score, 2)

    return render_template(
        "result.html",
        jd_filename=selected_jd,
        resume_filename=resume_name,
        score=score_rounded
    )

# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    # Chạy dev server
    app.run(host="0.0.0.0", port=5000, debug=True)