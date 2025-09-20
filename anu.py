# ---------------- Robust OMR Processing Notebook (Full, Auto-Grid) ----------------

# ---------------- Imports ----------------
import os, zipfile, tempfile, json, sqlite3
import cv2, imutils, numpy as np, pandas as pd
from imutils.perspective import four_point_transform
from PIL import Image
import fitz  # PyMuPDF
from tensorflow.keras.models import load_model
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

# ---------------- Config ----------------
CONFIG = {
    'BUBBLE_MODEL_PATH': 'bubble_cnn_model.h5',
    'OPTION_MAP': {'a':0,'b':1,'c':2,'d':3},
    'SUBJECT_RANGES': {'Python':(1,20),'EDA':(21,40),'SQL':(41,60),'PowerBI':(61,80),'Statistics':(81,100)},
    'FILL_RATIO_THRESHOLD':0.2,
    'PARTIAL_CREDIT': True,
    'DEBUG_DIR':'debug_output'
}

# ---------------- Answer Keys ----------------
# For brevity, only A key is included; add B as needed
ANSWER_KEY_A = {1: ['a'], 2: ['c'], 3: ['c'], 4: ['c'], 5: ['c'], 6: ['a'], 7: ['c'], 8: ['c'], 9: ['b'], 10: ['c'],
                11: ['a'], 12: ['a'], 13: ['d'], 14: ['a'], 15: ['b'], 16: ['a','b','c','d'], 17: ['c'], 18: ['d'], 19: ['a'], 20: ['b'],
                21: ['a'], 22: ['d'], 23: ['b'], 24: ['a'], 25: ['c'], 26: ['b'], 27: ['a'], 28: ['b'], 29: ['d'], 30: ['c'],
                31: ['c'], 32: ['a'], 33: ['b'], 34: ['c'], 35: ['a'], 36: ['b'], 37: ['d'], 38: ['b'], 39: ['a'], 40: ['b'],
                41: ['c'], 42: ['c'], 43: ['c'], 44: ['b'], 45: ['b'], 46: ['a'], 47: ['c'], 48: ['b'], 49: ['d'], 50: ['a'],
                51: ['c'], 52: ['b'], 53: ['c'], 54: ['c'], 55: ['a'], 56: ['b'], 57: ['b'], 58: ['a'], 59: ['a','b'], 60: ['b'],
                61: ['b'], 62: ['c'], 63: ['a'], 64: ['b'], 65: ['c'], 66: ['b'], 67: ['b'], 68: ['c'], 69: ['c'], 70: ['b'],
                71: ['b'], 72: ['b'], 73: ['d'], 74: ['b'], 75: ['a'], 76: ['b'], 77: ['b'], 78: ['b'], 79: ['b'], 80: ['b'],
                81: ['a'], 82: ['b'], 83: ['c'], 84: ['b'], 85: ['c'], 86: ['b'], 87: ['b'], 88: ['b'], 89: ['a'], 90: ['b'],
                91: ['c'], 92: ['b'], 93: ['c'], 94: ['b'], 95: ['b'], 96: ['b'], 97: ['c'], 98: ['a'], 99: ['b'], 100: ['c']}
ANSWER_KEY_B = ANSWER_KEY_A  # Replace with real B key

# ---------------- Database ----------------
def init_database(db_name='omr_results.db'):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS results
                 (student_id TEXT PRIMARY KEY, set_version TEXT,
                  python INT, eda INT, sql INT, powerbi INT, stats INT, total INT)''')
    conn.commit()
    return conn

def update_database(conn, student_result):
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO results
                 (student_id,set_version,python,eda,sql,powerbi,stats,total)
                 VALUES (?,?,?,?,?,?,?,?)''',
              (student_result['student_id'], student_result['set_version'],
               student_result['Python'], student_result['EDA'], student_result['SQL'],
               student_result['PowerBI'], student_result['Statistics'], student_result['total']))
    conn.commit()

# ---------------- Load Bubble CNN ----------------
def load_bubble_model():
    if os.path.exists(CONFIG['BUBBLE_MODEL_PATH']):
        try:
            model = load_model(CONFIG['BUBBLE_MODEL_PATH'])
            print("Bubble CNN loaded successfully.")
            return model
        except Exception as e:
            print(f"Failed to load CNN model: {e}. Using fill-ratio method.")
    print("Bubble CNN not found. Using fill-ratio method.")
    return None

# ---------------- Image Preprocessing & Auto Grid ----------------
def preprocess_image_grid(image_path, debug=False):
    try:
        # Load image
        if image_path.lower().endswith('.pdf'):
            doc = fitz.open(image_path)
            page = doc.load_page(0)
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img = np.array(img)
        else:
            img = cv2.imread(image_path)
        if img is None:
            return None, None, None

        img = imutils.resize(img, width=1000)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Detect outer contour
        cnts = cv2.findContours(cv2.Canny(thresh,50,150),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        docCnt = None
        if cnts:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02*peri, True)
                if len(approx) >=4:
                    docCnt = approx
                    break

        warped = four_point_transform(gray, docCnt.reshape(4,2)) if docCnt is not None and len(docCnt)==4 else gray.copy()
        h, w = warped.shape
        thresh = cv2.adaptiveThreshold(warped, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Create uniform bubble grid 100x4
        n_questions = 100
        n_options = 4
        row_height = h / n_questions
        col_width = w / n_options
        bubble_h = int(row_height*0.6)
        bubble_w = int(col_width*0.6)
        grid = []
        for q in range(n_questions):
            row_start = int(q * row_height)
            for opt in range(n_options):
                col_start = int(opt * col_width + col_width*0.1)
                grid.append((col_start, row_start, bubble_w, bubble_h))

        if debug:
            os.makedirs(CONFIG['DEBUG_DIR'], exist_ok=True)
            cv2.imwrite(f"{CONFIG['DEBUG_DIR']}/warped_{os.path.basename(image_path)}.png", warped)
            cv2.imwrite(f"{CONFIG['DEBUG_DIR']}/thresh_{os.path.basename(image_path)}.png", thresh)

        return warped, thresh, grid

    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return None, None, None

# ---------------- Bubble Detection ----------------
def preprocess_bubble(bubble_img):
    gray = cv2.cvtColor(bubble_img, cv2.COLOR_BGR2GRAY) if len(bubble_img.shape)==3 else bubble_img
    resized = cv2.resize(gray, (28,28))
    normalized = resized/255.0
    return normalized.reshape(28,28,1)

def is_bubble_marked(bubble_img, model=None):
    bubble_input = preprocess_bubble(bubble_img)
    if model:
        pred = model.predict(bubble_input.reshape(1,28,28,1), verbose=0)
        return np.argmax(pred[0])==1
    fill_ratio = cv2.countNonZero(bubble_img)/float(max(1,bubble_img.size))
    return fill_ratio>CONFIG['FILL_RATIO_THRESHOLD']

# ---------------- Extract Responses from Grid ----------------
def extract_responses_grid(thresh, grid, model=None, debug=False):
    responses = {}
    bubble_keys = list(CONFIG['OPTION_MAP'].keys())

    for q in range(100):
        marked = []
        for opt in range(4):
            x, y, w, h = grid[q*4 + opt]
            bubble_img = thresh[y:y+h, x:x+w]
            if bubble_img.size==0: continue
            if is_bubble_marked(bubble_img, model):
                marked.append(bubble_keys[opt])
            if debug:
                os.makedirs(CONFIG['DEBUG_DIR'], exist_ok=True)
                cv2.imwrite(f"{CONFIG['DEBUG_DIR']}/Q{q+1}_opt{opt+1}.png", bubble_img)
        responses[q+1] = sorted(marked) if marked else ['unmarked']
    return responses

# ---------------- Scoring ----------------
def score_responses(responses, set_version):
    key = ANSWER_KEY_A if set_version=='A' else ANSWER_KEY_B
    subject_scores = {subj:0 for subj in CONFIG['SUBJECT_RANGES']}
    total = 0.0
    per_question_scores = {}

    for q in range(1,101):
        student_ans = set(responses.get(q, []))
        student_ans.discard('unmarked')
        correct_ans = set(key.get(q, []))

        if not student_ans:
            increment = 0.0
        elif student_ans == correct_ans:
            increment = 1.0
        elif CONFIG['PARTIAL_CREDIT'] and student_ans & correct_ans:
            increment = len(student_ans & correct_ans)/len(correct_ans)
        else:
            increment = 0.0

        per_question_scores[q] = round(increment,2)
        total += increment

        for subj, (start,end) in CONFIG['SUBJECT_RANGES'].items():
            if start <= q <= end:
                subject_scores[subj] += increment
                break

    subject_scores = {k: round(v,2) for k,v in subject_scores.items()}
    total = round(total,2)
    return subject_scores, total, per_question_scores

# ---------------- Process Single Sheet ----------------
def process_omr_sheet(image_path, student_id, model=None, set_version='A'):
    warped, thresh, grid = preprocess_image_grid(image_path)
    if warped is None or thresh is None or grid is None:
        print(f"Failed to preprocess: {image_path}")
        return None

    responses = extract_responses_grid(thresh, grid, model)
    scores, total, per_question_scores = score_responses(responses, set_version)

    return {
        'student_id': student_id,
        'set_version': set_version,
        'Python': scores['Python'],
        'EDA': scores['EDA'],
        'SQL': scores['SQL'],
        'PowerBI': scores['PowerBI'],
        'Statistics': scores['Statistics'],
        'total': total,
        'responses': json.dumps(responses),
        'per_question_scores': json.dumps(per_question_scores),
        'warped': warped,
        'thresh': thresh
    }

# ---------------- Widget Interface ----------------
upload_btn = widgets.FileUpload(accept='.jpg,.png,.pdf,.zip', multiple=True)
student_id_text = widgets.Text(value='student', description='Student ID:')
set_version_dropdown = widgets.Dropdown(options=[None,'A','B'], description='Set:')
process_btn = widgets.Button(description='Process Sheets', button_style='success')
output_area = widgets.Output()

model = load_bubble_model()
conn = init_database()

def handle_processing(b):
    with output_area:
        clear_output()
        image_paths = []

        # Save uploaded files
        for uploaded_file in upload_btn.value:
            fname = uploaded_file['name']
            data = uploaded_file['content']
            os.makedirs('uploads', exist_ok=True)
            path = os.path.join('uploads', fname)
            with open(path, 'wb') as f: f.write(data)

            if path.lower().endswith('.zip'):
                tmp_dir = tempfile.mkdtemp()
                with zipfile.ZipFile(path,'r') as z: z.extractall(tmp_dir)
                for root,_,files in os.walk(tmp_dir):
                    for f in files:
                        if f.lower().endswith(('.jpg','.jpeg','.png','.pdf')):
                            image_paths.append(os.path.join(root,f))
            else:
                image_paths.append(path)

        results_list = []

        for idx, path in enumerate(image_paths):
            sid = f"{student_id_text.value}_{idx}" if len(image_paths)>1 else student_id_text.value
            res = process_omr_sheet(path, sid, model, set_version_dropdown.value or 'A')
            if res is None: 
                print(f"Skipped {path}"); continue
            update_database(conn, res)
            print(f"Processed {sid}: Total Score = {res['total']}")

            plt.figure(figsize=(6,6)); plt.imshow(res['warped'], cmap='gray'); plt.axis('off'); plt.title(f"{sid} Warped"); plt.show()
            plt.figure(figsize=(6,6)); plt.imshow(res['thresh'], cmap='gray'); plt.axis('off'); plt.title(f"{sid} Threshold"); plt.show()
            results_list.append(res)

        if results_list:
            df = pd.DataFrame(results_list)
            df_to_display = df.drop(columns=['warped','thresh'])
            per_q_df = pd.json_normalize(df['per_question_scores'].apply(json.loads))
            per_q_df = per_q_df.add_prefix('Q')
            final_df = pd.concat([df_to_display, per_q_df], axis=1)
            display(final_df)
            final_df.to_csv('omr_results_detailed.csv', index=False)
            print("All results saved to omr_results_detailed.csv and database.")

process_btn.on_click(handle_processing)
display(widgets.VBox([upload_btn, student_id_text, set_version_dropdown, process_btn, output_area]))
