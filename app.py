from flask import Flask, render_template, request, jsonify, send_file, abort
from audio import Sound
from features_extract import Extract
import sqlite3, numpy as np, os, time

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUERY_FOLDER = os.path.join(BASE_DIR, 'query')
os.makedirs(QUERY_FOLDER, exist_ok=True)

def load_sounds():
    conn = sqlite3.connect(os.path.join(BASE_DIR, 'database.db'))
    c = conn.cursor()
    c.execute("SELECT path, features, shape FROM sounds")
    rows = c.fetchall()
    sounds = []
    for row in rows:
        shape = eval(row[2])
        features = np.frombuffer(row[1], dtype=np.float64).reshape(shape)
        sounds.append(Sound(features=features, path=row[0]))
    conn.close()
    return sounds

def search_songs(q_sound, sounds, k=5):
    similarity = {}
    step_data = {}
    for sound in sounds:
        feaW = q_sound.features
        feaS = sound.features
        if len(feaW) > len(feaS):
            feaW, feaS = feaS, feaW
        overlap = max(1, int(0.2 * len(feaW)))
        n_steps = max(1, (len(feaS) - len(feaW)) // overlap + 1)
        dists = []
        for i in range(n_steps):
            iS = i * overlap
            seg = feaS[iS: iS + len(feaW)]
            if len(seg) < len(feaW): break
            per_frame = np.array([np.sum(np.abs(feaW[f] - seg[f])) for f in range(len(feaW))])
            dists.append(float(np.median(per_frame)))
        if not dists: dists = [float('inf')]
        similarity[sound.path] = float(np.min(dists))
        step_data[sound.path] = [round(v, 4) for v in dists[:60]]

    sorted_sim = sorted(similarity.items(), key=lambda x: x[1])
    top5 = sorted_sim[:k]

    # Normalize against GLOBAL max distance (all 500 songs) for meaningful percentages
    global_max = max(similarity.values()) or 1.0

    results = []
    for rank, (path, dist) in enumerate(top5, 1):
        steps = step_data[path]
        sim_pct = round((1 - dist / global_max) * 100, 1)
        results.append({
            'rank': rank,
            'path': path,
            'filename': os.path.basename(path),
            'distance': round(dist, 6),
            'sim_bar': max(sim_pct, 0.0),
            'steps': steps,
            'min_step': int(np.argmin(steps)),
            'total_steps': len(steps)
        })
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def stats():
    conn = sqlite3.connect(os.path.join(BASE_DIR, 'database.db'))
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM sounds")
    count = c.fetchone()[0]
    conn.close()
    return jsonify({'total': count, 'features': 6})

@app.route('/api/search', methods=['POST'])
def search():
    if 'audio' not in request.files:
        return jsonify({'error': 'Chưa chọn file'}), 400
    f = request.files['audio']
    safe_name = f.filename.replace('/', '_').replace('\\', '_')
    save_path = os.path.join(QUERY_FOLDER, safe_name)
    f.save(save_path)
    try:
        t0 = time.time()
        sounds = load_sounds()
        vec = Extract(sound_path=save_path).features()
        q = Sound(features=vec, path=save_path)
        results = search_songs(q, sounds)
        return jsonify({'ok': True, 'query': safe_name, 'results': results,
                        'time': round(time.time() - t0, 2), 'compared': len(sounds)})
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/serve/<path:filepath>')
def serve_file(filepath):
    full = os.path.normpath(os.path.join(BASE_DIR, filepath))
    if not full.startswith(BASE_DIR): abort(403)
    if not os.path.exists(full): abort(404)
    return send_file(full, mimetype='audio/wav')

if __name__ == '__main__':
    print("="*40)
    print("  MusicFinder - He CSDL Am Nhac")
    print("  Truy cap: http://localhost:5000")
    print("="*40)
    app.run(debug=False, port=5000)
