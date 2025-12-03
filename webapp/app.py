from flask import Flask, render_template, Response, jsonify, request, send_from_directory
app = Flask(__name__)

# Simulate detection for a video and save to DB
@app.route('/simulate_detection', methods=['POST'])
def simulate_detection():
    data = request.get_json()
    video_path = data.get('video_path')
    if not video_path:
        return jsonify({'status': 'error', 'message': 'No video_path provided'}), 400
    # Extract anomaly class from video path (folder name)
    parts = video_path.strip('/').split('/')
    if len(parts) < 3:
        return jsonify({'status': 'error', 'message': 'Invalid video_path'}), 400
    anomaly_class = parts[-2]
    confidence = 0.99
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn = get_db_connection()
    conn.execute('INSERT INTO detections (event_type, confidence, timestamp) VALUES (?, ?, ?)',
                 (anomaly_class, confidence, timestamp))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success', 'event_type': anomaly_class, 'confidence': confidence, 'timestamp': timestamp})




import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import requests
import sqlite3
from collections import Counter
from datetime import datetime
import cv2
import numpy as np
import torch
from torchvision import transforms
from model.inference import EnhancedAnomalyDetector
import time

# Serve video files from data/video
@app.route('/video_data/<path:filename>')
def video_data(filename):
    return send_from_directory(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/video')), filename)

# Route to get 500 random videos named 1 to 500
import glob
import random

@app.route('/video_list')
def video_list():
    # Find all mp4 files in data/video subfolders
    # Gather all folders
    folders = [f for f in glob.glob('data/video/*') if os.path.isdir(f)]
    all_videos = []
    for folder in folders:
        all_videos.extend(glob.glob(os.path.join(folder, '*.mp4')))
    # Randomly select 200 from all folders
    selected = random.sample(all_videos, min(200, len(all_videos)))
    # Debug: print selected video classes and count
    print('Selected video classes:', [os.path.basename(os.path.dirname(p)) for p in selected])
    print('Selected video count:', len(selected))
    video_list = []
    for idx, path in enumerate(selected, 1):
        folder = os.path.basename(os.path.dirname(path))
        filename = os.path.basename(path)
        video_list.append({
            'id': idx,
            'name': str(idx),
            'path': f'/video_data/{folder}/{filename}',
            'class': folder
        })
    response = jsonify(video_list)
    response.headers.add('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/delete_detection/<int:detection_id>', methods=['POST'])
def delete_detection(detection_id):
    conn = get_db_connection()
    conn.execute('DELETE FROM detections WHERE id = ?', (detection_id,))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})

DB_PATH = 'SmartHome.db'



VIDEO_MODEL_PATH = 'model/model_weights.pth'
video_model = EnhancedAnomalyDetector(
    visual_dim=512,
    audio_dim=0,
    hidden_dim=256,
    num_classes=18,
    sequence_length=16
)
video_model.load_state_dict(torch.load(VIDEO_MODEL_PATH, map_location='cpu'), strict=False)
video_model.eval()

# Load feature extractor (ResNet18 without final layer)
import torchvision.models as models
feature_extractor = models.resnet18(weights=None)
feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
feature_extractor.eval()

video_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Placeholder for audio model loading
AUDIO_MODEL_PATH = '../model/audio_model.pth'
# audio_model = ... # Load your trained audio model here

# Get anomaly class names from training folders
with open(os.path.join(os.path.dirname(__file__), '../config/label_map.json')) as f:
    label_map = json.load(f)
inv_label_map = {v: k for k, v in label_map.items()}
ANOMALY_CLASSES = [inv_label_map[i] for i in range(len(inv_label_map))]

# Initialize DB if needed
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    # Provide dummy data to avoid Jinja2 UndefinedError
    anomaly_data = {'labels': ['No Data'], 'counts': [0]}
    # Get recent detections from DB
    conn = get_db_connection()
    rows = conn.execute('SELECT id, event_type, confidence, timestamp, screenshot_path FROM detections ORDER BY timestamp DESC LIMIT 100').fetchall()
    detections = [
        {'id': row['id'], 'event_type': row['event_type'], 'confidence': row['confidence'], 'timestamp': row['timestamp'], 'screenshot_path': row['screenshot_path']}
        for row in rows
    ]
    conn.close()
    # Get 200 videos for dropdown
    folders = [f for f in glob.glob('data/video/*') if os.path.isdir(f)]
    all_videos = []
    for folder in folders:
        all_videos.extend(glob.glob(os.path.join(folder, '*.mp4')))
    selected = random.sample(all_videos, min(200, len(all_videos)))
    video_list = []
    for idx, path in enumerate(selected, 1):
        folder = os.path.basename(os.path.dirname(path))
        filename = os.path.basename(path)
        video_list.append({
            'id': idx,
            'name': str(idx),
            'path': f'/video_data/{folder}/{filename}',
            'class': folder
        })
    return render_template('index.html', anomaly_data=anomaly_data, detections=detections, video_list=video_list)


# Updated anomaly detection using trained video model
def detect_anomaly(frame):
    from PIL import Image
    img = Image.fromarray(frame)
    input_tensor = video_transform(img).unsqueeze(0)
    with torch.no_grad():
        output = video_model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred].item()
    print(f"[DEBUG] detect_anomaly: Predicted={pred}, Event={ANOMALY_CLASSES[pred] if pred < len(ANOMALY_CLASSES) else 'unknown'}, Confidence={confidence}")
    event_type = ANOMALY_CLASSES[pred] if pred < len(ANOMALY_CLASSES) else 'unknown'
    h, w = frame.shape[:2]
    # Bounding box around the whole frame
    x1, y1, x2, y2 = 0, 0, w - 1, h - 1
    print(f"[DEBUG] Predicted: {event_type}, Confidence: {confidence}")
    if event_type != 'normal' and confidence > 0.1:
        return {
            'event_type': event_type,
            'confidence': round(confidence, 2),
            'bbox': [x1, y1, x2, y2]
        }
    return None


def gen_frames(camera_source='builtin', video_path=None):
    import os
    import time
    # Select camera or video file
    if video_path:
        # Ensure video_path is relative, e.g., arson/Arson001_x264.mp4
        rel_path = video_path.lstrip('/')
        cap = cv2.VideoCapture(os.path.join('data/video', rel_path))
    else:
        if camera_source == 'builtin':
            cap = cv2.VideoCapture(0)
        elif camera_source == 'usb':
            cap = cv2.VideoCapture(1)
        elif camera_source == 'ip':
            ip_cam_url = 'http://192.168.1.100:8080/video'
            cap = cv2.VideoCapture(ip_cam_url)
        else:
            cap = cv2.VideoCapture(0)
    screenshot_dir = os.path.join(os.path.dirname(__file__), 'static/screenshots')
    os.makedirs(screenshot_dir, exist_ok=True)
    last_anomaly_time = {}
    # Buffer for robust anomaly detection
    from collections import deque, Counter
    prediction_buffer = deque(maxlen=7)  # For majority voting
    sequence_buffer = deque(maxlen=16)   # For sequence model input
    # Get FPS for video playback timing
    fps = cap.get(cv2.CAP_PROP_FPS) if video_path else None
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Add frame timing for video playback
            if fps and fps > 0:
                time.sleep(1.0 / fps)
            # Check if frame is too dark (average pixel value below threshold)
            if np.mean(frame) < 30:
                result = None
            else:
                # Add frame to sequence buffer
                sequence_buffer.append(frame)
                result = None
                # Run inference every 5 frames for smoother streaming
                if len(sequence_buffer) == 16 and len(prediction_buffer) % 5 == 0:
                    frames_seq = np.stack(sequence_buffer, axis=0)  # (16, H, W, C)
                    import torch
                    from PIL import Image
                    frame_features = []
                    for f in frames_seq:
                        frame_tensor = video_transform(Image.fromarray(f)).unsqueeze(0)  # (1, C, H, W)
                        with torch.no_grad():
                            feat = feature_extractor(frame_tensor)  # (1, 512, 1, 1)
                            feat = feat.view(512)
                        frame_features.append(feat)
                    visual_seq = torch.stack(frame_features).unsqueeze(0).to('cpu')  # (1, 16, 512)
                    audio_seq = torch.zeros((1, visual_seq.shape[1], 0)).to(visual_seq.device)
                    outputs = video_model(visual_seq, audio_seq)
                    avg_outputs = outputs['logits'].mean(dim=0, keepdim=True)
                    pred = torch.argmax(avg_outputs, dim=1).item()
                    confidence = torch.softmax(avg_outputs, dim=1)[0, pred].item()
                    event_type = ANOMALY_CLASSES[pred] if pred < len(ANOMALY_CLASSES) else 'unknown'
                    print(f"[DEBUG] gen_frames: Predicted={pred}, Event={event_type}, Confidence={confidence}")
                    if event_type != 'normal' and confidence > 0.1:
                        result = {
                            'event_type': event_type,
                            'confidence': round(confidence, 2),
                            'bbox': [0, 0, frame.shape[1] - 1, frame.shape[0] - 1]
                        }
                # Add result to prediction buffer for majority voting
                if result:
                    prediction_buffer.append(result['event_type'])
                else:
                    prediction_buffer.append('normal')

            # Only show anomaly if majority of buffer is not 'normal'
            buffer_count = Counter(prediction_buffer)
            if buffer_count:
                most_common, count = buffer_count.most_common(1)[0]
            else:
                most_common, count = 'normal', 0
            show_anomaly = most_common != 'normal' and count > len(prediction_buffer) // 2

            if show_anomaly and result and result['event_type'] == most_common:
                now = int(time.time())
                event_type = result['event_type']
                # Draw anomaly overlay at detected location
                if 'bbox' in result:
                    x1, y1, x2, y2 = result['bbox']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 4)
                    cv2.putText(frame, f"{result['event_type']} ({result['confidence']})", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                # Only save/log if not duplicate within 60 seconds
                if not (event_type in last_anomaly_time and now - last_anomaly_time[event_type] < 60):
                    last_anomaly_time[event_type] = now
                    filename = f"detection_{now}.jpg"
                    screenshot_path = os.path.join(screenshot_dir, filename)
                    cv2.imwrite(screenshot_path, frame)
                    rel_path = f"screenshots/{filename}"
                    conn = get_db_connection()
                    conn.execute('INSERT INTO detections (event_type, confidence, screenshot_path) VALUES (?, ?, ?)', (result['event_type'], result['confidence'], rel_path))
                    conn.commit()
                    conn.close()
                    # Send WhatsApp message with screenshot
                    try:
                        api_url = "https://xry8pl.api.infobip.com/whatsapp/1/message/image"
                        headers = {
                            "Authorization": "App cc4225c4f2ec0b8cb91a1ef8462db686-ea3dd16c-7bd2-41b1-a4cf-49817a456a08",
                            "Content-Type": "application/json",
                            "Accept": "application/json"
                        }
                        # Read image as base64
                        import base64
                        with open(screenshot_path, "rb") as img_file:
                            img_b64 = base64.b64encode(img_file.read()).decode('utf-8')
                        payload = {
                            "from": "12173969257",
                            "to": "+27744443628",
                            "content": {
                                "image": {
                                    "url": "",
                                    "caption": f"Anomaly detected: {event_type} ({result['confidence']})"
                                }
                            }
                        }
                        # Always send WhatsApp text message only
                        api_url = "https://xry8pl.api.infobip.com/whatsapp/1/message/text"
                        payload = {
                            "from": "12173969257",
                            "to": "+27744443628",
                            "content": {
                                "text": f"Anomaly detected: {event_type} ({result['confidence']})"
                            }
                        }
                        response = requests.post(api_url, headers=headers, json=payload)
                        print("Infobip response:", response.status_code, response.text)
                    except Exception as e:
                        print(f"WhatsApp notification failed: {e}")
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/video_feed')
def video_feed():
    source = request.args.get('source', 'builtin')
    video = request.args.get('video', None)
    if video:
        return Response(gen_frames(video_path=video), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(gen_frames(camera_source=source), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/log_detection', methods=['POST'])
def log_detection():
    # Example: expects JSON with event_type and confidence
    data = request.get_json()
    event_type = data.get('event_type')
    confidence = data.get('confidence')
    conn = get_db_connection()
    conn.execute('INSERT INTO detections (event_type, confidence) VALUES (?, ?)', (event_type, confidence))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})

app_start_time = time.time()

@app.route('/stats')
def stats():
    conn = sqlite3.connect('SmartHome.db')
    c = conn.cursor()
    # Get last 100 detections for table
    c.execute('SELECT id, event_type, confidence, timestamp, screenshot_path FROM detections ORDER BY timestamp DESC LIMIT 100')
    detections = [
        {'id': row[0], 'event_type': row[1], 'confidence': row[2], 'timestamp': row[3], 'screenshot_path': row[4]}
        for row in c.fetchall()
    ]
    # Get anomaly frequency for last 100 detections (for chart)
    c.execute('SELECT event_type FROM detections ORDER BY timestamp DESC LIMIT 100')
    all_types = [row[0] for row in c.fetchall()]
    anomaly_counts = Counter(all_types)
    anomaly_data = {
        'labels': list(anomaly_counts.keys()) if anomaly_counts else ['No Data'],
        'counts': list(anomaly_counts.values()) if anomaly_counts else [0]
    }
    # Detection rate (last 24h)
    import datetime
    now = datetime.datetime.now()
    c.execute('SELECT timestamp FROM detections WHERE timestamp >= ?', ((now - datetime.timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S'),))
    detections_24h = c.fetchall()
    detection_rate = len(detections_24h)
    # Line chart: detections per hour (last 24h, only nonzero)
    c.execute('SELECT timestamp FROM detections WHERE timestamp >= ?', ((now - datetime.timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S'),))
    timestamps = [row[0] for row in c.fetchall()]
    hours = [(datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').strftime('%H:00')) for ts in timestamps]
    hour_counts = Counter(hours)
    # Only show hours with at least one detection
    line_labels = [label for label, count in hour_counts.items() if count > 0]
    line_counts = [count for label, count in hour_counts.items() if count > 0]
    if not line_labels:
        line_labels = ['No Data']
        line_counts = [0]
    line_chart_data = {
        'labels': line_labels,
        'counts': line_counts
    }
    # Get stats
    c.execute('SELECT COUNT(*) FROM detections')
    total_detections = c.fetchone()[0]
    most_anomaly = Counter(all_types).most_common(1)[0][0] if all_types else 'N/A'
    c.execute('SELECT timestamp FROM detections ORDER BY timestamp DESC LIMIT 1')
    last_detection = c.fetchone()[0] if total_detections > 0 else 'N/A'
    conn.close()
    return render_template(
        'stats.html',
        detections=detections,
        total_detections=total_detections,
        most_anomaly=most_anomaly,
        last_detection=last_detection,
        detection_rate=detection_rate,
        anomaly_data=anomaly_data,
        line_chart_data=line_chart_data
    )

@app.route('/system')
def system():
    conn = get_db_connection()
    uptime = time.time() - app_start_time
    last_row = conn.execute('SELECT * FROM detections ORDER BY timestamp DESC LIMIT 1').fetchone()
    last_detection = last_row['timestamp'] if last_row else 'N/A'
    total_detections = conn.execute('SELECT COUNT(*) FROM detections').fetchone()[0]
    anomaly_row = conn.execute('SELECT event_type, COUNT(*) as cnt FROM detections GROUP BY event_type ORDER BY cnt DESC LIMIT 1').fetchone()
    most_anomaly = anomaly_row['event_type'] if anomaly_row else 'N/A'
    conn.close()
    return render_template('system.html', uptime=f"{uptime/3600:.2f} hours", last_detection=last_detection, total_detections=total_detections, most_anomaly=most_anomaly, title='System Status')

# Scaffold audio detection endpoint
@app.route('/audio_detect', methods=['POST'])
def audio_detect():
    # Placeholder: expects audio file upload, runs audio model inference
    # Implement actual audio detection logic here
    return {'status': 'not implemented'}

if __name__ == '__main__':
    app.run(debug=True, port=5001, threaded=True)
