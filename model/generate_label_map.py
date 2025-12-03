import os
import json

video_dir = os.path.join(os.path.dirname(__file__), '../data/video')
class_names = [d for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d)) and not d.startswith('.')]
class_names = sorted(class_names)
label_map = {name: idx for idx, name in enumerate(class_names)}

with open(os.path.join(os.path.dirname(__file__), '../config/label_map.json'), 'w') as f:
    json.dump(label_map, f, indent=2)

print('Label map generated:', label_map)
