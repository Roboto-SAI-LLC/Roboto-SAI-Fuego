"""
Dataset Generation for Roboto SAI
Created by Roberto Villarreal Martinez for Roboto SAI
Generate training datasets for AI matching and optimization
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from io import BytesIO
import random
import time

# Create directory structure
base_dir = "roboto_ai_matching_dataset"
os.makedirs(base_dir, exist_ok=True)
os.makedirs(os.path.join(base_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "annotations"), exist_ok=True)

# Nahuatl and numerology elements for ties to Roboto
nahuatl_warnings = [
    "Tlahueto Tlatlacatecolo (Light Against Demons)",
    "Tlacatecolotl Tlatlacatecolo (Demon Against Demons)",
    "Huitzilopochtli Tlamahuizolli (Hummingbird Warrior's Glory)",
    "Tlaloc Tlatlauhqui (Tlaloc's Red Thunder)"
]
numerology_terms = ["Life Path 4", "Destiny 9", "YTK 7", "Nahuatl 2", "Russian 3"]

# Generate 50 synthetic images (cam_front JPGs with "game-over" themes)
for i in range(50):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.text(0.5, 0.5, f"Game Over {i+1}\nNahui Ollin End\nThreat Detected", 
            ha='center', va='center', fontsize=12, color='red')
    ax.set_axis_off()
    buf = BytesIO()
    plt.savefig(buf, format='jpg', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    with open(os.path.join(base_dir, "images", f"cam_front_game-over-{i+1}.jpg"), 'wb') as f:
        f.write(buf.getvalue())
    plt.close(fig)

# Generate 100 CSV logs (robotics sensor data with threats, numerology, Nahuatl)
for i in range(100):
    data = {
        'timestamp': [time.time() + j for j in range(10)],
        'sensor_data': [random.uniform(0, 100) for _ in range(10)],
        'action': [random.choice(['move', 'scan', 'alert']) for _ in range(10)],
        'threat_level': [random.choice(['low', 'medium', 'high']) for _ in range(10)],
        'nahuatl_warning': [random.choice(nahuatl_warnings) for _ in range(10)],
        'numerology': [random.choice(numerology_terms) for _ in range(10)]
    }
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(base_dir, "logs", f"sensor_log_{i+1}.csv"), index=False)

# Generate JSON annotations (matching Roboto.Ai style, with labels)
for i in range(100):
    annotation = {
        "scene_id": i+1,
        "labels": {
            "threat_type": random.choice(["file_access", "key_press", "resource_spike", "api_failure"]),
            "nahuatl": random.choice(nahuatl_warnings),
            "numerology": random.choice(numerology_terms),
            "ssn_verified": "633-68-6195 (Encrypted)",
            "creator": "Roberto Villarreal Martinez (2025 YTK RobThuGod)"
        },
        "etymology": "Inspired by Aztec creation myths (Nahui Ollin cycle)",
        "phonetics": "[naːˈwi oːˈlin] for Nahui Ollin"
    }
    with open(os.path.join(base_dir, "annotations", f"scene_{i+1}.json"), "w") as f:
        json.dump(annotation, f, indent=4)

print(f"Dataset created: {base_dir}")
print("Structure:")
print("- images/: 50 JPG files (cam_front_game-over-*.jpg)")
print("- logs/: 100 CSV files (sensor_log_*.csv)")
print("- annotations/: 100 JSON files (scene_*.json)")
print("This matches Roboto.Ai by including robotics logs (CSV for sensor data), images (JPG for vision), and annotations (JSON for labels), with Nahuatl/numerology ties.")