from ultralytics import YOLO
import os
import pickle
model = YOLO("yolov8n-cls.pt")
outputs = {}
for file in os.listdir("sample_images/"):
    if file.endswith((".jpg", ".png")):
        results = model(f"sample_images/{file}")
        
        for result in results:
            # Check if any boxes were actually detected
            if result.probs is not None and len(result.probs) > 0:
                # print(f"Predicted class for {file}: {result.probs}")
                outputs[file] = result.probs
            result.save(f"outputs/{file}")

with open("outputs/outputs.pkl", "wb") as f:
    pickle.dump(outputs , f)
