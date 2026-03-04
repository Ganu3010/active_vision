from ultralytics import YOLO
import os
import torch

# Load the model
model = YOLO("yolov8m.pt")

all_distributions = {}

for file in os.listdir("sample_images/"):
    if file.endswith((".jpg", ".png")):
        path = f"sample_images/{file}"
        results = model(path)
        
        for result in results:
            # Check if classification probabilities exist
            if result.probs is not None:
                # result.probs.data contains the tensor of probabilities for all classes
                # .tolist() makes it easy to save to JSON or a dictionary
                probabilities = result.probs.data.tolist()
                all_distributions[file] = probabilities
                
                # If you just want the top 5 classes and their scores:
                # print(result.probs.top5conf) 
            
            # result.show()

print(all_distributions)
# Example: Save to a file for later analysis
# import json
# with open('distributions.json', 'w') as f:
#     json.dump(all_distributions, f)