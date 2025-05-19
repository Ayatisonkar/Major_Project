import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import torchvision
from torchvision import datasets, models
from torchvision.transforms import functional as FT
from torchvision import transforms as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split, Dataset
import copy
import math
from PIL import Image
import cv2
import albumentations as A
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict, deque
import datetime
import time
from tqdm import tqdm
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def load_model(weights_path, device):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False)
    num_classes = 5 
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, num_classes
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def test_model_with_images(model, image_paths, device, classes):
    transform = T.Compose([
        T.ToTensor(),
    ])
    for image_path in image_paths:
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).to(device)

        # Perform inference
        with torch.no_grad():
            prediction = model([image_tensor])[0]

        # Visualize the results
        boxes = prediction['boxes'][prediction['scores'] > 0.8]
        labels = prediction['labels'][prediction['scores'] > 0.8]
        scores = prediction['scores'][prediction['scores'] > 0.8]

        image_with_boxes = draw_bounding_boxes(
            (image_tensor * 255).byte(),
            boxes,
            labels=[classes[label.item()] for label in labels],
            colors="red",
            width=3
        )

        plt.figure(figsize=(10, 10))
        plt.imshow(image_with_boxes.permute(1, 2, 0))
        plt.axis("off")
        plt.show()

import cv2

def test_model_with_video(model, video_path, device, classes, output_path=None):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define video writer if output_path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    transform = T.Compose([
        T.ToTensor(),
    ])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB and preprocess
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = transform(image).to(device)

        # Perform inference
        with torch.no_grad():
            prediction = model([image_tensor])[0]

        # Extract predictions with scores > 0.8
        boxes = prediction['boxes'][prediction['scores'] > 0.8]
        labels = prediction['labels'][prediction['scores'] > 0.8]

        helmet = []
        humans = []
        motorcycles = []
        person_bike = []

        for box, label in zip (boxes, labels):
            if(classes[label.item()] == 'helmet'):
                helmet.append(box)
            elif(classes[label.item()] == 'human'):
                humans.append(box)
            elif(classes[label.item()] == 'motorcycle'):
                motorcycles.append(box)
            elif(classes[label.item()] == 'person_bike'):
                person_bike.append(box)
        


        # Draw bounding boxes on the frame
        image_with_boxes = draw_bounding_boxes(
            (image_tensor * 255).byte(),
            boxes,
            labels=[classes[label.item()] for label in labels],
            colors="red",
            width=3
        )
        frame_with_boxes = image_with_boxes.permute(1, 2, 0).cpu().numpy()
        frame_with_boxes = cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR)
        #detect_no_helmet_violation(helmet, humans, person_bike, frame_with_boxes)
        detect_overcrowding_violation(helmet, humans, motorcycles, person_bike, frame_with_boxes)
        if output_path:
            out.write(frame_with_boxes)
        cv2.imshow('Video Prediction', frame_with_boxes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def detect_overcrowding_violation(helmets, humans, motorcycles, person_bikes, frame_with_boxes):
    for person_bike in person_bikes:
        count_human = 0
        for human in humans:
            if(is_near(human, person_bike)):
                count_human+=1
        if(count_human > 2):
            print("Overcrowding violation detected")
            cv2.imwrite("output_frame_overcrowd.jpg", frame_with_boxes)
def detect_no_helmet_violation(helmets, humans, person_bikes, frame_with_boxes):
    for person_bike in person_bikes:
        for human in humans:
            has_helmet = False
            for helmet in helmets:
                if(is_near(human, helmet) and is_near(person_bike, human)):
                    has_helmet = True
                    print(helmet, human, person_bike)
                    break
            if(not has_helmet):
                print("helmet violation detected")
                cv2.imwrite("output_frame.jpg", frame_with_boxes)
            else:
                print("Helmet detected")

def is_near(box1, box2, threshold=50):
    """
    Check if two bounding boxes are near each other.
    """
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
    return distance < threshold
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = "/home/eleensmathew/Traffic/model_weights_epoch_50.pth"
    model = load_model(weights_path, device)

    # Define class names (update based on your dataset)
    classes = ["__background__",'helmet', 'human' ,'motorcycle','person_bike']  # Replace with actual class names

    # Test with example images
    example_images = [
        "/home/eleensmathew/Traffic/traffic2/person on bike.v8i.coco/test/19134757_png.rf.c3a9d4b504e1e1834c8349e761c04249.jpg"
    ]
    #test_model_with_images(model, example_images, device, classes)
    test_model_with_video(model, "/home/eleensmathew/Traffic/videos/video1.mp4", device, classes)