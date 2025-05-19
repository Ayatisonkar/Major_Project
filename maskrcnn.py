import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import numpy as np
import requests
import threading
#from traffic_analysis_agent import image_to_json
import uuid
import os
# Load the class label names
CLASS_NAMES = ['BG', 'helmet', 'human', 'motorcycle', 'person_bike', 'extra']
API_URL = "http://localhost:5005/analyze"
class SimpleConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "coco_inference"
    
    # Set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)

# Initialize the Mask R-CNN model for inference and then load the weights
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(),
                             model_dir=os.getcwd())

# Load the weights into the model
model.load_weights(filepath="/home/eleensmathew/Traffic/mask_rcnn_traffic_cfg_0044.h5", 
                   by_name=True)
def draw_instances(image, boxes, masks, class_ids, class_names, scores):
    """Draw bounding boxes, masks, and class labels on the image."""
    for i in range(boxes.shape[0]):
        if not np.any(boxes[i]):
            continue

        # Extract box coordinates
        y1, x1, y2, x2 = boxes[i]
        color = tuple(np.random.randint(0, 256, 3).tolist())  # Green for bounding boxes

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Add the class label and score
        class_id = class_ids[i]
        score = scores[i]
        label = f"{class_names[class_id]}: {score:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Apply the mask to the image
        mask = masks[:, :, i]
        image = apply_mask(image, mask, color)

    return image

def draw_violation(image, boxes, masks, class_ids, class_names, scores):
    """Draw bounding boxes, masks, and class labels on the image."""
    helmet_boxes = [boxes[i] for i in range(len(class_ids)) if class_ids[i] == 1]  # Helmets
    human_boxes = [boxes[i] for i in range(len(class_ids)) if class_ids[i] == 2]  # Humans
    motorcycle_boxes = [boxes[i] for i in range(len(class_ids)) if class_ids[i] == 3]  # Motorcycles
    data = {}
    image_copy = image.copy()
    for i in range(boxes.shape[0]):
        if not np.any(boxes[i]):
            continue

        # Extract box coordinates
        y1, x1, y2, x2 = boxes[i]
        class_id = class_ids[i]
        score = scores[i]
        label = f"{class_names[class_id]}: {score:.2f}"

        # Default color for bounding boxes and masks
        color = (0, 255, 0)  # Green for normal cases

        #Check for humans without helmets
        if class_id == 2:  # Human
            has_helmet = any(iou(boxes[i], helmet_box) > 0.5 for helmet_box in helmet_boxes)
            if not has_helmet:
                color = (0, 0, 255)  # Red for humans without helmets
                mask = masks[:, :, i]
                image = apply_mask(image, mask, color)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Centroid of the bounding box
                
                cv2.putText(image, "No Helmet", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                data['helmet_violation'] = True

        if class_id == 1:
            mask = masks[:, :, i]
            color = (255, 0, 0)
            image = apply_mask(image, mask, color)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(image, "Helmet", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Check for overcrowded motorcycles
        if class_id == 4:  
            human_count = sum(iou(boxes[i], human_box) > 0.25 or is_inside(human_box, boxes[i])for human_box in human_boxes)
            #print(human_count)
            if human_count > 2:  # Overcrowded if more than 2 humans
                color = (0, 255, 255)  # Red for overcrowded motorcycles
                mask = masks[:, :, i]
                image = apply_mask(image, mask, color)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Centroid of the bounding box
                cv2.putText(image, "Overcrowded", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                data['overcrowded_violation'] = True

        # Add the class label and score
        #cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    unique_filename = f"violation_pictures/{uuid.uuid4()}.png"
    if(data.get('helmet_violation') or data.get('overcrowded_violation')):
        cv2.imwrite(unique_filename, image_copy)
        if(not data.get('helmet_violation')):
            data['helmet_violation'] = False
        if(not data.get('overcrowded_violation')):
            data['overcrowded_violation'] = False
        data['context_image'] = unique_filename
        fire_and_forget(API_URL, data)
    return image
def is_inside(box1, box2):#box2 is smaleer 
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    return (x1 >= x3 and y1 >= y3 and x2 <= x4 and y2 <= y4)
def iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two (x1, y1, x2, y2) boxes"""
    # Get coordinates of intersection area
    y1 = max(box1[0], box2[0])
    x1 = max(box1[1], box2[1])
    y2 = min(box1[2], box2[2])
    x2 = min(box1[3], box2[3])

    # Calculate intersection area
    inter_area = max(0, y2 - y1) * max(0, x2 - x1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = min(box1_area, box2_area)
    #print(inter_area/ union_area)
    return inter_area / union_area if union_area > 0 else 0

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image."""
    for c in range(3):  # Apply the mask to each channel (R, G, B)
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image
def detect_in_video(video_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define video writer to save the output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 5 == 0:

        # Convert the frame from BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform a forward pass of the network to obtain the results
            results = model.detect([image], verbose=0)
            r = results[0]

            # Visualize the detected objects on the frame
            frame_with_boxes = draw_violation(
                image=image,
                boxes=r['rois'],
                masks=r['masks'],
                class_ids=r['class_ids'],
                class_names=CLASS_NAMES,
                scores=r['scores'],
                #show_mask=False,  # Set to True if you want to display masks
                #show_bbox=True
            )

            # Convert the frame back to BGR for saving
            frame_with_boxes = cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR)

            # Write the frame to the output video
            out.write(frame_with_boxes)

            # Display the frame (optional)
            cv2.imshow('Video Detection', frame_with_boxes)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            out.write(frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def detect_in_image(image_path, output_path):
    """Detect objects in a single image and save the output."""
    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to open image file {image_path}")
        return

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform a forward pass of the network to obtain the results
    results = model.detect([image_rgb], verbose=0)
    r = results[0]

    # Visualize the detected objects on the image
    image_with_boxes = draw_violation(
        image=image_rgb,
        boxes=r['rois'],
        masks=r['masks'],
        class_ids=r['class_ids'],
        class_names=CLASS_NAMES,
        scores=r['scores']
    )

    # Save the output image
    cv2.imwrite(output_path, image_with_boxes)

    # Display the image (optional)
    cv2.imshow('Image Detection', image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR)

def fire_and_forget(url, data):
    def send():
        try:
            requests.post(url, json=data, timeout=1) 
        except Exception:
            pass

    threading.Thread(target=send, daemon=True).start()

# Path to the input video
video_path = "/home/eleensmathew/Traffic/videos/video5.mp4"

# Path to save the output video
output_path = "/home/eleensmathew/Traffic/videos/output_video.mp4"

# Perform detection on the video
#detect_in_video(video_path, output_path)
detect_in_image("/home/eleensmathew/Traffic/image.png", "output.png")