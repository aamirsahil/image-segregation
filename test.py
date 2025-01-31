import cv2
import numpy as np
import torch
from deepface import DeepFace
from torchvision import transforms
from PIL import Image
from aesthetic_model import NIMA  # Hypothetical NIMA model

def check_lighting(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray) / 255
    return brightness  # Closer to 1 means well-lit

def check_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return min(laplacian_var / 100, 1)  # Higher means sharper

def check_resolution(image):
    h, w, _ = image.shape
    return 1 if h >= 720 and w >= 720 else 0  # True if above 720p

def check_frontal_face(image):
    faces = DeepFace.extract_faces(image)
    return 1 if len(faces) > 0 else 0  # True if face detected

def check_smiling(image):
    analysis = DeepFace.analyze(image, actions=['emotion'])
    return analysis['dominant_emotion'] in ['happy', 'smile']

def check_rule_of_thirds(image):
    h, w, _ = image.shape
    thirds_x = [w/3, 2*w/3]
    thirds_y = [h/3, 2*h/3]
    
    faces = DeepFace.extract_faces(image)
    if faces:
        x, y, _, _ = faces[0]['facial_area'].values()
        return any(abs(x - tx) < w/10 for tx in thirds_x) and any(abs(y - ty) < h/10 for ty in thirds_y)
    return 0

def check_vibrant_colors(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv[:, :, 1]) / 255
    return saturation

def check_golden_hour(image):
    avg_color = np.mean(image, axis=(0, 1))
    warm_tone = avg_color[2] > avg_color[1] > avg_color[0]  # R > G > B
    return 1 if warm_tone else 0

def check_nima_score(image):
    model = NIMA()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = transform(Image.fromarray(image)).unsqueeze(0)
    score = model(image).item()
    return score / 10  # Normalize between 0 and 1

def check_emotional_impact(image):
    return check_smiling(image)  # Placeholder for deeper sentiment analysis

def analyze_image(image_path):
    image = cv2.imread(image_path)
    return {
        "lighting": check_lighting(image),
        "sharpness": check_sharpness(image),
        "resolution": check_resolution(image),
        "face_present": check_frontal_face(image),
        "smiling": check_smiling(image),
        "rule_of_thirds": check_rule_of_thirds(image),
        "vibrant_colors": check_vibrant_colors(image),
        "golden_hour": check_golden_hour(image),
        "aesthetic_score": check_nima_score(image),
        "emotional_impact": check_emotional_impact(image)
    }

# Example usage
image_path = "sample.jpg"
print(analyze_image(image_path))

import cv2
import numpy as np
import torch
from deepface import DeepFace
from torchvision import transforms
from PIL import Image
from aesthetic_model import NIMA  # Hypothetical NIMA model

def check_lighting(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray) / 255
    return brightness  # Closer to 1 means well-lit

def check_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return min(laplacian_var / 100, 1)  # Higher means sharper

def check_resolution(image):
    h, w, _ = image.shape
    return 1 if h >= 720 and w >= 720 else 0  # True if above 720p

def check_frontal_face(image):
    faces = DeepFace.extract_faces(image)
    return 1 if len(faces) > 0 else 0  # True if face detected

def check_smiling(image):
    analysis = DeepFace.analyze(image, actions=['emotion'])
    return analysis['dominant_emotion'] in ['happy', 'smile']

def check_rule_of_thirds(image):
    h, w, _ = image.shape
    thirds_x = [w/3, 2*w/3]
    thirds_y = [h/3, 2*h/3]
    
    faces = DeepFace.extract_faces(image)
    if faces:
        x, y, _, _ = faces[0]['facial_area'].values()
        return any(abs(x - tx) < w/10 for tx in thirds_x) and any(abs(y - ty) < h/10 for ty in thirds_y)
    return 0

def check_vibrant_colors(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv[:, :, 1]) / 255
    return saturation

def check_golden_hour(image):
    avg_color = np.mean(image, axis=(0, 1))
    warm_tone = avg_color[2] > avg_color[1] > avg_color[0]  # R > G > B
    return 1 if warm_tone else 0

def check_nima_score(image):
    model = NIMA()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = transform(Image.fromarray(image)).unsqueeze(0)
    score = model(image).item()
    return score / 10  # Normalize between 0 and 1

def check_emotional_impact(image):
    return check_smiling(image)  # Placeholder for deeper sentiment analysis

def check_depth_of_field(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return np.mean(edges) / 255  # Higher means stronger focus variation

def check_leading_lines(image):
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    return 1 if lines is not None else 0

def check_symmetry(image):
    h, w, _ = image.shape
    left = image[:, :w//2]
    right = cv2.flip(image[:, w//2:], 1)
    return np.mean(cv2.absdiff(left, right)) / 255

def check_dynamic_range(image):
    min_val, max_val, _, _ = cv2.minMaxLoc(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return (max_val - min_val) / 255

def check_candid_moment(image):
    return check_smiling(image)  # Placeholder for deeper candid analysis

def check_foreground_interest(image):
    return check_depth_of_field(image)  # Placeholder for layered depth analysis

def check_natural_framing(image):
    return check_leading_lines(image)  # Placeholder for framing detection

def check_minimalist_composition(image):
    std_dev = np.std(image)
    return 1 if std_dev < 50 else 0  # Lower variance means minimalism

def check_storytelling(image):
    return check_emotional_impact(image)  # Placeholder for complex AI storytelling

def check_texture_detail(image):
    return check_sharpness(image)  # Sharpness correlates with texture detail

def check_golden_ratio(image):
    return check_rule_of_thirds(image)  # Placeholder for full golden ratio detection

def check_reflections(image):
    return check_symmetry(image)  # Placeholder for reflection detection

def check_timelessness(image):
    return 1  # Placeholder assuming no trendy filters

def analyze_image(image_path):
    image = cv2.imread(image_path)
    return {
        "lighting": check_lighting(image),
        "sharpness": check_sharpness(image),
        "resolution": check_resolution(image),
        "face_present": check_frontal_face(image),
        "smiling": check_smiling(image),
        "rule_of_thirds": check_rule_of_thirds(image),
        "vibrant_colors": check_vibrant_colors(image),
        "golden_hour": check_golden_hour(image),
        "aesthetic_score": check_nima_score(image),
        "emotional_impact": check_emotional_impact(image),
        "depth_of_field": check_depth_of_field(image),
        "leading_lines": check_leading_lines(image),
        "symmetry": check_symmetry(image),
        "dynamic_range": check_dynamic_range(image),
        "candid_moment": check_candid_moment(image),
        "foreground_interest": check_foreground_interest(image),
        "natural_framing": check_natural_framing(image),
        "minimalist_composition": check_minimalist_composition(image),
        "storytelling": check_storytelling(image),
        "texture_detail": check_texture_detail(image),
        "golden_ratio": check_golden_ratio(image),
        "reflections": check_reflections(image),
        "timelessness": check_timelessness(image)
    }

# Example usage
image_path = "sample.jpg"
print(analyze_image(image_path))

import cv2
import numpy as np
import torch
from deepface import DeepFace
from torchvision import transforms
from PIL import Image
from aesthetic_model import NIMA  # Hypothetical NIMA model

def check_lighting(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray) / 255
    return brightness  # Closer to 1 means well-lit

def check_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return min(laplacian_var / 100, 1)  # Higher means sharper

def check_resolution(image):
    h, w, _ = image.shape
    return 1 if h >= 720 and w >= 720 else 0  # True if above 720p

def check_frontal_face(image):
    faces = DeepFace.extract_faces(image)
    return 1 if len(faces) > 0 else 0  # True if face detected

def check_smiling(image):
    analysis = DeepFace.analyze(image, actions=['emotion'])
    return analysis['dominant_emotion'] in ['happy', 'smile']

def check_image_centered(image):
    h, w, _ = image.shape
    faces = DeepFace.extract_faces(image)
    if faces:
        x, y, fw, fh = faces[0]['facial_area'].values()
        return 1 if (w/4 < x + fw/2 < 3*w/4) and (h/4 < y + fh/2 < 3*h/4) else 0
    return 0

def check_no_obstructions(image):
    return check_frontal_face(image)  # Placeholder for deeper object obstruction detection

def check_proper_framing(image):
    h, w, _ = image.shape
    faces = DeepFace.extract_faces(image)
    if faces:
        x, y, fw, fh = faces[0]['facial_area'].values()
        return 1 if (x > 10 and y > 10 and x + fw < w - 10 and y + fh < h - 10) else 0
    return 0

def check_balanced_exposure(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_val, max_val, _, _ = cv2.minMaxLoc(gray)
    return 1 if (max_val - min_val) / 255 > 0.3 else 0

def analyze_image(image_path):
    image = cv2.imread(image_path)
    return {
        "lighting": check_lighting(image),
        "sharpness": check_sharpness(image),
        "resolution": check_resolution(image),
        "face_present": check_frontal_face(image),
        "eyes_open" : check_eyes_open(image),
        "natural_colors" : check_natural_colors(image),
        "uniform_backgroud" : check_uniform_backgroud(image),
        "strong_focus" : check_strong_focus(image),
        "strong_focal_point" : check_focal_point(image),
        "smiling": check_smiling(image),
        "image_centered": check_image_centered(image),
        "no_obstructions": check_no_obstructions(image),
        "proper_framing": check_proper_framing(image),
        "balanced_exposure": check_balanced_exposure(image),
        "rule_of_thirds": check_rule_of_thirds(image),
        "vibrant_colors": check_vibrant_colors(image),
        "golden_hour": check_golden_hour(image),
        "aesthetic_score": check_nima_score(image),
        "emotional_impact": check_emotional_impact(image),
        "depth_of_field": check_depth_of_field(image),
        "leading_lines": check_leading_lines(image),
        "symmetry": check_symmetry(image),
        "dynamic_range": check_dynamic_range(image),
        "candid_moment": check_candid_moment(image),
        "foreground_interest": check_foreground_interest(image),
        "natural_framing": check_natural_framing(image),
        "minimalist_composition": check_minimalist_composition(image),
        "storytelling": check_storytelling(image),
        "texture_detail": check_texture_detail(image),
        "golden_ratio": check_golden_ratio(image),
        "reflections": check_reflections(image),
        "timelessness": check_timelessness(image)
    }