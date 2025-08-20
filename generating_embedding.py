import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import pickle
import logging
import albumentations as A

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(BASE_DIR, "data", "Faces")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "data", "face_embeddings.pkl")
DEBUG_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "Debug_Output")
AUGMENTED_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "Augmented_Output")

LOG_FILE = os.path.join(BASE_DIR, "data", "log_embed.log")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE)
    ]
)

AUGMENTATIONS = [
    A.HorizontalFlip(p=1),
    A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=(-10, 10), p=1),
    A.RandomBrightnessContrast(p=1),
    A.GaussNoise(p=1),
    A.GaussianBlur(blur_limit=(3, 7), p=1),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1),
    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.2, 0.5), p=1),
    A.RandomScale(scale_limit=(-0.1, 0.1), p=1),
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.3),
    A.RandomGridShuffle(grid=(3, 3), p=0.2),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.RandomShadow(shadow_roi=(0, 0, 1, 0.5), num_shadows_lower=1, num_shadows_upper=2, p=0.3),
    A.CoarseDropout(max_holes=2, max_height=0.2, max_width=0.2, fill_value=0, p=0.3)
]

def save_debug_image(image, faces, filename, preprocessed_image=None, augmentation_name=None):
    if preprocessed_image is not None:
        debug_img = preprocessed_image.copy()
    else:
        debug_img = image.copy()

    for face in faces:
        box = face.bbox.astype(int)
        cv2.rectangle(debug_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        if hasattr(face, 'kps'):
            for kp in face.kps:
                x, y = kp
                cv2.circle(debug_img, (int(x), int(y)), 2, (255, 0, 0), 2)

    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
    
    if augmentation_name is None:
        debug_path = os.path.join(DEBUG_OUTPUT_DIR, f"debug_{os.path.basename(filename)}")
    else:
        debug_path = os.path.join(DEBUG_OUTPUT_DIR, f"debug_{augmentation_name}_{os.path.basename(filename)}")
    
    cv2.imwrite(debug_path, debug_img)
    logging.debug(f"Saved debug image to {debug_path}")

def preprocess_image_and_save_variations(image, filename):
    os.makedirs(AUGMENTED_OUTPUT_DIR, exist_ok=True)
    augmented_images = {}
    
    for aug in AUGMENTATIONS:
        transform = A.Compose([aug])
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = transform(image=image_rgb)
        augmented_image = augmented['image']

        max_size = 1280 
        height, width = augmented_image.shape[:2] 
        if max(height, width) > max_size: 
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            augmented_image = cv2.resize(augmented_image, (new_width, new_height))
        
        augmented_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(augmented_bgr, cv2.COLOR_BGR2LAB) 
        lightness, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) 
        cl = clahe.apply(lightness)
        enhanced_lab = cv2.merge((cl, a, b)) 
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        aug_name = aug.__class__.__name__ 
        augmented_images[aug_name] = enhanced
        
        aug_path = os.path.join(AUGMENTED_OUTPUT_DIR, f"{aug_name}_{os.path.basename(filename)}")
        cv2.imwrite(aug_path, enhanced)
        logging.debug(f"Saved augmented image to {aug_path}")
        
    return augmented_images 

def generate_embeddings():
    all_embeddings = []
    label_map = []
    
    total_images = 0
    processed_images = 0

    logging.info(f"Starting embedding generation from directory: {FACES_DIR}")

    for person_folder in os.listdir(FACES_DIR):
        person_path = os.path.join(FACES_DIR, person_folder)
        if os.path.isdir(person_path):
            logging.info(f"Processing person: {person_folder}")

            images = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for image_file in images:
                total_images += 1
                image_path = os.path.join(person_path, image_file)
                logging.debug(f"Processing image: {image_path}")

                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        raise ValueError(f"Failed to load image: {image_path}")

                    augmented_images = preprocess_image_and_save_variations(image, image_file)

                    for aug_name, processed_image in augmented_images.items():
                        faces = app.get(processed_image)
                        
                        save_debug_image(image, faces, image_file, processed_image, aug_name)

                        if faces:
                            face = faces[0]
                            embedding = face.embedding
                            all_embeddings.append(embedding)
                            label_map.append(person_folder)
                            processed_images += 1
                            logging.debug(f"Generated embedding for augmented {image_file} with {aug_name}")
                        else:
                            logging.warning(f"No face detected in augmented {image_file} with {aug_name}")
                            for det_size in [(320, 320), (960, 960)]:
                                app.prepare(ctx_id=0, det_size=det_size)
                                faces = app.get(processed_image)
                                if faces:
                                    face = faces[0]
                                    embedding = face.embedding
                                    all_embeddings.append(embedding)
                                    processed_images += 1
                                    logging.debug(
                                        f"Generated embedding for {image_file} with {aug_name} and det_size={det_size}")
                                    break
                except Exception as e:
                    logging.error(f"Error processing {image_path}: {str(e)}")

    logging.info("Saving all embeddings...")
    
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(
            {
                'label_map': label_map,
                'all_embeddings': np.array(all_embeddings)
            }, f)

    logging.info(f"Saved {len(all_embeddings)} embeddings to {EMBEDDINGS_FILE}")
    logging.info(f"Total embeddings collected: {len(all_embeddings)}")
    logging.info(f"Total images processed: {processed_images} out of {total_images}")

def main():
    generate_embeddings()

if __name__ == "__main__":
    main()

