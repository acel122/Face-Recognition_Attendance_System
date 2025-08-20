import cv2
import os
import shutil
import time
import logging
import tkinter as tk
from tkinter import font as tkFont
from PIL import Image, ImageTk
import numpy as np
from ultralight import UltraLightDetector

detector = UltraLightDetector()
DETECTION_THRESHOLD = 0.5

class Face_Register:
    def __init__(self):
        self.current_frame_faces_cnt = 0
        self.existing_faces_cnt = 0
        self.ss_cnt = 0

        # Tkinter GUI
        self.win = tk.Tk()
        self.win.title("Attendance System")
        self.win.geometry("1000x550")

        # GUI left part
        self.frame_left_camera = tk.Frame(self.win)
        self.label = tk.Label(self.win)
        self.label.pack(side=tk.LEFT)
        self.frame_left_camera.pack()

        # GUI right part
        self.frame_right_info = tk.Frame(self.win)
        self.label_cnt_face_in_database = tk.Label(self.frame_right_info, text=str(self.existing_faces_cnt))
        self.label_fps_info = tk.Label(self.frame_right_info, text="")

        # Nama
        self.input_name = tk.Entry(self.frame_right_info)
        self.input_name_char = ""

        # NIK
        self.input_id = tk.Entry(self.frame_right_info)
        self.input_id_char = ""

        # Divisi
        self.input_department = tk.Entry(self.frame_right_info)
        self.input_department_char = ""

        #Information
        self.label_warning = tk.Label(self.frame_right_info)
        self.label_face_cnt = tk.Label(self.frame_right_info, text="Faces in current frame: ")
        self.log_all = tk.Label(self.frame_right_info)

        self.font_title = tkFont.Font(family='Helvetica', size=20, weight='bold')
        self.font_step_title = tkFont.Font(family='Helvetica', size=15, weight='bold')
        self.font_warning = tkFont.Font(family='Helvetica', size=15, weight='bold')

        self.path_photos_from_camera = "data/Faces"
        self.current_face_dir = ""
        self.font = cv2.FONT_ITALIC

        self.current_frame = np.ndarray
        self.face_ROI_image = np.ndarray
        self.face_ROI_width_start = 0
        self.face_ROI_height_start = 0
        self.face_ROI_width = 0
        self.face_ROI_height = 0
        self.ww = 0
        self.hh = 0
        self.current_frame_copy = None

        self.out_of_range_flag = False
        self.face_folder_created_flag = False

        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        self.target_size = (300, 400)

        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Warning: Failed to initialize camera. Will retry in process loop.")
                self.cap = None
        except Exception as e:
            print(f"Error initializing camera: {e}")
            self.cap = None

    def GUI_clear_data(self):
        folders_rd = os.listdir(self.path_photos_from_camera)
        for i in range(len(folders_rd)):
            shutil.rmtree(self.path_photos_from_camera + folders_rd[i])
        if os.path.isfile("data/face_embeddings.pkl"):
            os.remove("data/face_embeddings.pkl")
        self.label_cnt_face_in_database['text'] = "0"
        self.existing_faces_cnt = 0
        self.log_all["text"] = "Face images and face embedding file already been removed!"

    def GUI_get_input_name_and_id(self):
        self.input_name_char = self.input_name.get()
        self.input_id_char = self.input_id.get()
        self.input_department_char = self.input_department.get()
        self.create_face_folder()
        self.label_cnt_face_in_database['text'] = str(self.existing_faces_cnt)

    def GUI_info(self):
        tk.Label(self.frame_right_info,
                 text="Attendance System",
                 font=self.font_title).grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=2, pady=20)

        tk.Label(self.frame_right_info, text="FPS: ").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_fps_info.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        self.label_warning.grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        # Step 1: Clear old data
        tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text="Step 1: Hapus Foto").grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)
        tk.Button(self.frame_right_info,
                  text='Hapus',
                  command=self.GUI_clear_data).grid(row=6, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        # Step 2: Input name and create folders for face
        tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text="Step 2: Input Nama dan NIK").grid(row=7, column=0, columnspan=2, sticky=tk.W, padx=5,
                                                           pady=20)

        tk.Label(self.frame_right_info, text="Nama: ").grid(row=8, column=0, sticky=tk.W, padx=5, pady=0)
        self.input_name.grid(row=8, column=1, sticky=tk.W, padx=0, pady=2)

        tk.Label(self.frame_right_info, text="NIK: ").grid(row=9, column=0, sticky=tk.W, padx=5, pady=0)
        self.input_id.grid(row=9, column=1, sticky=tk.W, padx=0, pady=2)

        tk.Label(self.frame_right_info, text="Divisi: ").grid(row=10, column=0, sticky=tk.W, padx=5, pady=0)
        self.input_department.grid(row=10, column=1, sticky=tk.W, padx=0, pady=2) 

        tk.Button(self.frame_right_info,
                  text='Input',
                  command=self.GUI_get_input_name_and_id
                  ).grid(row=11, column=0, columnspan=3, sticky=tk.W, pady=(0, 20))

        # Step 3: Save current face in frame
        tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text="Step 3: Simpan Foto").grid(row=12, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)

        tk.Button(self.frame_right_info,
                  text='Ambil Foto',
                  command=self.save_current_face).grid(row=13, column=0, columnspan=3, sticky=tk.W)

        # Show log in GUI
        self.log_all.grid(row=14, column=0, columnspan=20, sticky=tk.W, padx=5, pady=20)

        self.frame_right_info.pack()

    def pre_work_mkdir(self):
        if os.path.isdir(self.path_photos_from_camera):
            pass
        else:
            os.mkdir(self.path_photos_from_camera)

    def check_existing_faces_cnt(self):
        if os.listdir("data/Faces"):
            person_list = os.listdir("data/Faces")
            person_num_list = []
            for person in person_list:
                try:
                    person_order = person.split('_')[1]
                    person_num_list.append(int(person_order))
                except (ValueError, IndexError):
                    print(f"Skipping non-numeric folder: {person}")
            self.existing_faces_cnt = max(person_num_list) if person_num_list else 0
        else:
            self.existing_faces_cnt = 0

    def update_fps(self):
        now = time.time()
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now
        self.label_fps_info["text"] = str(self.fps.__round__(2))

    def create_face_folder(self):
        if self.input_name_char and self.input_id_char and self.input_department_char:
            self.current_face_dir = self.path_photos_from_camera + \
                                    self.input_id_char + "_" + self.input_name_char + "_" + self.input_department_char
        else:
            self.current_face_dir = self.path_photos_from_camera + \
                                    "person_" + str(self.existing_faces_cnt)

        if os.path.exists(self.current_face_dir):
            self.log_all["text"] = "Folder \"" + self.current_face_dir + "/\" already exists!"
            logging.info("\n%-40s %s", "Folder exists:", self.current_face_dir)
            existing_images = [f for f in os.listdir(self.current_face_dir) if f.endswith('.jpg')]
            if existing_images:
                last_image_num = max([int(f.split('_')[-1].split('.')[0]) for f in existing_images])
                self.ss_cnt = last_image_num 
            else:
                self.ss_cnt = 0
        else:
            os.makedirs(self.current_face_dir)
            self.log_all["text"] = "\"" + self.current_face_dir + "/\" created!"
            logging.info("\n%-40s %s", "Create folders:", self.current_face_dir)
            self.ss_cnt = 0

        self.face_folder_created_flag = True

    def crop_face(self, image, bbox, landmarks):
        height, width = image.shape[:2]
        
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        
        crop_width = int(face_width * 1.8)
        crop_height = int(face_height * 2.2)
        
        target_ratio = self.target_size[0] / self.target_size[1]
        current_ratio = crop_width / crop_height
        
        if current_ratio > target_ratio:
            crop_height = int(crop_width / target_ratio)
        else:
            crop_width = int(crop_height * target_ratio)
        
        left = center_x - (crop_width // 2)
        right = center_x + (crop_width // 2)
        top = center_y - (crop_height // 2) - (face_height // 4)
        bottom = top + crop_height
        
        left = max(0, left)
        right = min(width, right)
        top = max(0, top)
        bottom = min(height, bottom)
        
        cropped = image[top:bottom, left:right]
        if cropped.size > 0:
            resized = cv2.resize(cropped, self.target_size)
            return resized
        
        return image    
    
    def save_current_face(self):
        if self.face_folder_created_flag:
            if self.current_frame_faces_cnt == 1:
                if not self.out_of_range_flag:
                    self.ss_cnt += 1
                    image_filename = f"{self.input_id_char}_{self.input_name_char}_{self.input_department_char}_{self.ss_cnt}.jpg"
                    image_path = os.path.join(self.current_face_dir, image_filename)
                    
                    boxes, scores = detector.detect_one(self.current_frame_copy)
                    if boxes is not None:  # Check if a face was detected
                        bbox = boxes[0].astype(int)

                        cropped_face = self.crop_face(self.current_frame_copy, bbox, None) # No landmarks with ultralight

                        self.log_all["text"] = "\"" + image_filename + "\"" + " saved!"
                        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(image_path, cropped_face)
                        logging.info("%-40s %s", "Save into:", image_path)
                    else:
                        self.log_all["text"] = "No face detected!"
                else:
                    self.log_all["text"] = "Please do not out of range!"
            else:
                self.log_all["text"] = "No face or multiple faces detected!"
        else:
            self.log_all["text"] = "Please run step 2!"

    def get_frame(self):
        try:
            if self.cap is None or not self.cap.isOpened():
                print("Error: Camera is not initialized or opened")
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    return False, None
                
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 480))
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                print("Error: Could not read frame from camera")
                return False, None
                
        except cv2.error as e:
            print(f"OpenCV Error: {e}")
            return False, None
        except AttributeError as e:
            print(f"Attribute Error: {e}")
            return False, None
        except Exception as e:
            print(f"Unexpected Error: {e}")
            return False, None

    def process(self):
        ret, self.current_frame = self.get_frame()

        if ret:
            self.update_fps()

            self.current_frame_copy = self.current_frame.copy()
        
            boxes, scores = detector.detect_one(self.current_frame)

            if boxes is not None:  # Check if faces were detected
                self.current_frame_faces_cnt = len(boxes)
                self.label_face_cnt["text"] = str(self.current_frame_faces_cnt)


                for i, box in enumerate(boxes):
                    bbox = box.astype(int)
                    score = scores[i]

                    if score < DETECTION_THRESHOLD:
                        continue

                    # Draw bounding box
                    cv2.rectangle(self.current_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    
                    cv2.putText(self.current_frame, f"{score:.2f}", (bbox[0], bbox[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if i == 0: 
                        face_width = bbox[2] - bbox[0]
                        face_height = bbox[3] - bbox[1]
                        center_x = (bbox[0] + bbox[2]) // 2
                        center_y = (bbox[1] + bbox[3]) // 2
                        
                        crop_width = int(face_width * 1.8)
                        crop_height = int(face_height * 2.2)
                        
                        target_ratio = self.target_size[0] / self.target_size[1]
                        current_ratio = crop_width / crop_height
                        
                        if current_ratio > target_ratio:
                            crop_height = int(crop_width / target_ratio)
                        else:
                            crop_width = int(crop_height * target_ratio)
                            
                        preview_left = center_x - (crop_width // 2)
                        preview_right = center_x + (crop_width // 2)
                        preview_top = center_y - (crop_height // 2) - (face_height // 4)
                        preview_bottom = preview_top + crop_height
                        
                        cv2.rectangle(self.current_frame, 
                                    (int(preview_left), int(preview_top)), 
                                    (int(preview_right), int(preview_bottom)), 
                                    (255, 0, 0), 2)

                    if i == 0:
                        self.face_ROI_width_start = bbox[0]
                        self.face_ROI_height_start = bbox[1]
                        self.face_ROI_width = bbox[2] - bbox[0]
                        self.face_ROI_height = bbox[3] - bbox[1]
                        self.hh = int(self.face_ROI_height / 2)
                        self.ww = int(self.face_ROI_width / 2)

                        if (bbox[2] + self.ww) > 640 or (bbox[3] + self.hh > 480) or \
                        (bbox[0] - self.ww < 0) or (bbox[1] - self.hh < 0):
                            self.label_warning["text"] = "OUT OF RANGE"
                            self.label_warning['fg'] = 'red'
                            self.out_of_range_flag = True
                        else:
                            self.out_of_range_flag = False
                            self.label_warning["text"] = ""
                            
            else:
                self.current_frame_faces_cnt = 0
                self.label_face_cnt["text"] = "0"

            img_Image = Image.fromarray(self.current_frame)
            img_PhotoImage = ImageTk.PhotoImage(image=img_Image)
            self.label.img_tk = img_PhotoImage 
            self.label.configure(image=img_PhotoImage)

        self.win.after(20, self.process)

    def run(self):
        self.pre_work_mkdir()
        self.check_existing_faces_cnt()
        self.GUI_info()
        self.process()
        self.win.mainloop()

def main():
    logging.basicConfig(level=logging.INFO)
    Face_Register_con = Face_Register()
    Face_Register_con.run()


if __name__ == '__main__':
    main()
