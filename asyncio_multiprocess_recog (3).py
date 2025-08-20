import os
import asyncio
import threading
import logging
import time
import albumentations as A
from dataclasses import dataclass
from typing import Dict
import multiprocessing as mproc
import queue as queue_module
from datetime import datetime
import pytz
from collections import defaultdict, deque, Counter
from sklearn.neighbors import NearestNeighbors

import numpy as np
import psycopg2
from insightface.app import FaceAnalysis
import pickle
import torch
import torch.nn.functional as F

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import cv2 as cv

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('face_recognition_debug.log')
    ]
)

@dataclass
class PerformanceMetrics:
    fps: float = 0.0
    detection_time: float = 0.0
    recognition_time: float = 0.0
    total_frames: int = 0
    faces_detected: int = 0
    successful_recognitions: int = 0
    failed_recognitions: int = 0
    average_confidence: float = 0.0
    confidence_variance: float = 0.0
    low_confidence_matches: int = 0

class PerformanceMonitor:
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.confidence_scores = []
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.faces_processed = 0
        self.confidence_threshold = 0.6
        
    def start_frame(self):
        return time.time()
    
    def update_detection_metrics(self, faces: list, detection_time: float):
        with self.lock:
            self.metrics.total_frames += 1
            self.metrics.faces_detected += len(faces)
            self.metrics.detection_time += detection_time
            self.faces_processed += len(faces)
        
    def update_recognition_metrics(self, valid_matches, recognition_time):
        with self.lock:
            self.metrics.recognition_time += recognition_time

            successful = len([m for m in valid_matches if m[1] >= self.confidence_threshold])
            self.metrics.successful_recognitions += successful
            self.metrics.failed_recognitions += (len(valid_matches) - successful)

            if valid_matches:
                scores = [m[1] for m in valid_matches]
                self.confidence_scores.extend(scores)
                self.metrics.average_confidence = np.mean(scores)
                self.metrics.confidence_variance = np.var(scores)
                self.metrics.low_confidence_matches += len([s for s in scores if s < self.confidence_threshold])

    def should_log_metrics(self):
        return self.faces_processed >= 100
    
    def reset_faces_processed(self):
        self.faces_processed = 0

    def get_metrics(self) -> Dict:
        with self.lock:
            elapsed_time = time.time() - self.start_time

            return {
                "FPS": self.metrics.total_frames / elapsed_time if elapsed_time > 0 else 0,
                "Average Detection Time (ms)": (self.metrics.detection_time / self.metrics.total_frames) * 1000 if self.metrics.total_frames > 0 else 0,
                "Average Recognition Time (ms)": (self.metrics.recognition_time / self.metrics.total_frames) * 1000 if self.metrics.total_frames > 0 else 0,
                "Total Faces Detected": self.metrics.faces_detected,
                "Recognition Rate (%)": (self.metrics.successful_recognitions /
                                    (self.metrics.successful_recognitions + self.metrics.failed_recognitions)
                                    if (self.metrics.successful_recognitions + self.metrics.failed_recognitions) > 0
                                    else 0),
                "Average Confidence Score": self.metrics.average_confidence,
                "Confidence Variance": self.metrics.confidence_variance,
                "Low Confidence Matches": self.metrics.low_confidence_matches
            }

    def log_metrics(self):
        metrics = self.get_metrics()
        logging.info("Performance Metrics:")
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.2f}")
        self.reset_faces_processed()

DB_USER = DB_USER 
DB_PASSWORD = DB_PASSWORD
DB_NAME = DB_NAME
DB_HOST = DB_HOST

RTSP_URLS = ["rtspurl"]

class AsyncFaceRecognizerProcess:
    def __init__(self, embeddings_file="data/face_embeddings.pkl",
                similarity_threshold=0.5, database_config=None):
        self.app = FaceAnalysis(name='buffalo_l', 
                                providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
        
        self.embeddings_file = embeddings_file
        self.similarity_threshold = similarity_threshold
        self.knn_match_threshold = similarity_threshold
        self.database_config = database_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        torch.cuda.empty_cache()
        
        self.temporal_cache = defaultdict(lambda: deque(maxlen=5))
        self.embedding_indices = defaultdict(list)
        
        self.load_embeddings()
        self._prepare_embeddings_tensor()
    
        self.conn = None
        self.cursor = None
        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.connect_to_db()
        
        self.performance_monitor = PerformanceMonitor()

        self.frame_cnt = 0
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        self.bbox_history = {}
        self.bbox_persistence_frames = 12
        self.loop = None
        
        self.recording = False
        self.video_writer = None
        self.recording_start_time = None
        self.output_directory = "data/recorded_videos"
        self.max_video_duration = 14400
        os.makedirs(self.output_directory, exist_ok=True)
    
    def _prepare_embeddings_tensor(self):
        if not hasattr(self, 'all_embeddings'):
            raise ValueError("Embeddings not loaded! Call load_embeddings() first")
            
        self.embeddings_tensor = torch.tensor(
            self.all_embeddings, 
            device=self.device
        ).float()
        self.embeddings_tensor = F.normalize(self.embeddings_tensor, p=2, dim=1)
        
        if self.device.type == 'cuda':
            mem_usage = torch.cuda.memory_allocated() / 1024**3
            logging.info(f"GPU Memory used by embeddings: {mem_usage:.2f}GB")
        
    def start_recording(self, frame_shape, fps=20.0):
        """Start recording video stream"""
        if self.recording:
            return
        
        try:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_directory, f"recording_{current_time}.mp4")
            
            codecs = [('XVID', '.avi')]
            
            for codec, ext in codecs:
                    if ext != '.mp4':
                        output_path = output_path.rsplit('.', 1)[0] + ext
                    
                    fourcc = cv.VideoWriter_fourcc(*codec)
                    height, width = frame_shape[:2]
                    
                    writer = cv.VideoWriter(
                        output_path,
                        fourcc,
                        fps,
                        (width, height),
                        True  
                    )
                    
                    if writer.isOpened():
                        self.video_writer = writer
                        self.recording = True
                        self.recording_start_time = time.time()
                        logging.info(f"Started recording to {output_path} using codec {codec}")
                        break
                    else:
                        writer.release()
                
            if not self.recording:
                raise Exception("Could not initialize video writer with any codec")
                
        except Exception as e:
            logging.error(f"Error starting recording: {e}")
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None

    def stop_recording(self):
        if not self.recording:
            return
        try:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            
            self.recording = False
            self.recording_start_time = None
            logging.info("Stopped recording")
        except Exception as e:
            logging.error(f"Error stopping recording: {e}")
        finally:
            self.recording = False
            self.video_writer = None
            self.recording_start_time = None

    def check_recording_duration(self):
        if not self.recording or not self.recording_start_time:
            return
        
        current_duration = time.time() - self.recording_start_time
        if current_duration >= self.max_video_duration:
            self.stop_recording()
            frame_shape = (576, 704, 3) 
            self.start_recording(frame_shape)

    def load_embeddings(self):
        try:
            with open(self.embeddings_file, 'rb') as f:
                data = pickle.load(f)
                self.all_embeddings = data['all_embeddings']
                self.label_map = data['label_map']
            
            if len(self.all_embeddings) == 0:
                raise ValueError("No embeddings found in file!")
            if len(self.all_embeddings[0]) != 512:
                raise ValueError(f"Unexpected embedding dimension: {len(self.all_embeddings[0])}")
            
            self.embedding_indices = defaultdict(list)
            for idx, name in enumerate(self.label_map):
                self.embedding_indices[name].append(idx)
            
            logging.info(f"Loaded {len(self.all_embeddings)} embeddings for {len(self.embedding_indices)} identities")
            logging.debug(f"Sample embedding: {self.all_embeddings[0][:5]}...") 
        
        except Exception as e:
            logging.error(f"Error loading embeddings: {e}")
            raise
        
    def knn_match(self, embedding, k=5, threshold=0.5):
        try:
            query_tensor = torch.tensor(embedding, device=self.device).float()
            query_tensor = F.normalize(query_tensor.unsqueeze(0), p=2, dim=1)
            
            similarities = torch.mm(query_tensor, self.embeddings_tensor.t()).squeeze(0)
            
            top_k_values, top_k_indices = torch.topk(similarities, k=k)
            
            valid_matches = []
            for score, idx in zip(top_k_values.cpu().numpy(), top_k_indices.cpu().numpy()):
                if score > threshold:
                    valid_matches.append((self.label_map[idx], score))
        
            return valid_matches
        
        except Exception as e:
            logging.error(f"Error performing KNN match: {e}")
            return []

    def connect_to_db(self):
        try:
            self.conn = psycopg2.connect(**self.database_config)
            self.cursor = self.conn.cursor()

            self.cursor.execute("SELECT version();")
            version = self.cursor.fetchone()
            logging.info(f"Connected to PostgreSQL database: {version[0]}")

            self.cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'fr_attendance'
                );
            """)
            table_exists = self.cursor.fetchone()[0]
            if not table_exists:
                logging.error("Table 'fr_attendance' does not exist!")
            else:
                logging.info("Found attendance table successfully")

        except Exception as e:
            logging.error(f"Unable to connect to the database: {e}")
            raise

    def close(self):
        self.stop_recording()
        self.close_db()
        cv.destroyAllWindows()
        logging.info("All connections closed")

    def close_db(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def attendance(self, id_number):
        try:
            jakarta_tz = pytz.timezone('Asia/Jakarta')
            current_time = datetime.now(jakarta_tz)
            logging.info(f"Processing attendance for ID {id_number} at {current_time.isoformat()}")

            self.cursor.execute("SELECT record_attendance(%s)", (id_number,))
            result = self.cursor.fetchone()[0]
            
            if self.cursor.rowcount == 1:
                print("Success: Only one record inserted.")
            else:
                print("Error: More than one record inserted!")
            
            self.conn.commit()
            logging.info(f"Attendance result for ID {id_number}: {result}")
            return result

        except psycopg2.Error as e:
            logging.error(f"Database error recording attendance: {e.pgcode} - {e.pgerror}")
            self.conn.rollback()
            if e.pgcode in ('08000', '08003', '08006', '08001', '08004'):
                logging.info("Attempting to reconnect to database...")
                self.connect_to_db()
                self.cursor.execute("SELECT record_attendance(%s)", (id_number,))
                result = self.cursor.fetchone()[0]
                self.conn.commit()
                return result
        except Exception as e:
            logging.error(f"Unexpected error recording attendance: {e}")
            self.conn.rollback()
            return f"Error recording attendance: {str(e)}"

    async def process_rtsp_stream(self, rtsp_url, queue):
        logging.info(f"Attempting to open RTSP stream: {rtsp_url}")
        
        def capture_frames(cap, queue):
            while True:
                ret, frame = cap.read()
                if not ret:
                    logging.warning(f"Failed to read frame from {rtsp_url}")
                    break
                    
                try:
                    if queue.full():
                        while not queue.empty():
                            try:
                                queue.get_nowait()
                            except queue_module.Empty:
                                break
                    queue.put_nowait((rtsp_url, frame))
                except Exception as e:
                    logging.error(f"Error putting frame in queue: {e}")
                    break
                
                time.sleep(0.001)
        
        while True:
            try:
                cap = cv.VideoCapture(rtsp_url, cv.CAP_FFMPEG)
                
                if not cap.isOpened():
                    cap = cv.VideoCapture(rtsp_url, cv.CAP_GSTREAMER)
                
                if cap.isOpened():
                    logging.info(f"Successfully opened stream: {rtsp_url}")
                    logging.info("Stream properties:")
                    logging.info(f"Width: {cap.get(cv.CAP_PROP_FRAME_WIDTH)}")
                    logging.info(f"Height: {cap.get(cv.CAP_PROP_FRAME_HEIGHT)}")
                    logging.info(f"FPS: {cap.get(cv.CAP_PROP_FPS)}")
                    
                    cap.set(cv.CAP_PROP_BUFFERSIZE, 3)
                    
                    capture_thread = threading.Thread(
                        target=capture_frames,
                        args=(cap, queue),
                        daemon=True
                    )
                    capture_thread.start()
                    
                    while capture_thread.is_alive():
                        await asyncio.sleep(1)
                        
                    logging.error(f"Capture thread died for {rtsp_url}")
                    cap.release()
                    
                else:
                    logging.error(f"Failed to open stream: {rtsp_url}")
                    await asyncio.sleep(2)
                    
            except Exception as e:
                logging.error(f"Error in RTSP stream processing: {e}")
                if cap:
                    cap.release()
                await asyncio.sleep(2)
        

    def update_fps(self, frame):
        now = time.time()
        frame_time = now - frame
        if frame_time != 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / frame_time)
        return now

    def process_frame(self, frame):
        frame_start_time = time.time()
        
        frame = cv.resize(frame, (704, 576))
        
        if not self.recording:
            self.start_recording(frame.shape)
        self.check_recording_duration()
        
        if cv.cuda.getCudaEnabledDeviceCount() > 0:
            gpu_frame = cv.cuda_GpuMat()
            gpu_frame.upload(frame)
            rgb_frame = cv.cuda.cvtColor(gpu_frame, cv.COLOR_BGR2RGB).download()
        else:
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        detection_start = time.time()
        faces = self.app.get(rgb_frame)
        detection_time = time.time() - detection_start
        self.performance_monitor.update_detection_metrics(faces, detection_time)

        if faces:
            recognition_start = time.time()
            matched_embeddings = []

            for face in faces:
                display_name = "Unknown"
                similarity_score = 0.0
                matched_name = None

                try:
                    embedding = face.embedding
                    logging.debug(f"Embedding shape: {embedding.shape}")
                    
                    valid_matches = self.knn_match(embedding, k=5, threshold=self.knn_match_threshold)

                    if valid_matches:
                        vote_weights = defaultdict(float)
                        for name, score in valid_matches:
                            vote_weights[name] += score
                        
                        best_match, total_score = max(vote_weights.items(), 
                                                    key=lambda x: x[1], 
                                                    default=(None, 0))
                        
                        face_key = face.bbox.tobytes()
                        self.temporal_cache[face_key].append(best_match)
                        
                        final_name = Counter(self.temporal_cache[face_key]).most_common(1)[0][0]
                        similarity_score = total_score / len(valid_matches)
                        
                        if similarity_score > self.similarity_threshold:
                            matched_name = final_name
                            try:
                                id_number, name, department = matched_name.split('_')
                                self.attendance(id_number)
                                display_name = f"{name} ({similarity_score:.2f})"
                            except ValueError:
                                display_name = "Unknown (Format Error)"
                        else:
                            display_name = f"Unknown ({similarity_score:.2f})"
                
                    logging.debug(f"Best match similarity score: {similarity_score}")
                    
                    bbox = face.bbox.astype(int)
                    cv.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (201, 202, 247), 2)
                    cv.putText(frame, display_name, (bbox[0], bbox[3] + 20), self.font, 0.8, (245, 242, 245), 2)
                    
                    logging.debug(f"Processed face: {display_name}")
                    
                except AttributeError as e:
                    logging.error(f"Attribute error during recognition (likely missing embedding): {e}")
                    continue
                except Exception as e:
                    logging.error(f"Recognition Error: {e}")
                    display_name = "Unknown (Error)"
            
            self.performance_monitor.update_recognition_metrics(
                valid_matches if 'valid_matches' in locals() else [],
                time.time() - recognition_start
            )

        if self.recording and self.video_writer is not None:
            try:
                self.video_writer.write(frame)
            except Exception as e:
                logging.error(f"Error writing frame to video: {e}")
                self.stop_recording()
                self.start_recording(frame.shape)
                
        frame_start_time = self.update_fps(frame_start_time)
        cv.putText(frame, f"FPS: {self.fps:.2f}", (10, 30), self.font, 0.8, (0, 255, 0), 2)
        
        if self.recording:
            cv.putText(frame, "REC", (10, 60), self.font, 0.8, (0, 0, 255), 2)

        return frame

    async def run_async(self, rtsp_url, queue):
        self.loop = asyncio.get_running_loop()
        
        stream_task = asyncio.create_task(self.process_rtsp_stream(rtsp_url, queue))
        
        try:
            while True:
                try:
                    try:
                        frame_with_source = await self.loop.run_in_executor(
                            None,
                            lambda: queue.get(timeout=1.0)
                        )
                    except queue_module.Empty:
                        continue
                    
                    rtsp_url, frame = frame_with_source
                    if frame is None:
                        continue
                    
                    processed_frame = await self.loop.run_in_executor(
                        None,
                        self.process_frame,
                        frame
                    )
                    
                    if processed_frame is None:
                        continue
                    
                    window_name = f'Face Recognition - {rtsp_url}'
                    cv.imshow(window_name, processed_frame)
                    
                    if self.performance_monitor.should_log_metrics():
                        self.performance_monitor.log_metrics()
                    
                    key = cv.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'): 
                        if self.recording:
                            self.stop_recording()
                        else:
                            self.start_recording(frame.shape)
                    
                except Exception as e:
                    logging.error(f"Error in frame processing loop: {e}")
                    await asyncio.sleep(1)
                    
        except asyncio.CancelledError:
            pass
        finally:
            stream_task.cancel()
            try:
                await stream_task
            except asyncio.CancelledError:
                pass
            self.close()
            logging.info("Exiting Program...")

def process_stream(rtsp_url, embeddings_file, similarity_threshold, database_config, queue):
    logging.info(f"Starting process for RTSP URL: {rtsp_url}")
    try:
        recognizer = AsyncFaceRecognizerProcess(embeddings_file, similarity_threshold, database_config)
        asyncio.run(recognizer.run_async(rtsp_url, queue))
    except Exception as e:
        logging.error(f"Error in process for {rtsp_url}: {e}")
    finally:
        logging.info(f"Process finished for RTSP URL: {rtsp_url}")
                    
async def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        processes = []
        queues = []
        logging.info(f"Processing {len(RTSP_URLS)} RTSP URLs:")
        
        for url in RTSP_URLS:
            queue = mproc.Queue(maxsize=100)
            queues.append(queue)
            p = mproc.Process(
                target=process_stream,
                args=(
                    url,
                    "data/face_embeddings.pkl",
                    0.5,
                    {
                        'database': DB_NAME,
                        'user': DB_USER,
                        'password': DB_PASSWORD,
                        'host': DB_HOST,
                        'port': 5432
                    },
                    queue
                )
            )
            processes.append(p)
            p.start()

        while any(p.is_alive() for p in processes):
            await asyncio.sleep(1)

        for p in processes:
            p.join()

    except Exception as e:
        logging.error(f"An unexpected error occurred in main: {e}")
        
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Program stopped by user.")
    except Exception as e:
        logging.critical(f"A critical error occurred: {e}")
