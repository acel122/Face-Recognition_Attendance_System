# Face Recognition Attendance System

An **automatic face recognition-based attendance system** designed to improve employee time tracking and operational efficiency. The system supports **real-time multi-face recognition** integrated with **CCTV** and provides a **web-based dashboard** for monitoring and reporting.

---

## 🚀 Key Features
- 🔍 **Face Registration** using **Ultra Light Face Detector**
- ⚡ **Real-time Face Recognition** powered by **InsightFace (buffalo_l)**
- 🗄️ **PostgreSQL Database** for attendance data storage
- 🌐 **Laravel Dashboard** for real-time monitoring and reporting
- 🖼️ **Image Preprocessing & Embedding** to improve recognition accuracy
- 📡 **RTSP Integration** for CCTV video streaming
- ⏱️ Average face detection & recognition time: **~10ms per face**

---

## 🛠️ Tech Stack
- **Python** → Face detection, preprocessing, embedding, recognition  
- **InsightFace (buffalo_l)** → Pre-trained face recognition model (MIT License)  
- **Ultra Light Face Detector** → Lightweight face detection for registration (MIT License)  
- **Laravel** → Web-based dashboard and backend  
- **PostgreSQL** → Relational database for attendance records  
- **RTSP** → CCTV video streaming integration  

---

## 📂 Dataset
- **128 individuals**  
- **7 images per person** (various angles: frontal, left/right profile, ¾ views, slightly up/down)  
- Images are preprocessed and converted into **embedding vectors** for recognition  

---

## 🔄 System Workflow
1. **Face Registration** → Capture and register faces using Ultra Light Face Detector  
2. **Database Entry** → Store identity data into PostgreSQL  
3. **Preprocessing** → Resize, normalize, and prepare images  
4. **Embedding** → Convert images into feature vectors using buffalo_l  
5. **Face Recognition** → Match embeddings with database identities  
6. **Data Storage** → Save attendance logs into PostgreSQL  
7. **Dashboard Visualization** → Display attendance data via Laravel dashboard  

---

## 📜 License

This project is licensed under the MIT License.
It also makes use of the following models/tools under MIT License:
InsightFace (buffalo_l): https://github.com/deepinsight/insightface
Ultra Light Face Detector: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
See the LICENSE file for more details.
