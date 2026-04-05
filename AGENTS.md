- Always be DRY and clean code
- If you be asked or plan to do something make sure it can be scalability
- Architecture workflow of this project 

Webcam → ai_inference (detect + embed) → POST /api/faces/verify
                                              ↓
                                     backend: Qdrant similarity search
                                              ↓
                                     cosine distance < threshold?
                                     ✅ YES → trả JWT token → vào app
                                     ❌ NO  → reject hoặc mời đăng ký

- Tech stack
Detection: YOLOv8-face
Alignment: (optional hoặc dùng landmark YOLO)
Embedding: ArcFace (ONNX) (pull từ Hugging Face)
Database: 
+ Qdrant cho face embeddings similarity search cực nhanh
+ PostgreSQL cho user data, session, auth logs
Backend: 
+ FastAPI + SQLAlchemy + Alembic cho migrations
+ python-jose cho JWT token sau khi authenticate thành công
Frontend: React (WebRTC)

Frontend:
+ React+ Vite: nhẹ, HMR nhanh cho development
+ Gọi thẳng api từ backend để stream frames

- Using GPU for infer or decode frames 