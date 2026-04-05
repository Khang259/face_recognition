import cv2
import numpy as np
import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from insightface.app import FaceAnalysis

# --- CẤU HÌNH ---
os.environ['ORT_LOGGING_LEVEL'] = '3'
logging.basicConfig(level=logging.WARNING)

COLLECTION_NAME = "factory_staff"
# Để lưu trữ lâu dài trên ổ D, hãy đổi thành: QdrantClient(path="D:/qdrant_data")
qclient = QdrantClient(":memory:") 

if not qclient.collection_exists(COLLECTION_NAME):
    qclient.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )

app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# --- HÀM ĐĂNG KÝ (Lưu kèm đường dẫn ảnh) ---
def register_face(name, img_path):
    img = cv2.imread(img_path)
    if img is None: return
    faces = app.get(img)
    if len(faces) > 0:
        embedding = faces[0].embedding.tolist()
        # Lưu path vào payload để sau này load ảnh hiển thị
        qclient.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(
                id=np.random.randint(1, 999999), 
                vector=embedding, 
                payload={"name": name, "img_path": img_path}
            )]
        )
        print(f"✅ Đã đăng ký: {name}")

# TEST: Đăng ký một vài ảnh mẫu (Hãy đảm bảo file tồn tại)
register_face("Admin", "unit_test_ai/test.jpg")

# --- CHƯƠNG TRÌNH CHÍNH ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Tạo một khung đen bên phải để hiển thị ảnh mẫu (ảnh so sánh)
    # Kích thước: giữ nguyên chiều cao frame, chiều rộng 300px
    h, w, _ = frame.shape
    side_panel = np.zeros((h, 300, 3), dtype=np.uint8)
    
    faces = app.get(frame)
    
    for face in faces:
        bbox = face.bbox.astype(int)
        query_vector = face.embedding.tolist()

        try:
            search_result = qclient.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=1,
                score_threshold=0.45
            ).points
        except:
            search_result = []

        if search_result:
            res = search_result[0]
            name = res.payload.get('name', 'Unknown')
            match_img_path = res.payload.get('img_path', '')
            score = res.score
            
            color = (0, 255, 0) # Xanh - Hợp lệ
            
            # Hiển thị ảnh mẫu từ Database lên side_panel
            if os.path.exists(match_img_path):
                db_img = cv2.imread(match_img_path)
                # Resize ảnh database cho khớp với khung side_panel
                db_img_resized = cv2.resize(db_img, (280, 280))
                side_panel[50:330, 10:290] = db_img_resized
                cv2.putText(side_panel, "DATABASE MATCH", (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(side_panel, f"Score: {score:.2f}", (10, 360), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            label = f"{name} ({score:.2f})"
        else:
            label = "Unknown"
            color = (0, 0, 255)
            cv2.putText(side_panel, "NO MATCH", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Vẽ lên Frame camera
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Ghép 2 cửa sổ lại thành 1 (Hconcat)
    combined_view = np.hstack((frame, side_panel))
    
    cv2.imshow('Face Recognition Auth - Comparison View', combined_view)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()