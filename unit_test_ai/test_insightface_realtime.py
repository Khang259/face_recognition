import cv2
import time
from insightface.app import FaceAnalysis

# 1. Khởi tạo InsightFace với GPU
# buffalo_l là bộ model chuẩn gồm: detection, landmark, genderage, recognition
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 2. Mở Camera (0 là camera mặc định)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở Camera.")
    exit()

print("Bắt đầu thực hiện nhận diện. Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_t0 = time.perf_counter()

    # 3. Chạy Detection và Embedding (đo thời gian inference — ORT/CUDA thường sync khi trả kết quả)
    infer_t0 = time.perf_counter()
    faces = app.get(frame)
    infer_ms = (time.perf_counter() - infer_t0) * 1000.0

    for i, face in enumerate(faces):
        # Lấy tọa độ Bounding Box
        bbox = face.bbox.astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Lấy Embedding Vector (ArcFace)
        embedding = face.embedding
        
        # In thông tin vector ra console
        print(f"Face {i+1}: Embedding Shape = {embedding.shape} | Vector (5 giá trị đầu) = {embedding[:5]}")

        # Hiển thị thông tin lên khung hình
        cv2.putText(frame, f"Face {i+1}: {embedding.shape}", (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    fps = 1.0 / (time.perf_counter() - frame_t0)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"Infer GPU: {infer_ms:.1f} ms",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 200, 255),
        2,
    )

    # 4. Hiển thị kết quả
    cv2.imshow('InsightFace Real-time Testing', frame)

    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()