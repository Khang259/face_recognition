# import torch
# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.cuda.current_device())
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))

import insightface
import onnxruntime

# Kiểm tra xem ONNX có nhận GPU không
print(onnxruntime.get_device()) # Phải trả về 'GPU'
print(onnxruntime.get_available_providers()) # Phải có 'CUDAExecutionProvider'

# Khởi tạo model với GPU
from insightface.app import FaceAnalysis
app = FaceAnalysis(providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0) # ctx_id=0 là GPU đầu tiên