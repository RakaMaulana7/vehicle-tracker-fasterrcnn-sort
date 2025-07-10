import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from sort.sort import Sort  # Pastikan file sort.py tersedia

# Menghindari error duplikasi library
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Gunakan GPU jika tersedia
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Load model Faster R-CNN dengan pretrained COCO weights
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
model = model.to(device)
model.eval()

# Transformasi gambar menjadi tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# ID kendaraan berdasarkan COCO dataset
vehicle_ids = {3, 4, 6, 8}  # car, motorcycle, bus, truck
confidence_threshold = 0.5

# URL stream CCTV
stream_url = "https://cctvkanjeng.gresikkab.go.id/stream/10.120.0.117/index-10.120.0.117.m3u8"
cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error: Cannot open stream")
    exit()

# Inisialisasi tracker SORT
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)

# Variabel untuk menyimpan ID kendaraan yang sudah dihitung
counted_ids = set()
vehicle_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perkecil ukuran frame untuk efisiensi
    frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Konversi BGR ke RGB untuk model deteksi
    frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    img_tensor = transform(frame_rgb).to(device)

    # Deteksi objek menggunakan model
    with torch.no_grad():
        predictions = model([img_tensor])[0]

    # Filter deteksi kendaraan berdasarkan skor dan label
    detections = []
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score > confidence_threshold and label.item() in vehicle_ids:
            x1, y1, x2, y2 = box.cpu().numpy()
            detections.append([x1, y1, x2, y2, score.item()])

    dets = np.array(detections) if detections else np.empty((0, 5))

    # Update tracker dan dapatkan objek yang sedang dilacak
    tracks = tracker.update(dets)

    for track in tracks:
        x1, y1, x2, y2, track_id = track
        track_id = int(track_id)

        # Gambar kotak dan ID pada frame
        cv2.rectangle(frame_small, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame_small, f'ID:{track_id}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Tambahkan ke hitungan jika ID belum pernah dihitung
        if track_id not in counted_ids:
            counted_ids.add(track_id)
            vehicle_count += 1

    # Tampilkan jumlah kendaraan
    cv2.putText(frame_small, f"Vehicle Count: {vehicle_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow("Frame", frame_small)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan sumber daya
cap.release()
cv2.destroyAllWindows()
