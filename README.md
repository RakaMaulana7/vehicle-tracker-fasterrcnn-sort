## 🔍 Penjelasan tracking.py

File `tracking.py` merupakan inti dari sistem pelacakan kendaraan secara real-time. Skrip ini mendeteksi kendaraan dari stream CCTV dan melacaknya menggunakan algoritma SORT (Simple Online and Realtime Tracking).

### 🔧 Teknologi yang Digunakan

- **PyTorch**: Untuk memuat dan menjalankan model deteksi objek Faster R-CNN.
- **Torchvision**: Menyediakan model deteksi pretrained.
- **OpenCV**: Untuk pengambilan dan pemrosesan frame video.
- **NumPy**: Untuk manipulasi array numerik.
- **SORT**: Algoritma pelacakan real-time berbasis Kalman Filter.

---

### ⚙️ Alur Kerja tracking.py

1. **Inisialisasi:**
   - Model `fasterrcnn_resnet50_fpn` diload dengan bobot pretrained dari dataset COCO.
   - Stream CCTV dibuka dari URL HLS (`.m3u8`) menggunakan OpenCV.

2. **Frame Processing Loop:**
   - Setiap frame dari CCTV dibaca dan diperkecil ukurannya.
   - Frame dikonversi dari BGR ke RGB lalu diubah menjadi tensor.
   - Diberikan ke model deteksi untuk menghasilkan prediksi (bounding box, skor, dan label objek).

3. **Filtering Deteksi:**
   - Hanya objek dengan label kendaraan (car, motorcycle, bus, truck) dan skor > 0.5 yang diproses.
   - Koordinat bounding box + confidence dimasukkan ke dalam array deteksi.

4. **Pelacakan:**
   - Deteksi dikirim ke tracker `Sort` untuk dilacak antar frame.
   - Setiap objek diberi ID unik.

5. **Hitung Kendaraan:**
   - Jika ID kendaraan belum pernah muncul, akan dihitung satu kali.
   - Jumlah total kendaraan ditampilkan di layar.

6. **Visualisasi:**
   - Bounding box dan ID objek ditampilkan di atas frame.
   - Total kendaraan ditampilkan di kiri atas layar.
   - Tekan tombol `q` untuk keluar dari tampilan.

---

### 📺 Output

- Frame video real-time dengan kotak hijau dan ID di setiap kendaraan.
- Teks "Vehicle Count: X" di kiri atas layar menunjukkan total kendaraan terdeteksi dan dihitung.

---

### 📌 Catatan

- URL CCTV default adalah:
https://cctvkanjeng.gresikkab.go.id/stream/10.120.0.117/index-10.120.0.117.m3u8

Kamu bisa menggantinya dengan stream CCTV lain (pastikan mendukung `.m3u8` HLS).

- Untuk menghentikan proses, tekan tombol `q` saat tampilan video aktif.
