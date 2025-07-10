## 🔍 Overview of `tracking.py`

The `tracking.py` script is the core component of this vehicle tracking system. It detects vehicles in real-time from a CCTV video stream using the **Faster R-CNN** object detection model and tracks them using the **SORT (Simple Online and Realtime Tracking)** algorithm.

### 🔧 Technologies Used

- **PyTorch**: For loading and running the object detection model.
- **Torchvision**: Provides pretrained models including Faster R-CNN.
- **OpenCV**: For capturing and processing video frames.
- **NumPy**: For numerical operations.
- **SORT**: Lightweight real-time tracking algorithm based on Kalman Filter and data association.

---

### ⚙️ Workflow

1. **Initialization**:
   - Loads the `fasterrcnn_resnet50_fpn` model with pretrained COCO weights.
   - Opens a CCTV stream via an HLS `.m3u8` URL using OpenCV.

2. **Processing Video Frames**:
   - Reads frames from the video stream and resizes them for efficiency.
   - Converts frames from BGR to RGB, then to PyTorch tensors.
   - Passes the image tensor to the model to get predictions (bounding boxes, scores, and labels).

3. **Filtering Detections**:
   - Filters detections to only keep vehicles (`car`, `motorcycle`, `bus`, `truck`) with confidence scores above 0.5.
   - Prepares the bounding boxes and scores in a format required by the tracker.

4. **Object Tracking**:
   - Sends detections to the `Sort` tracker, which assigns unique IDs to tracked objects across frames.

5. **Vehicle Counting**:
   - Keeps track of unique vehicle IDs to ensure each vehicle is only counted once.
   - Displays the running total of detected vehicles.

6. **Visualization**:
   - Draws bounding boxes and IDs on the video frames.
   - Displays the current vehicle count at the top-left corner.
   - Press `q` to quit the video stream display.

---

### 📺 Output

- Live video stream with green bounding boxes and unique IDs for each vehicle.
- Vehicle count displayed on-screen with the text:  
  **"Vehicle Count: X"**

---

### 📌 Notes

- Default CCTV stream URL:
https://cctvkanjeng.gresikkab.go.id/stream/10.120.0.117/index-10.120.0.117.m3u8

You can replace this with another `.m3u8` stream URL if needed.

- Press **`q`** during video display to stop the program and release resources.

