import os
import cv2
import threading
import tkinter as tk
from collections import Counter
from ultralytics import YOLO

# ─── Configuration ─────────────────────────────────────────────────────────────

os.environ["YOLO_VERBOSE"] = "0"        # suppress YOLO console logs
MODEL_PATH   = "yolo11n.pt"       # your .pt file
CAMERA_INDEX = 2                   # adjust if your webcam isn't at index 0

# ─── Load Model ────────────────────────────────────────────────────────────────

model = YOLO(MODEL_PATH)

# ─── Shared State ─────────────────────────────────────────────────────────────

# this will accumulate counts across all frames
global_counts = Counter()

# flag to signal threads to stop
stop_event = threading.Event()

# ─── Detection Thread ─────────────────────────────────────────────────────────

def detection_loop():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        stop_event.set()
        return

    cv2.namedWindow("YOLO Live", cv2.WINDOW_NORMAL)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Failed to grab frame")
            break

        # run inference in streaming mode
        for result in model(frame, stream=True, verbose=False):
            annotated = result.plot()  # get frame with boxes drawn

            # update cumulative counts
            cls_ids = result.boxes.cls.int().tolist()
            names   = [result.names[i] for i in cls_ids]
            global_counts.update(names)

        # show live
        cv2.imshow("YOLO Live", annotated)

        # quit if 'q' pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()
    stop_event.set()

# ─── Tkinter GUI ────────────────────────────────────────────────────────────────

class CountWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Cumulative Object Counts")
        self.label = tk.Label(root, text="Waiting for detections…", font=("Helvetica", 14))
        self.label.pack(padx=10, pady=10)
        # poll every 500ms
        self.poll()

    def poll(self):
        # build display text from global_counts
        if global_counts:
            lines = [f"{cls_name}: {cnt}" for cls_name, cnt in global_counts.items()]
            text = "\n".join(lines)
        else:
            text = "No objects detected yet."
        self.label.config(text=text)
        # schedule next poll
        if not stop_event.is_set():
            self.root.after(500, self.poll)
        else:
            self.root.quit()

# ─── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1) Start YOLO detection in background
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()

    # 2) Launch tkinter window (blocks until closed)
    root = tk.Tk()
    app = CountWindow(root)
    root.mainloop()

    # 3) Clean exit
    stop_event.set()
    t.join()
    print("Exiting.")
