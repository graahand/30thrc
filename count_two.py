import cv2
import threading
import tkinter as tk
from collections import Counter
from ultralytics import YOLO
import queue

MODEL_PATH = "yolo11n.pt"
CAMERA_INDICES = [0, 2]
CONFIG_FILE = "allowed_objects.txt"

def load_allowed_objects(file_path):
    """Load allowed object names from configuration file"""
    allowed = set()
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    allowed.add(line)
        print(f"Loaded {len(allowed)} allowed object types")
        return allowed
    except FileNotFoundError:
        print(f"Config file {file_path} not found. Allowing all objects.")
        return None

model = YOLO(MODEL_PATH)
allowed_objects = load_allowed_objects(CONFIG_FILE)

data_lock = threading.Lock()
frame_queues = {}
seen_classes = {idx: set() for idx in CAMERA_INDICES}
counts = {idx: Counter() for idx in CAMERA_INDICES}
stop_event = threading.Event()

def camera_thread(camera_idx):
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        return
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        
        results = model(frame, verbose=False)
        if results and len(results) > 0:
            result = results[0]
            annotated_frame = result.plot() if result.boxes is not None else frame
            
            if result.boxes is not None and result.boxes.cls is not None:
                cls_ids = result.boxes.cls.int().tolist()
                names = [result.names[i] for i in cls_ids if i in result.names]
                
                # Filter objects based on configuration file
                if allowed_objects is not None:
                    names = [name for name in names if name in allowed_objects]
                
                with data_lock:
                    for name in set(names):
                        if name not in seen_classes[camera_idx]:
                            seen_classes[camera_idx].add(name)
                            counts[camera_idx][name] += 1
        else:
            annotated_frame = frame
        
        if camera_idx in frame_queues:
            try:
                while not frame_queues[camera_idx].empty():
                    frame_queues[camera_idx].get_nowait()
                frame_queues[camera_idx].put(annotated_frame, block=False)
            except:
                pass
    
    cap.release()

def display_streams():
    windows = {idx: f"Camera {idx}" for idx in CAMERA_INDICES}
    
    for idx in CAMERA_INDICES:
        cv2.namedWindow(windows[idx], cv2.WINDOW_NORMAL)
    
    while not stop_event.is_set():
        for idx in CAMERA_INDICES:
            if idx in frame_queues:
                try:
                    frame = frame_queues[idx].get_nowait()
                    cv2.imshow(windows[idx], frame)
                except queue.Empty:
                    pass
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
    
    cv2.destroyAllWindows()

class CountWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Detection Counts")
        self.root.geometry("400x500")
        
        self.label = tk.Label(root, text="", font=("Consolas", 11), 
                             justify="left", anchor="nw", bg="white", padx=10, pady=10)
        self.label.pack(fill=tk.BOTH, expand=True)
        
        tk.Button(root, text="Reset", command=self.reset_counts,
                 bg="lightblue").pack(side=tk.LEFT, padx=10, pady=10)
        tk.Button(root, text="Quit", command=self.quit_app,
                 bg="lightcoral").pack(side=tk.RIGHT, padx=10, pady=10)
        
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.update_gui()

    def reset_counts(self):
        with data_lock:
            for idx in CAMERA_INDICES:
                seen_classes[idx].clear()
                counts[idx].clear()

    def quit_app(self):
        stop_event.set()
        self.root.quit()

    def update_gui(self):
        lines = []
        with data_lock:
            for idx in CAMERA_INDICES:
                lines.append(f"Camera {idx}:")
                if counts[idx]:
                    for cls, cnt in sorted(counts[idx].items()):
                        lines.append(f"  {cls}: {cnt}")
                else:
                    lines.append("  No detections")
                lines.append("")
        
        self.label.config(text="\n".join(lines))
        
        if not stop_event.is_set():
            self.root.after(500, self.update_gui)

if __name__ == "__main__":
    # Initialize frame queues
    for idx in CAMERA_INDICES:
        test_cap = cv2.VideoCapture(idx)
        if test_cap.isOpened():
            frame_queues[idx] = queue.Queue(maxsize=2)
        test_cap.release()
    
    if not frame_queues:
        print("No cameras available")
        exit(1)
    
    # Start camera threads
    threads = []
    for idx in frame_queues.keys():
        t = threading.Thread(target=camera_thread, args=(idx,), daemon=True)
        t.start()
        threads.append(t)
    
    # Start display thread
    display_thread = threading.Thread(target=display_streams, daemon=False)
    display_thread.start()
    
    # Start GUI
    root = tk.Tk()
    CountWindow(root)
    root.mainloop()
    
    # Cleanup
    stop_event.set()
    display_thread.join(timeout=2)
    cv2.destroyAllWindows()