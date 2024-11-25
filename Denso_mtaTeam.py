import cv2 as cv
import mediapipe as mp
import time
import numpy 
import cv2 as cv
import mediapipe as mp
import time
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class HandDetector:
    def __init__(self, mode=False, maxHands=2, modelC=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelC = modelC
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Khởi tạo các mô hình của MediaPipe
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelC,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame):
        """ Nhận diện bàn tay trong khung hình. """
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)

        # Vẽ khung xương bàn tay nếu phát hiện
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
        return frame
 
    def findPosition(self, frame):
        """ Tìm vị trí các điểm landmark của bàn tay. """
        lmList = []
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
        return lmList

def isHandInBox(lmList, box):
    """ Kiểm tra xem bất kỳ điểm nào của bàn tay có nằm trong khung hình vuông không. """
    for point in lmList:
        x, y = point[1], point[2]
        if box[0] < x < box[2] and box[1] < y < box[3]:
            return True
    return False

def create_custom_boxes():
    """ Tạo danh sách các ô với vị trí tùy ý. """
    return [
        (50, 50, 150, 150),     # Ô 1
        (200, 50, 300, 150),   # Ô 2
        (350, 50, 450, 150),    # Ô 3
        (500, 50, 600, 150)    # Ô 4
    ]
class Timer:
    def __init__(self):
        """
        Quản lý thời gian trong từng ô và toàn bộ quá trình.
        - `time_stamps`: Thời gian đã sử dụng cho từng ô.
        - `process_times`: Tổng thời gian cho mỗi chu trình hoàn thành.
        """
        self.time_stamps = [0] * 4
        self.start_times = [0] * 4
        self.active_boxes = [False] * 4
        self.finished_boxes = [False] * 4
        self.current_step = 0
        self.process_times = []

    def update(self, lmList, boxes):
        """
        Cập nhật thời gian cho từng ô dựa vào vị trí bàn tay.
        - Nếu bàn tay nằm trong ô hiện tại, bắt đầu đếm thời gian.
        - Nếu rời khỏi ô, dừng đếm và lưu lại thời gian.
        """
        if len(lmList) != 0 and isHandInBox(lmList, boxes[self.current_step]):
            if not self.active_boxes[self.current_step]:
                self.start_times[self.current_step] = time.time()
                self.active_boxes[self.current_step] = True
        else:
            if self.active_boxes[self.current_step]:
                self.time_stamps[self.current_step] += time.time() - self.start_times[self.current_step]
                self.active_boxes[self.current_step] = False
                self.finished_boxes[self.current_step] = True

                if self.current_step < len(boxes) - 1 and self.finished_boxes[self.current_step]:
                    self.current_step += 1

        # Move to the next step immediately after the hand is detected in the box
        if self.active_boxes[self.current_step] and not self.finished_boxes[self.current_step]:
            self.finished_boxes[self.current_step] = True
            if self.current_step < len(boxes) - 1:
                self.current_step += 1

        if all(self.finished_boxes[i] for i in range(len(boxes))):
            total_time = sum(self.time_stamps)
            self.process_times.append(total_time)
            self.time_stamps = [0] * len(boxes)
            self.finished_boxes = [False] * len(boxes)
            self.current_step = 0

        return self.time_stamps, self.process_times

def check_time_limit(total_time, time_limit):   
    """ Kiểm tra xem tổng thời gian có vượt quá giới hạn cho phép hay không. """
    return total_time > time_limit

class Process:
    """
    Lớp quản lý thứ tự quy trình đi qua các ô. 
    - Cho phép tùy chỉnh thứ tự ô theo yêu cầu.
    """
    def __init__(self, sequence):
        """
        sequence: Danh sách thứ tự các ô (ví dụ: [0, 1, 2, 3]).
        """
        self.sequence = sequence  # Thứ tự các bước trong quy trình
        self.current_step = 0     # Bước hiện tại

    def reset(self):
        """Đặt lại quy trình về trạng thái ban đầu."""
        self.current_step = 0

    def getCurrentBox(self):
        """Trả về ô hiện tại cần đặt tay trong quy trình."""
        if self.current_step < len(self.sequence):
            return self.sequence[self.current_step]
        return None

    def updateStep(self, finished_boxes):
        """
        Kiểm tra xem bước hiện tại đã hoàn thành chưa và chuyển sang bước tiếp theo.
        finished_boxes: Danh sách các ô đã hoàn thành từ lớp Timer.
        """
        if self.current_step < len(self.sequence):
            current_box = self.sequence[self.current_step]
            if finished_boxes[current_box]:  # Nếu ô hiện tại đã hoàn thành
                self.current_step += 1      # Chuyển sang bước tiếp theo

    def isCompleted(self):
        """Kiểm tra xem toàn bộ quy trình đã hoàn thành chưa."""
        return self.current_step >= len(self.sequence)

class CameraApp:
    def __init__(self, window, window_title):
        print("Initializing CameraApp")
        self.window = window
        self.window.title(window_title)
        self.is_processing = False
        self.process_times = {}  # Lưu thời gian của từng bước
        self.target_time_var = tk.StringVar(value="5")  # Thời gian quy cách (mặc định 5 giây)
        self.current_process_var = tk.StringVar(value="1")  # Quy trình hiện tại
        self.current_step = 1  # Bước hiện tại

        # Cấu hình layout grid
        self.window.rowconfigure(0, weight=8)  # Tăng trọng số cho màn hình camera
        self.window.rowconfigure(1, weight=1)
        self.window.columnconfigure(0, weight=5)  # Camera chiếm 5 phần
        self.window.columnconfigure(1, weight=2)  # Thông tin bên phải chiếm 2 phần

        # Màn hình hiển thị cam (rộng hơn)
        self.video_frame = tk.Frame(self.window, bg="black", bd=2, relief=tk.SOLID)
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Khu vực hiển thị thông tin bên phải
        self.info_frame = tk.Frame(self.window)
        self.info_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.info_frame.rowconfigure(0, weight=1)
        self.info_frame.rowconfigure(1, weight=2)
        self.info_frame.rowconfigure(2, weight=1)

        # Nút "Quy cách" và "Quy trình" trên cùng một dòng
        self.controls_frame = tk.Frame(self.info_frame)
        self.controls_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.controls_frame.columnconfigure(0, weight=1)
        self.controls_frame.columnconfigure(1, weight=1)

        tk.Label(self.controls_frame, text=" Quy cách (giây):", font=("Arial", 12)).grid(row=0, column=0, sticky="w")
        self.target_time_entry = tk.Entry(self.controls_frame, textvariable=self.target_time_var, font=("Arial", 12))
        self.target_time_entry.grid(row=0, column=1, sticky="ew", padx=5)

        tk.Label(self.controls_frame, text="Quy trình hiện tại:", font=("Arial", 12)).grid(row=0, column=2, sticky="w")
        self.process_entry = tk.Entry(self.controls_frame, textvariable=self.current_process_var, font=("Arial", 12))
        self.process_entry.grid(row=0, column=3, sticky="ew", padx=5)

        # Hiển thị thời gian tổng hợp của từng bước
        self.times_frame = tk.Frame(self.info_frame, bd=2, relief=tk.SOLID)
        self.times_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)

        self.canvas = tk.Canvas(self.times_frame)
        self.scrollable_frame = tk.Frame(self.canvas)
        self.scrollbar = ttk.Scrollbar(self.times_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        # Danh sách các ô TextBox
        self.textbox_entries = []  # Lưu các ô TextBox cho từng bước
        for i in range(5):  # Giả sử có 5 bước quy trình
            textbox = tk.Entry(self.scrollable_frame, font=("Arial", 12))
            textbox.grid(row=i, column=0, padx=5, pady=5)
            self.textbox_entries.append(textbox)  # Lưu vào danh sách
        def add_textbox(self):
            textbox = tk.Entry(self.scrollable_frame, font=("Arial", 12))
            textbox.grid(row=len(self.textbox_entries), column=0, padx=5, pady=5, sticky="ew")
            self.textbox_entries.append(textbox)  # Add to the list
        def update_textboxes(self,process_time):
            for i, time in enumerate (self.process_time):
                
             if i < len(self.textbox_entries):
                self.textbox_entries[i].delete(0, tk.END)  # Xóa nội dung cũ
                self.textbox_entries[i].insert(0, f"Quy trình {i+1}: {time:.2f} giây")  # Thêm thời gian mới
    #     def start_process(self):
    #      self.is_processing = True
    #      self.timer.start()
    #      self.process_thread = threading.Thread(target=self.process_video)
    #      self.process_thread.start()

    # def stop_process(self):
    #     self.is_processing = False
    #     self.cap.release()
    #     process_times = self.timer.process_times  # Lấy danh sách thời gian
    #     self.update_textboxes(process_times)     # Cập nhật lên TextBox
        # self.show_chart()

    # def process_video(self):
    #     while self.is_processing:
    #         ret, frame = self.cap.read()
    #         if ret:
    #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             frame = cv2.resize(frame, (640, 480))
    #             img = tk.PhotoImage(data=cv2.imencode('.ppm', frame)[1].tobytes())
    #             self.video_frame.imgtk = img
    #             self.video_frame.configure(image=img)
    #             self.root.update_idletasks()
    #         else:
    #             break
    #     self.timer.stop()  # Dừng tính thời gian khi kết thúc video
        # Biểu đồ
        self.figure, self.ax = plt.subplots(figsize=(6, 2))
        self.ax.set_title("Biểu đồ đo thời gian")
        self.chart = FigureCanvasTkAgg(self.figure, self.window)
        self.chart.get_tk_widget().grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

        # Nút "Bắt đầu" và "Kết thúc"
        self.button_frame = tk.Frame(self.window)
        self.button_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

        self.start_button = tk.Button(self.button_frame, text="Bắt đầu", font=("Arial", 12), bg="#90EE90", command=self.start_process)
        self.start_button.pack(side=tk.RIGHT, padx=5)

        self.stop_button = tk.Button(self.button_frame, text="Kết thúc", font=("Arial", 12), bg="#FFB6C1", command=self.stop_process)
        self.stop_button.pack(side=tk.RIGHT, padx=5)
        
        self.start_button = tk.Button(self.button_frame, text="Xem Video", font=("Arial", 12), bg="#90EE90", command=self.start_process)
        self.start_button.pack(side=tk.LEFT, padx=5)

        # Khởi tạo camera và HandDetector
        self.cap = cv.VideoCapture(0)
        self.detector = HandDetector()
        self.timer = Timer()
        self.process = None
        self.boxes = create_custom_boxes()

        # Kiểm tra xem camera có mở được không
        if not self.cap.isOpened():
            print("Không thể mở camera!")
            self.window.destroy()
            return

        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        print("CameraApp initialized")

    def update_video(self): 
        if not self.is_processing:
            return

        try:
            success, frame = self.cap.read()
            if success: 
                frame = self.detector.findHands(frame)
                lmList = self.detector.findPosition(frame)
                for i, box in enumerate(self.boxes):
                    color = (0, 255, 0)
                    if i == self.process.getCurrentBox():
                        color = (0, 0, 255)
                        if isHandInBox(lmList, box):
                            print(f"Box {i+1} completed")
                            self.process.updateStep(self.timer.finished_boxes)
                            self.update_step_labels()
                            color = (0, 255, 0)  # Change color to green after step is completed
                    cv.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv.putText(frame, f"Box {i+1}", (box[0] + 10, box[1] - 10),  
                            cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
                _, process_times = self.timer.update(lmList, self.boxes)
                if self.process.isCompleted():
                    cv.putText(frame, "Process Completed", (50, frame.shape[0] - 50),
                            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    self.stop_process()  # Stop the process when completed
                if len(process_times) > 0:
                    total_time = sum(process_times)
                    color = (0, 0, 255) if check_time_limit(total_time, 15) else (0, 255, 0)
                    cv.putText(frame, f"Total Time: {int(total_time)}s", (500, frame.shape[0] - 100),
                            cv.FONT_HERSHEY_PLAIN, 1, color, 2)
                img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)
            else:
                print("Failed to read frame from camera")
            self.window.after(10, self.update_video)
        except Exception as e:
            print("Error updating video:", e)

    def start_process(self):
        """Bắt đầu quá trình nhận diện và xử lý"""
        try:
            sequence = list(map(int, self.process_entry.get().split()))
            self.process = Process(sequence)
            self.process.reset()
            self.timer = Timer()
            self.is_processing = True
            self.process_times = {}  # Khởi tạo lại thời gian quy trình
            self.current_step = 1  # Quay lại bước 1
            # self.update_step_labels()
            self.window.after(0, self.update_video)  # Bắt đầu cập nhật video
        except ValueError:
            print("Invalid input for process sequence")

    def stop_process(self):
        self.is_processing = False
        self.cap.release()
        self.show_chart()

    def update_step_labels(self):
        """Cập nhật trạng thái các bước quy trình"""
        if self.process:
            current_box = self.process.getCurrentBox()
            step_label = f"Bước {current_box + 1}"
            self.current_process_var.set(step_label)

    def show_chart(self):
        """Hiển thị biểu đồ tổng hợp thời gian"""
        self.ax.clear()
        self.ax.set_title("Biểu đồ đo thời gian")
        steps = list(self.process_times.keys())
        times = list(self.process_times.values())
        self.ax.plot(steps, times, marker='o', linestyle='-', color='b')
        self.ax.set_xlabel("Bước")
        self.ax.set_ylabel("Thời gian (giây)")
        self.ax.grid(True)
        self.chart.draw()

def main():
    print("App starting")
    root = tk.Tk()
    app = CameraApp(root, "Camera with Process Tracking")
    print("App started")
    root.mainloop()

if __name__ == "__main__":
    main()
