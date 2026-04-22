import cv2
import tkinter as tk
from PIL import Image, ImageTk
import pyautogui
import mediapipe as mp
import numpy as np
import time

# -----------------------------
# إعداد Mediapipe Face Mesh
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5)

# -----------------------------
# إعداد الشاشة والتنعيم
# -----------------------------
screen_width, screen_height = pyautogui.size()
prev_x, prev_y = 0, 0

def smooth(x, y, prev_x, prev_y, factor=0.5):
    # يخفف تحركات الماوس المفاجئة
    new_x = prev_x + int((x - prev_x) * factor)
    new_y = prev_y + int((y - prev_y) * factor)
    return new_x, new_y

# -----------------------------
# Blink detection بسيط
# -----------------------------
blink_threshold = 0.25
last_blink_time = 0
blink_delay = 0.3  # ثانية
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def eye_aspect_ratio(landmarks, indices):
    a = np.linalg.norm(np.array([landmarks[indices[1]].x, landmarks[indices[1]].y]) -
                       np.array([landmarks[indices[5]].x, landmarks[indices[5]].y]))
    b = np.linalg.norm(np.array([landmarks[indices[2]].x, landmarks[indices[2]].y]) -
                       np.array([landmarks[indices[4]].x, landmarks[indices[4]].y]))
    c = np.linalg.norm(np.array([landmarks[indices[0]].x, landmarks[indices[0]].y]) -
                       np.array([landmarks[indices[3]].x, landmarks[indices[3]].y]))
    return (a + b) / (2.0 * c)

# -----------------------------
# إعداد نافذة Tkinter
# -----------------------------
root = tk.Tk()
root.title("Eye Control GUI")
root.geometry("400x650")
root.configure(bg="white")

# عرض الكاميرا
camera_frame = tk.Label(root, bg="#d9d9d9")
camera_frame.pack(pady=20)

# تعليمات بسيطة
label_text = tk.Label(root, text="Move your eye and blink to click",
                      font=("Arial", 10), bg="white")
label_text.pack()
blink_label = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="white", fg="red")
blink_label.pack(pady=10)

# -----------------------------
# الأزرار
# -----------------------------
btn_frame = tk.Frame(root, bg="white")
btn_frame.pack(pady=40)

buttons = {}
def button_pressed(btn):
    btn.config(bg="red")
    root.after(300, lambda: btn.config(bg="#77dd77"))

# زرار الاتجاهات
up_btn = tk.Button(btn_frame, text="↑", font=("Arial", 18, "bold"),
                   bg="#77dd77", width=5, height=2, command=lambda: button_pressed(up_btn))
up_btn.grid(row=0, column=1, padx=10, pady=10)
buttons['up'] = up_btn

left_btn = tk.Button(btn_frame, text="←", font=("Arial", 18, "bold"),
                     bg="#77dd77", width=5, height=2, command=lambda: button_pressed(left_btn))
left_btn.grid(row=1, column=0, padx=10, pady=10)
buttons['left'] = left_btn

down_btn = tk.Button(btn_frame, text="↓", font=("Arial", 18, "bold"),
                     bg="#77dd77", width=5, height=2, command=lambda: button_pressed(down_btn))
down_btn.grid(row=1, column=1, padx=10, pady=10)
buttons['down'] = down_btn

right_btn = tk.Button(btn_frame, text="→", font=("Arial", 18, "bold"),
                      bg="#77dd77", width=5, height=2, command=lambda: button_pressed(right_btn))
right_btn.grid(row=1, column=2, padx=10, pady=10)
buttons['right'] = right_btn

# -----------------------------
# تشغيل الكاميرا
# -----------------------------
cap = cv2.VideoCapture(0)

def update_camera():
    global prev_x, prev_y, last_blink_time
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_camera)
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        # نقطة العين للتتبع
        iris_points = [474, 475, 476, 477]
        h, w, _ = frame.shape
        iris_coords = [(int(face_landmarks[i].x*w), int(face_landmarks[i].y*h)) for i in iris_points]
        cx = int(np.mean([p[0] for p in iris_coords]))
        cy = int(np.mean([p[1] for p in iris_coords]))

        # تحريك الماوس
        x_screen = screen_width - int(cx / w * screen_width)
        y_screen = int(cy / h * screen_height)
        x_screen, y_screen = smooth(x_screen, y_screen, prev_x, prev_y)
        prev_x, prev_y = x_screen, y_screen
        pyautogui.moveTo(x_screen, y_screen, duration=0.05)

        # كشف blink للنقر
        ear = eye_aspect_ratio(face_landmarks, RIGHT_EYE)
        current_time = time.time()
        if ear < blink_threshold and (current_time - last_blink_time) > blink_delay:
            widget = root.winfo_containing(prev_x, prev_y)
            if widget in buttons.values():
                button_pressed(widget)
            last_blink_time = current_time
            blink_label.config(text="Blink Detected!")
            root.after(500, lambda: blink_label.config(text=""))

        # رسم النقاط على الكاميرا
        for p in iris_coords:
            cv2.circle(frame, p, 2, (0,0,255), -1)
        cv2.circle(frame, (cx, cy), 3, (0,255,0), -1)

    img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    camera_frame.config(image=img)
    camera_frame.image = img
    root.after(10, update_camera)

root.after(100, update_camera)
root.mainloop()
cap.release()
