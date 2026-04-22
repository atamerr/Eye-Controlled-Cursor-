import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np
import pyautogui
import threading
import time
import vlc


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_points):
    p1 = np.array([landmarks[eye_points[1]].x, landmarks[eye_points[1]].y])
    p2 = np.array([landmarks[eye_points[2]].x, landmarks[eye_points[2]].y])
    p3 = np.array([landmarks[eye_points[5]].x, landmarks[eye_points[5]].y])
    p4 = np.array([landmarks[eye_points[4]].x, landmarks[eye_points[4]].y])
    p5 = np.array([landmarks[eye_points[0]].x, landmarks[eye_points[0]].y])
    p6 = np.array([landmarks[eye_points[3]].x, landmarks[eye_points[3]].y])
    vertical1 = np.linalg.norm(p1 - p3)
    vertical2 = np.linalg.norm(p2 - p4)
    horizontal = np.linalg.norm(p5 - p6)
    return (vertical1 + vertical2) / (2.0 * horizontal)


video_list = []
current_index = 0
paused = False
blink_threshold = 0.20
blink_flag = False
smooth_factor = 0.2
prev_x, prev_y = 0, 0


root = tk.Tk()
root.title("Eye Controlled Player")
root.geometry("1200x650")
root.configure(bg="#222")


video_frame = tk.Frame(root, bg="black")
video_frame.place(x=20, y=20, width=720, height=400)

player = None 


def add_videos():
    global video_list, current_index, player
    files = filedialog.askopenfilenames(title="Select Videos", filetypes=[("Video files","*.mp4 *.avi *.mkv")])
    if files:
        video_list.extend(files)
        if player is None and video_list:
            current_index = 0
            player = vlc.MediaPlayer(video_list[current_index])
            player.set_hwnd(video_frame.winfo_id())
            player.play()

add_btn = tk.Button(root, text="Add Videos", font=("Arial", 14), bg="#555", fg="white", command=add_videos)
add_btn.place(x=800, y=300) 


def play_pause():
    global paused
    if player:
        if paused:
            player.play()
        else:
            player.pause()
        paused = not paused

def change_video(step):
    global current_index, player
    if video_list:
        current_index = (current_index + step) % len(video_list)
        if player:
            player.stop()
        player = vlc.MediaPlayer(video_list[current_index])
        player.set_hwnd(video_frame.winfo_id())
        player.play()

def skip_forward():
    if player:
        t = player.get_time()
        player.set_time(t + 10000)  

def skip_backward():
    if player:
        t = player.get_time()
        player.set_time(max(0, t - 10000)) 

btn_prev = tk.Button(root, text="⏮ Previous", width=13, command=lambda: change_video(-1))
btn_prev.place(x=800, y=350)
btn_back = tk.Button(root, text="⏪ Back 10s", width=13, command=skip_backward)
btn_back.place(x=800, y=390)
btn_play = tk.Button(root, text="▶️ Play/Pause", width=13, command=play_pause)
btn_play.place(x=800, y=430)
btn_for = tk.Button(root, text="⏩ Forward 10s", width=13, command=skip_forward)
btn_for.place(x=800, y=470)
btn_next = tk.Button(root, text="⏭ Next", width=13, command=lambda: change_video(1))
btn_next.place(x=800, y=510)


camera_label = tk.Label(root, bg="black")
camera_label.place(x=800, y=20, width=360, height=260)

cap_cam = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

def camera_thread():
    global blink_flag, prev_x, prev_y
    while True:
        ret, frame = cap_cam.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            pupil_x = int(landmarks[468].x * w)
            pupil_y = int(landmarks[468].y * h)
            cv2.circle(frame, (pupil_x, pupil_y), 5, (0,0,255), -1)

            mouse_x = np.interp(pupil_x, [0, w], [0, screen_width])
            mouse_y = np.interp(pupil_y, [0, h], [0, screen_height])
            smooth_x = prev_x + (mouse_x - prev_x) * smooth_factor
            smooth_y = prev_y + (mouse_y - prev_y) * smooth_factor
            pyautogui.moveTo(smooth_x, smooth_y)
            prev_x, prev_y = smooth_x, smooth_y

            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_LANDMARKS)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_LANDMARKS)
            ear = (left_ear + right_ear) / 2.0
            if ear < blink_threshold:
                if not blink_flag:
                    threading.Thread(target=pyautogui.click).start()
                    blink_flag = True
            else:
                blink_flag = False


      
        frame_disp = cv2.resize(frame, (360, 260), interpolation=cv2.INTER_LINEAR)

        frame_disp_rgb = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)


        img = ImageTk.PhotoImage(Image.fromarray(frame_disp_rgb))
        camera_label.config(image=img)
        camera_label.image = img


        time.sleep(0.01)

threading.Thread(target=camera_thread, daemon=True).start()
root.mainloop()
