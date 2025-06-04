import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
from customtkinter.windows.widgets.image import CTkImage
import zipfile
import re
import clipboard

import cv2 as cv
from PIL import Image, ImageTk
import threading
import os

# Imports the code from backend.py
from backend import SpriteMaker

class PoseDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Make Me A Sprite")
        self.root.geometry("1280x720")
        self.root.configure(bg="#e9f1fc")

        # Needs to start not recording
        self.recording = False

        # Initialises the camera
        self.selected_camera = tk.IntVar()

        # Initialise these for camera in tkinter
        self.cap = None
        self.after_id = None

        # Initialize these variablea for webcam
        self.first_frame = True
        self.frame_size = None

        # Set up backend
        self.backend = SpriteMaker()

        # Initialises this for later use with backend functions
        self.thread = None

        # Set toolbar
        style = tk.ttk.Style()
        style.configure('TMenubutton', background='#4D97FF', foreground='#FFFFFF')

        # Create menubar buttons instead of using built in menubar for customisation options
        menubar = tk.Frame(self.root, bg='#4D97FF')
        menubar.pack(fill='x')

        file_mb = tk.ttk.Menubutton(menubar, text='File', style='TMenubutton')
        file_menu = tk.Menu(file_mb, tearoff=False, bg='#4D97FF', fg='#FFFFFF')
        file_menu.add_command(label="Open", command=self.file_path_open)
        file_menu.add_command(label="Exit", command=self.root.quit)
        file_mb['menu'] = file_menu
        file_mb.pack(side='left')

        edit_mb = tk.ttk.Menubutton(menubar, text='Edit', style='TMenubutton')
        edit_menu = tk.Menu(edit_mb, tearoff=False, bg='#4D97FF', fg='#FFFFFF')
        edit_menu.add_command(label="Copy File Path", command=lambda: clipboard.copy(self.file_path_txt.get("0.0", "end")))
        edit_mb['menu'] = edit_menu
        edit_mb.pack(side='left')

        settings_mb = tk.ttk.Menubutton(menubar, text='Settings', style='TMenubutton')
        settings_menu = tk.Menu(settings_mb, tearoff=False, bg='#4D97FF', fg='#FFFFFF')
        settings_mb['menu'] = settings_menu
        camera_menu = tk.Menu(settings_menu, tearoff=False)
        for i in range(5):
            camera_menu.add_radiobutton(
                label=str(i),
                variable=self.selected_camera,
                value=i,
                command=self.on_camera_selected
            )
        settings_menu.add_cascade(label="Select Camera", menu=camera_menu)
        
        settings_mb.pack(side='left')

        # Create buttons, labels, etc
        self.create_widgets()

        # Start the camera automatically when the application launches
        self.root.after(100, self.initialize_camera) # 100ms delay gives time for the window to load properly before running function

        # Bind hotkeys for functions
        self.root.bind_all('<KeyPress>', self.hotkey_manager)

        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def initialize_camera(self):
        camera_id = self.selected_camera.get()
        
        # Release any existing camera
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        
        # Open the newly selected camera
        self.cap = cv.VideoCapture(camera_id)
        
        if self.cap.isOpened():
            self.feedback_lbl.configure(text=f"Camera {camera_id}\nActive")
            self.feedback_lbl.configure(text_color="#FFFFFF")
            self.first_frame = True  # Reset frame size calculation
            self.open_camera()  # Start camera feed
        else:
            self.feedback_lbl.configure(text=f"Failed to open\nCamera {camera_id}")
            self.feedback_lbl.configure(text_color="#FF0000")

    def on_camera_selected(self):
        if self.recording:
            # Don't allow camera switching while recording
            self.feedback_lbl.configure(text="Stop recording\nbefore changing\ncamera")
            self.feedback_lbl.configure(text_color="#FF0000")
            # Reset selection to current camera
            for camera_name, camera_id in self.available_cameras.items():
                if camera_id == self.cap.get(cv.CAP_PROP_POS_FRAMES):
                    self.selected_camera.set(camera_name)
                    break
            return
        
        # Cancel any existing after callbacks to avoid multiple camera feeds
        if hasattr(self, 'after_id') and self.after_id:
            self.webcam_lbl.after_cancel(self.after_id)
            self.after_id = None
        
        self.initialize_camera()

    def create_widgets(self):
        self.button_frame = ctk.CTkFrame(
            master=self.root,
            width=220,
            height=600,
            fg_color="#FFFFFF"
        )
        self.button_frame.place(x=10, y=50)
        self.button_frame.lower() # Frame needs to be in the back of the layers

        # Initialise images
        rec_img = ImageTk.PhotoImage(Image.open('images\\RecordButton.png').resize((125, 125), Image.LANCZOS))
        camera_img = ImageTk.PhotoImage(Image.open('images\\CameraButton.png').resize((125, 125), Image.LANCZOS))
        file_img = ImageTk.PhotoImage(Image.open('images\\FileButton.png').resize((125, 125), Image.LANCZOS))

        self.record_btn = ctk.CTkButton(
            master=self.button_frame,
            image=rec_img,
            text="",
            hover=True,
            hover_color="#949494",
            height=125,
            width=125,
            border_width=0,
            corner_radius=0,
            fg_color="transparent",
            command=self.record_poses,
            )
        self.record_btn.place(x=45, y=470)

        self.camera_btn = ctk.CTkButton(
            master=self.button_frame,
            text="",
            image=camera_img,
            hover=True,
            hover_color="#949494",
            height=125,
            width=125,
            border_width=0,
            corner_radius=0,
            fg_color="transparent",
            state="disabled",
            command=self.capture_frame
            )
        self.camera_btn.place(x=45, y=325)

        self.file_btn = ctk.CTkButton(
            master=self.button_frame,
            text="",
            image=file_img,
            hover=True,
            hover_color="#949494",
            height=125,
            width=125,
            border_width=0,
            corner_radius=0,
            fg_color="transparent",
            command=self.file_path_open,
            )
        self.file_btn.place(x=45, y=175)

        # Initialise labels that will have its text is going to be updated
        self.file_path_txt = ctk.CTkTextbox(
            master=self.button_frame,
            font=("Segoe UI", 14),
            text_color="#000000",
            height=35,
            width=185,
            corner_radius=0,
            fg_color="#FFFFFF",
            border_width=2,
            border_color="#808080",
            state="disabled",
            wrap="none",
            )
        self.file_path_txt.place(x=20, y=120)
        self.file_path_txt.configure(state="normal")
        self.file_path_txt.insert("0.0", "file/path/name.sb3")
        self.file_path_txt.configure(state="disabled")

        self.sprite_name_entry = ctk.CTkEntry(
            master=self.button_frame,
            placeholder_text="Example Name",
            placeholder_text_color="#454545",
            font=("Segoe UI", 14),
            text_color="#000000",
            height=40,
            width=160,
            border_width=2,
            corner_radius=6,
            border_color="#000000",
            fg_color="#F0F0F0",
            )
        self.sprite_name_entry.place(x=30, y=50)

        self.enter_sprite_name_lbl = ctk.CTkLabel(
            master=self.button_frame,
            text="Enter Sprite Name",
            font=("Segoe UI", 14),
            text_color="#000000",
            height=30,
            width=95,
            corner_radius=0,
            fg_color="transparent",
            anchor="center"
            )
        self.enter_sprite_name_lbl.place(x=45, y=15)

        self.feedback_frame = ctk.CTkFrame(
            master=self.root,
            width=230,
            height=600,
            fg_color="#4d97ff"
        )
        self.feedback_frame.place(x=1040, y=50)

        self.feedback_lbl = ctk.CTkLabel(
            master=self.feedback_frame,
            text="Input File Path",
            anchor="center",
            justify="left",
            font=("Segoe UI", 28),
            text_color="#FFFFFF",
            fg_color="transparent",
            height=30,
            width=95,
            corner_radius=0,
            )
        self.feedback_lbl.place(x=25, y=20)

        self.webcam_lbl = ctk.CTkLabel(
            master=self.root,
            text=""
        )
        self.webcam_lbl.place(x=230, y=50)

    # Displays webcam in label, code mostly taken from GeekForGeeks
    def open_camera(self): 
        if self.cap is None or not self.cap.isOpened():
            self.webcam_lbl.after(1000, self.initialize_camera)  # Try to reinitialize after 1 second
            return
            
        # Capture the video frame by frame 
        ret, frame = self.cap.read()

        if not ret:
            # Camera could have been disconnected
            self.feedback_lbl.configure(text="Camera\nDisconnected")
            self.feedback_lbl.configure(text_color="#FF0000")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.webcam_lbl.after(1000, self.initialize_camera)  # Try to reinitialize after 1 second
            return

        if self.first_frame:
            og_height, og_width = frame.shape[:2]
            width = 800 # Width hard coded to fit design
            ratio = width / og_width
            height = og_height * ratio # height calculated based on webcam's aspect ratio
            self.frame_size = (int(width), int(height))
            self.first_frame = False

        # Update feedback label to match status input during recording
        if self.recording:
            self.update_feedback()

        resized_frame = cv.resize(frame, self.frame_size, cv.INTER_LINEAR)
    
        # Convert image from one color space to other
        opencv_image = cv.cvtColor(resized_frame, cv.COLOR_BGR2RGBA) 
    
        # Capture the latest frame and transform to image 
        captured_image = Image.fromarray(opencv_image) 
    
        # Convert captured image to photoimage 
        photo_image = CTkImage(light_image=captured_image, size=self.frame_size)
    
        # Displaying photoimage in the label 
        self.webcam_lbl.photo_image = photo_image 
    
        # Configure image in the label 
        self.webcam_lbl.configure(image=photo_image)
    
        # Repeat the same process after every 10 milliseconds
        self.after_id = self.webcam_lbl.after(10, self.open_camera)

    def hotkey_manager(self, event):
        key = event.keysym
        # For recording we need ctrl to be pressed so that recording doesn't start while typing
        control = event.state & 0x4
        if control and key == 'r':
            self.record_poses()
        elif self.recording and key == 'space':
            self.capture_frame()

    def update_feedback(self):
        feedback = self.backend.feedback
        self.feedback_lbl.configure(text=feedback)
        if feedback == "NO PERSON\nIN FRAME": # Not best way of doing this TODO: Fix later
            self.feedback_lbl.configure(text_color="#FF0000") # Change red with no person in frame
        else:
            self.feedback_lbl.configure(text_color="#FFFFFF")

    def record_poses(self):
        # Get input variables
        sprite_name = self.sprite_name_entry.get()
        scratch_file = self.file_path_txt.get("0.0", "end")
        scratch_file = scratch_file.rstrip('\n') # Text box has \n at the end often, need to remove it to have the file path correctly read

        if not self.recording:
            if str(sprite_name) and re.match(r"^[A-Za-z0-9_\-\.]*$", sprite_name) and len(sprite_name) <= 100:
                if self.check_scratch_file(scratch_file):
                    self.backend.recording = True
                    
                    # Enable camera button
                    self.camera_btn.configure(state="normal")
                    self.file_btn.configure(state="disabled")
                    self.sprite_name_entry.configure(state="disabled")

                    # Run in separate thread to let frontend run while the backend loops
                    self.thread = threading.Thread(target=self.backend.take_frames, args=(self.cap, sprite_name))
                    self.thread.start()

                    self.recording = True
                else:
                    self.feedback_lbl.configure(text="Enter a valid\nfile path\nto start")
                    self.feedback_lbl.configure(text_color="#FF0000")
            else:
                self.feedback_lbl.configure(text="Enter a valid\nSprite Name\nto start")
                self.feedback_lbl.configure(text_color="#FF0000")
        else:
            self.backend.recording = False

            # Disable camera button
            self.camera_btn.configure(state="disabled")
            self.file_btn.configure(state="normal")
            self.sprite_name_entry.configure(state="normal")


            # Wait for thread to finish after setting backend to false, exiting loop
            self.thread.join()
            frame_count = self.backend.frame_count

            if frame_count > 0:
                try:
                    # Let user select what frames to exclude
                    exclude_list, blur_face = self.frame_select_window(sprite_name)

                    # import images as sprites
                    output_path = self.backend.import_scratch_sprite(scratch_file, sprite_name, frame_count, exclude_list, blur_face)
                    self.feedback_lbl.configure(text="Success!")
                    self.feedback_lbl.configure(text_color="#FFFFFF")
                except Exception as e:
                    self.feedback_lbl.configure(text=f"Error during\nScratch process:\n{e}")
                    self.feedback_lbl.configure(text_color="#FF0000")
            else:
                self.feedback_lbl.configure(text="Camera closed\nbefore any\nimages taken")
                self.feedback_lbl.configure(text_colfor="#FFFFFF")

            self.recording = False

    def check_scratch_file(self, file_path):
        if not os.path.isfile(file_path):
           return False
       
        try:
            # Try to open as zip file
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # Check for project.json which is required in all Scratch projects
                if 'project.json' not in zip_ref.namelist():
                    return False
            return True
        except (zipfile.BadZipFile, KeyError):
            return False

    def capture_frame(self):
        _, frame = self.cap.read()
        sprite_name = self.sprite_name_entry.get()
        self.backend.capture_frame(sprite_name, frame)

    def file_path_open(self):
        filename = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = (("Scratch files","*.sb3*"),
                                                       ("all files","*.*")))
      
        # Change label contents
        self.file_path_txt.configure(state="normal")
        self.file_path_txt.delete('0.0', 'end')
        self.file_path_txt.insert("0.0", filename)
        self.file_path_txt.configure(state="disabled")
        
        self.feedback_lbl.configure(text="Press Record to\nStart Process")
        self.feedback_lbl.configure(text_color="#FFFFFF")

    def frame_select_window(self, sprite_name):
        frame_list = []
        for i in range(self.backend.frame_count):
            img = ImageTk.PhotoImage(Image.open(f"{sprite_name}_{i}.png"))
            frame_list.append((img, True))

        top = tk.Toplevel(self.root)
        width, height = self.frame_size
        top.geometry(f"{width}x{(height + 20)}")
        
        self.frame_lbl = tk.Label(top, image=frame_list[0][0])
        self.left_btn = tk.Button(top, text="<-", state='disabled', command=lambda: self.change_frame(frame_list, False))
        self.right_btn = tk.Button(top, text="->", state='normal', command=lambda: self.change_frame(frame_list, True))
        
        check_boxes_frame = tk.Frame(top)
        include_frame_frame = tk.Frame(check_boxes_frame)
        include_frame_lbl = tk.Label(include_frame_frame, text="Include Frame?")
        self.include_frame = tk.BooleanVar(value=True)
        self.include_frame_check = tk.Checkbutton(include_frame_frame,
                                                  variable=self.include_frame,
                                                  command=lambda: self.toggle_include_frame(frame_list))

        blur_face_frame = tk.Frame(check_boxes_frame)
        blur_face_lbl = tk.Label(blur_face_frame, text="Blur Face?")
        self.blur_face = tk.BooleanVar(value=False)
        blur_face_check = tk.Checkbutton(blur_face_frame, variable=self.blur_face)

        confirm_btn = tk.Button(top, text="Confirm Selection", command=lambda: self.confirm_selection(top, frame_list))

        self.frame_lbl.pack(side=tk.TOP, expand=False)
        
        self.left_btn.pack(side=tk.LEFT, expand=True)
        self.right_btn.pack(side=tk.LEFT, expand=True)
        confirm_btn.pack(side=tk.RIGHT, expand=True)

        check_boxes_frame.pack(side=tk.RIGHT, expand=True)
        include_frame_frame.pack(side=tk.BOTTOM, expand=True)
        include_frame_lbl.pack(side=tk.LEFT, expand=True)
        self.include_frame_check.pack(side=tk.LEFT, expand=False)
        blur_face_frame.pack(side=tk.BOTTOM, expand=True)
        blur_face_lbl.pack(side=tk.LEFT, expand=True)
        blur_face_check.pack(side=tk.LEFT, expand=False)
        
        self.index = 0

        self.excluded_frames = None

        top.wait_window()

        return self.excluded_frames, self.blur_face.get()

    def change_frame(self, frame_list: list[tuple[ImageTk.PhotoImage, bool]], direction: bool):
        # True means right, False means left
        if direction:
            self.index += 1
        else:
            self.index -= 1
        self.frame_lbl.config(image=frame_list[self.index][0])
        
        # if the first image, deactivate left button
        if self.index == 0:
            self.left_btn.configure(state='disabled')
        else:
            self.left_btn.configure(state='normal')

        # if the last image, deactivate right button
        if self.index == (len(frame_list) - 1):
            self.right_btn.configure(state='disabled')
        else:
            self.right_btn.configure(state='normal')

        # Set Checkbutton
        self.include_frame.set(frame_list[self.index][1])

    def toggle_include_frame(self, frame_list: list[tuple[ImageTk.PhotoImage, bool]]):
        frame_list[self.index] = (frame_list[self.index][0], self.include_frame.get())

    def confirm_selection(self, window, frame_list):
        self.excluded_frames = [i for i, frame in enumerate(frame_list) if not frame[1]]
        window.destroy()

    def on_closing(self):
        # Cancel any pending after callbacks
        if hasattr(self, 'after_id') and self.after_id:
            self.webcam_lbl.after_cancel(self.after_id)
        
        # Properly release camera resources when closing the app
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        
        self.root.destroy()

    # When the program closes
    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()