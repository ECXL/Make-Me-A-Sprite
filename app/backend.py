import numpy as np
import cv2 as cv
import json
import shutil
import hashlib
from PIL import Image
import zipfile
from io import BytesIO
import argparse
import time
import threading

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" # To stop pygame opening message
import pygame

parser = argparse.ArgumentParser(prog='Scratch Sprite Creator')
parser.add_argument('--usb-camera', action='store_true', default=False)
parser.add_argument('--use-file', action='store_true', default=False)

class SpriteMaker:
    def __init__(self):
        # For Live_Stream callbacks
        self.latest_segmentation_mask = None
        self.og_height = 1
        self.og_width = 1

        # For playing sounds
        pygame.mixer.init()
        self.sound_channel = pygame.mixer.Channel(0)
        self.recording = False

        # Initialise for frame capture loop
        self.frame_count = 0

        self.feedback = "MOVING"

        self.video_input = False

    def create_pose_detector(self, running_mode):
        # Live Stream requires result callback to function, otherwise it should not be provided
        if running_mode == mp.tasks.vision.RunningMode.LIVE_STREAM:
            base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task') # Uses lighter model to help with processing during livestream
            min_pose_detection_confidence = 0.75
            min_pose_presence_confidence = 0.75
            min_tracking_confidence = 0.75
            result_callback = self.pose_detector_callback
        else:
            base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
            min_pose_detection_confidence = 0.5
            min_pose_presence_confidence = 0.5
            min_tracking_confidence = 0.5
            result_callback = None # Image does not use result_callback

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            output_segmentation_masks=True,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            result_callback=result_callback)
        detector = vision.PoseLandmarker.create_from_options(options)

        return detector

    # pose_detector_callback needs output_image and time_stamp_ms in order for it to work. Even though they are not used, they are needed for pose_detector
    def pose_detector_callback(self, result: mp.tasks.vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        if result.segmentation_masks:
            # Get mask of the pose segmentation
            pose_mask = result.segmentation_masks[0].numpy_view()
            # Resize mask back to original size
            pose_mask_resized = cv.resize(pose_mask, (self.og_width, self.og_height))
            # Convert mask to proper data type
            self.latest_segmentation_mask = (pose_mask_resized * 255).astype(np.uint8)
        else:
            self.latest_segmentation_mask = None

    def take_frames(self, cap, image_name):
        self.frame_count = 0

        # Taking a frame needs you still for a second and then have a cooldown of 3 seconds
        on_cooldown = True # Wait 5 seconds to start once recording starts
        tick = 5 # every 5 ticks initialiser
        start_cooldown_time = int(time.time() * 1000)
        start_capture_time = None
        pre_capture_delay = 1000 # both these are in ms
        capture_cooldown = 3000
        threshold = 90.0

        # This will be used on the isolated person to detect when a person is still
        backSub = cv.createBackgroundSubtractorMOG2()

        # Isolate person using pose segmenter, needs Live_Stream running mode to efficiently segment frames in real time
        running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM

        i = 0
        detector = self.create_pose_detector(running_mode)
        
        try:
            # Get first frame to determine camera dimensions
            ret, frame = cap.read()
            if ret:
                self.og_height, self.og_width = frame.shape[:2]
            else:
                print("Can't receive frame. Exiting ...")
                exit()

            # For video files, determine the proper FPS
            if self.video_input: # If not a live input
                fps = cap.get(cv.CAP_PROP_FPS)
                if fps > 0:
                    frame_delay = 1000 / fps
                else:
                    frame_delay = 0.033  # Default to ~30fps if not available
                prev_time = time.time() * 1000

            # This allows for the frontend to run this function as a thread and then cancel it with a frontend button press 
            while self.recording:
                # Add timestamp to the current frame
                current_time = time.time() * 1000 # milliseconds

                # This is for video files debugging
                if self.video_input: # If not a live input
                    elapsed_time = current_time - prev_time
                    # Video files play back as fast as processing allows in cv.VideoCapture and thus plays back at inconsistent speeds, sometimes faster than the frame speed recorded in
                    # To circumvent this we skip loops going faster than frame rate with this check
                    # Makes the playback more real world accurate
                    if elapsed_time <= frame_delay:
                        continue 
                    prev_time = time.time() * 1000

                ret, frame = cap.read() # if frame is read correctly ret is True
                if not ret:
                    print("Can't receive frame. Exiting ...")
                    break

                # Display the resulting frame
                result = frame.copy()
                
                # If running just the backend
                if __name__ == "__main__":
                    key = cv.waitKey(1)
                    if key == 32: # if spacebar is pressed, capture a frame
                        self.capture_frame(image_name, frame)

                    elif key == ord('q'): # if q key is pressed, exit
                        break
                    cv.putText(result, self.feedback, (10, 30), 
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    window_name = 'Webcam'
                    cv.imshow(window_name, result)
                    cv.setWindowProperty(window_name, cv.WND_PROP_TOPMOST, 1)

                if not on_cooldown:    
                    # Do every 5th tick
                    if tick >= 5:
                        i += 1
                        # Run in separate thread to prevent lag
                        thread = threading.Thread(target=self.process_frame_async, args=(detector, frame, i))
                        thread.start()
                        tick = 0
                    
                    tick += 1

                    # If pose has been struck (person in video is still)
                    if self.check_stillness(frame, backSub, threshold):
                        # Needs the subject to be still for a second before captures frame
                        if start_capture_time is None:
                            start_capture_time = current_time
                            self.feedback = "CAPTURING"

                        elif current_time - start_capture_time >= pre_capture_delay:
                            self.capture_frame(image_name, frame)
                            # puts capture into a cooldown
                            on_cooldown = True
                            start_cooldown_time = current_time
                            start_capture_time = None
                    else:
                        # resets if subject is not still
                        start_capture_time = None
                        if self.latest_segmentation_mask is not None:
                            self.feedback = "MOVING"
                        else:
                            self.feedback = "NO PERSON\nIN FRAME"

                # Check if cooldown period has elapsed
                if on_cooldown:
                    if (current_time - start_cooldown_time >= capture_cooldown):
                        on_cooldown = False
                    else:
                        # This allows for the model to get a better idea of the background
                        # Also allows for start to not immediately detect you as still just because it does not yet know what is foreground and background
                        backSub.apply(frame)

        finally:
            # When everything done, close the pose detector
            detector.close()
            if __name__ == "__main__":
                cv.destroyAllWindows()

    def capture_frame(self, image_name, frame):
        frame_path = f"{image_name}_{self.frame_count}.png"
        cv.imwrite(frame_path, frame) # write image to image path
        self.frame_count += 1
        self.feedback = "CAPTURED"
        self.sound_channel.play(pygame.mixer.Sound('sounds/camera_click.wav'))

    def process_frame_async(self, detector, frame, frame_id):
        img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # Before passing to the detector, resize the frame to lessen computing power required
        scale_factor = 0.5
        resized_frame = cv.resize(img_rgb, (0, 0), fx=scale_factor, fy=scale_factor)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=resized_frame)
        detector.detect_async(mp_image, frame_id)         

    def check_stillness(self, frame, backgroundSub, threshold):
        # As latest_segmentation-mask updates asyncronously, a copy must be made within this if statement so it does not update to None mid synchrous process
        segmentation_mask = self.latest_segmentation_mask

        # Check if person was detected and if the mask has updated due to the async
        if segmentation_mask is not None:
            # Apply MoG background subtraction
            fg_mask = backgroundSub.apply(frame)

            # If running just the backend
            if __name__ == "__main__":
                cv.imshow('Segmentation Mask', segmentation_mask)
                cv.imshow('MoG', fg_mask)

            # Combine the pose mask with background subtraction to only consider the person area in calculation
            combined_mask = cv.bitwise_and(fg_mask, segmentation_mask)

            # Count total pixels in the person area
            total_pixels = np.sum(segmentation_mask > 0)

            if total_pixels > 0:  # Avoid division by zero
                # Count black pixels within person area
                black_pixels = np.sum((combined_mask == 0) & (segmentation_mask > 0))
                
                # Calculate percentage
                black_percentage = (black_pixels / total_pixels) * 100

                cv.putText(combined_mask, str(black_percentage), (10, 30), 
                            cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # If running just the backend
                if __name__ == "__main__":
                    cv.imshow('Mask', combined_mask)
                    
                return black_percentage > threshold
    
        return False

    def import_scratch_sprite(self, sb3_path, sprite_name, frame_count, exclude_images: list[int], blur_face: bool):
        # Extract the .sb3 file
        temp_dir = "temp_scratch_project"

        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            
            with zipfile.ZipFile(sb3_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            sprite_num = 0
            costumes = []
            for i in range(frame_count):
                if not i in exclude_images:
                    sprite_num +=1
                    frame_path = f"{sprite_name}_{i}.png"
                    # Get saved image for processing
                    mp_image = mp.Image.create_from_file(frame_path)

                    # Remove Background to Sprite
                    if self.remove_background(mp_image, blur_face):

                        # Get removed background image
                        temp_img_path = 'temp.png'
                        img = Image.open(temp_img_path)
                        img = img.convert('RGBA')

                        # Check if image is the correct size for a Scratch project, if not correct it
                        img = self.correct_img_size(img)

                        # Save image to memory
                        img_buffer = BytesIO()
                        img.save(img_buffer, format='PNG')
                        img_data = img_buffer.getvalue()

                        # Calculate MD5 hash
                        md5_hash = hashlib.md5(img_data).hexdigest()
                        
                        # Save the image in the project
                        costume_filename = f"{md5_hash}.png"
                        with open(os.path.join(temp_dir, costume_filename), 'wb') as f:
                            f.write(img_data)

                        costume = {
                            "assetId": md5_hash,
                            "name": f"{sprite_name}_{sprite_num}",
                            "md5ext": costume_filename,
                            "bitmapResolution": 1,
                            "dataFormat": "png",
                            "rotationCenterX": img.width // 2,
                            "rotationCenterY": img.height // 2
                        }
                        costumes.append(costume)
                    
                # Load project.json
                with open(os.path.join(temp_dir, 'project.json'), 'r') as f:
                    project_json = json.load(f)

            # Create new sprite
            sprite = {
                "isStage": False,
                "name": sprite_name,
                "variables": {},
                "lists": {},
                "broadcasts": {},
                "blocks": {},
                "comments": {},
                "currentCostume": 0,
                "costumes": costumes,
                "sounds": [],
                "volume": 100,
                "layerOrder": len(project_json["targets"]),
                "visible": True,
                "x": 0,
                "y": 0,
                "size": 100,
                "direction": 90,
                "draggable": False,
                "rotationStyle": "all around"
            }

            project_json["targets"].append(sprite)

            # Update project.json
            with open(os.path.join(temp_dir, 'project.json'), 'w') as f:
                json.dump(project_json, f)

            # Create new .sb3 file
            with zipfile.ZipFile(sb3_path, 'w') as zf:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_name = os.path.relpath(file_path, temp_dir)
                        zf.write(file_path, arc_name)
            
            # Cleanup
            for i in range(frame_count):
                frame_path = f"{sprite_name}_{i}.png"
                os.remove(frame_path)
            shutil.rmtree(temp_dir)
            if os.path.isfile(temp_img_path):
                os.remove(temp_img_path)

        except IOError as e:
            print("Error opening file: ", e)
        
        return sb3_path

    def correct_img_size(self, img):
        width, height = img.size

        # Check if image is too big
        max_width, max_height = 480, 360
        
        # Calculate scaling ratio
        width_ratio = max_width / width
        height_ratio = max_height / height
        ratio = min(width_ratio, height_ratio)
        
        if ratio < 1:
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return img

    def remove_background(self, image, blur_face: bool):
        running_mode = mp.tasks.vision.RunningMode.IMAGE
        detector = self.create_pose_detector(running_mode)

        # Detect pose landmarks from the input image.
        detection_result = detector.detect(image)

        # Check if segmentation masks exist
        if not detection_result.segmentation_masks:
            print("No person detected in the image")
            return False # if no person in image, skip frame
        
        if blur_face:
            # Parts of code in this blur_face if statement were generated by Claude AI. The parts that are original code are explicitly pointed out.
            image_array = image.numpy_view()
            # Create a mask from the face landmarks
            h, w, _ = image_array.shape
            face_landmarks = []

            for i in range(11): # 0-10 are face landmarks
                if i < len(detection_result.pose_landmarks[0]):
                    landmark = detection_result.pose_landmarks[0][i]
                    position = (int(landmark.x * w), int(landmark.y * h))
                    face_landmarks.append(position)

            if len(face_landmarks) > 2:
                # Find the center of the face (using nose as reference - usually landmark 0)
                nose = face_landmarks[0] if len(face_landmarks) > 0 else None
                right_ear = face_landmarks[7] if len(face_landmarks) > 7 else None
                left_ear = face_landmarks[8] if len(face_landmarks) > 8 else None

                if nose and right_ear and left_ear:
                    # Create face mask based on available landmarks
                    face_mask = np.zeros((h, w), dtype=np.uint8)

                    # This part and the following if statement are not AI generated
                    # This part is to detect whether the person in the stream is looking at the camera straight on or looking to the left or right
                    left_ear_to_nose = np.sqrt((left_ear[0] - nose[0])**2 + (left_ear[1] - nose[1])**2)  
                    right_ear_to_nose = np.sqrt((right_ear[0] - nose[0])**2 + (right_ear[1] - nose[1])**2)
                    ear_distance = np.sqrt((left_ear[0] - right_ear[0])**2 + (left_ear[1] - right_ear[1])**2)

                    radius = max(left_ear_to_nose, right_ear_to_nose, ear_distance)
                    if radius == ear_distance:
                        face_center = nose
                        x_radius = int(radius * 0.7)
                        y_radius = int(radius * 0.9)

                    elif radius == right_ear_to_nose: # If the furthest distance is from the nose to the right ear, the subject is looking left
                        face_center = (
                                int(nose[0] * 0.3 + right_ear[0] * 0.7),  # Center weighted toward ear
                                int((nose[1] + right_ear[1]) / 2)         # Vertical midpoint
                            )
                        x_radius = int(radius * 1.25)
                        y_radius = int(radius * 1.45)
                    else: # If the furthest distance is from the nose to the right ear, the subject is looking right
                        face_center = (
                                int(nose[0] * 0.3 + left_ear[0] * 0.7),   # Center weighted toward ear
                                int((nose[1] + left_ear[1]) / 2)          # Vertical midpoint
                            )
                        x_radius = int(radius * 1.25)
                        y_radius = int(radius * 1.45)
                        
                    cv.ellipse(face_mask, face_center, (x_radius, y_radius), 0, 0, 360, 255, -1)

                    # Determine blur strength based on detected face size
                    if right_ear and left_ear:
                        ear_distance = np.sqrt((left_ear[0] - right_ear[0])**2 + (left_ear[1] - right_ear[1])**2)
                        blur_strength = max(41, int(ear_distance * 0.3))
                    else:
                        blur_strength = 41  # Default value

                    # Ensure blur kernel size is odd
                    if blur_strength % 2 == 0:
                        blur_strength += 1

                    blurred_image = cv.GaussianBlur(image_array, (blur_strength, blur_strength), blur_strength/5)

                    # Create a normalized mask for alpha blending (values from 0 to 1)
                    normalized_mask = face_mask.astype(float) / 255.0
                    normalized_mask = np.expand_dims(normalized_mask, axis=2)
                    if image_array.shape[2] > 1:
                        normalized_mask = np.repeat(normalized_mask, image_array.shape[2], axis=2)
                        
                    # Apply the face mask to blend the blurred and original images
                    result_array = (blurred_image * normalized_mask + 
                                image_array * (1 - normalized_mask)).astype(np.uint8)  

                    # Convert back to mp_image
                    image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=result_array
                    )
                
        # Process the detection result and create visualization
        segmentation_mask = detection_result.segmentation_masks[0].numpy_view()

        # Convert image to BGR
        image_array = cv.cvtColor(image.numpy_view(), cv.COLOR_RGB2BGR)

        # Create RGBA image
        rgba_image = cv.cvtColor(image_array, cv.COLOR_BGR2BGRA)

        alpha_mask = (segmentation_mask * 255).astype(np.uint8)

        # Set alpha channel using the mask
        rgba_image[:, :, 3] = alpha_mask

        # Save image with transparency
        cv.imwrite('temp.png', rgba_image)

        return True

# To run just the backend.py
if __name__ == "__main__":
    scratch_file = input("Enter path to .sb3 file: ")
    sprite_name = input("Enter sprite name: ")
    
    args = parser.parse_args()

    sprite_maker = SpriteMaker()
    # Get around not having a button when running without
    sprite_maker.recording = True
    
    if args.use_file:
        camera_input = input("Enter path to video: ") # takes footage from video file rather than camera
    else:
        if args.usb_camera: # takes video from USB rather than main camera
            camera_input = 1
        else:
            camera_input = 0 # takes video from built-in camera

    try:
        cap = cv.VideoCapture(camera_input) # open the video feed from the webcam or video file

        if not cap.isOpened():
            print("Cannot open camera")
            exit()   

        # This is for debugging with test footage input
        if not type(camera_input) == int:
            sprite_maker.video_input = True

        sprite_maker.take_frames(cap, sprite_name)
        cap.release()

        frame_count = sprite_maker.frame_count
        if frame_count > 0:
            try:
                exclude_images = input("Type which images to exclude (start at zero, comma separated, leave blank if none): ")
                if exclude_images:
                    exclude_list = [int(x.strip()) for x in exclude_images.split(",")] # int(x.strip) is to delete spaces and convert string into ints
                else:
                    exclude_list = []    
            except Exception as e:
                print("Error in exclude list, excluding no images")
                exclude_list = []

            blur_face_input = input("Do you want to blur face? (y if yes): ")
            blur_face = (blur_face_input == "y")

            try:
                output_path = sprite_maker.import_scratch_sprite(scratch_file, sprite_name, frame_count, exclude_list, blur_face)
                print(f"\nSuccess! Modified project saved to: ", output_path)
            except Exception as e:
                print("Error during Scratch process: ", e)
        else:
            print("Camera closed before any images taken")

    except Exception as e:
        print("Error during capture: ", e)