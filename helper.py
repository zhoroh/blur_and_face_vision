from ultralytics import YOLO
import time
import os
import tempfile
import streamlit as st
import cv2
from pytube import YouTube
import moviepy.editor as mpy

import settings

file_out = tempfile.NamedTemporaryFile(delete=False,suffix='.mp4')


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(video_writer,conf, model, image,classes, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    

    # Resize the image to a standard size
    # image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, classes=classes, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image,classes=classes, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    # return res_plotted
    video_writer.write(res_plotted)
    # st_frame.image(res_plotted,
    #                caption='Detected Video',
    #                channels="BGR",
    #                use_column_width=True
    #                )


def play_youtube_video(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_youtube = st.sidebar.text_input("YouTube Video url")

    is_display_tracker, tracker = display_tracker_options()
    selected_options = st.sidebar.multiselect("Choose The classes of objects     you interested in", list(settings.CLASS_NAMES.keys()))
    if selected_options:
        message = "You selected:", ", ".join(selected_options)
        st.sidebar.success(message[0] + " " + message[1])
        classes_to_detect = [settings.CLASS_NAMES[key] for key in selected_options]
        if st.sidebar.button('Detect Objects'):
            st.sidebar.success("Processing Youtube Video......")
            try:
                yt = YouTube(source_youtube)
                stream = yt.streams.filter(file_extension="mp4", res=720).first()
                vid_cap = cv2.VideoCapture(stream.url)
                pathToWriteVideo='result_youtube.mp4'
                fourcc = cv2.VideoWriter_fourcc(*'mpv4')
                width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frames_per_second = vid_cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 2500:
                    total_frames = 2500
                checkpoints = [25, 50, 75, 100]
                current_frame = 0
                video_writer = cv2.VideoWriter(pathToWriteVideo, fourcc , fps=float(frames_per_second), frameSize=(width, height), isColor=True)
                while (vid_cap.isOpened()):
                    success, image = vid_cap.read()
                    percent_complete = (current_frame / total_frames) * 100
                    if checkpoints and percent_complete >= checkpoints[0]:
                        st.sidebar.success(f"Video Processing is {checkpoints.pop(0)}% completed.")
                    current_frame += 1
                    if current_frame < total_frames + 1: # youtube videos can be very very long

                        if success:
                            _display_detected_frames(video_writer,
                                                        conf,
                                                        model,
                                                        image,
                                                        classes_to_detect,
                                                        is_display_tracker,
                                                        tracker
                                                        )
                        
                        else:
                            video_writer.release()
                            vid_cap.release()
                            break
                    else:
                        ""
                        video_writer.release()
                        vid_cap.release()
                        break
                
                st_video = open('result_youtube.mp4','rb')
                video_bytes = st_video.read()
                st.video(video_bytes)
                st.write("Detected Video") 
            except Exception as e:
                st.sidebar.error("Error loading video: " + str(e))
    else:
        st.sidebar.error("No classes Selected")

def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)
    
    selected_options = st.sidebar.multiselect("Choose The classes of objects     you interested in", list(settings.CLASS_NAMES.keys()))
    if selected_options:
        message = "You selected:", ", ".join(selected_options)
        st.sidebar.success(message[0] + " " + message[1])
        classes_to_detect = [settings.CLASS_NAMES[key] for key in selected_options]
        if st.sidebar.button('Detect  Objects'):
            st.sidebar.success("Processing Video......")
            try:
                vid_cap = cv2.VideoCapture(
                    str(settings.VIDEOS_DICT.get(source_vid)))
                pathToWriteVideo = file_out.name
                fourcc = cv2.VideoWriter_fourcc(*'mpv4')
                width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frames_per_second = vid_cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 2500:
                    total_frames = 2500
                checkpoints = [25, 50, 75, 100]

                # Read and process the video frame by frame
                current_frame = 0
                video_writer = cv2.VideoWriter(pathToWriteVideo, fourcc , fps=float(frames_per_second), frameSize=(width, height), isColor=True)
                # video_row=[]
                while (vid_cap.isOpened()):
                    success, image = vid_cap.read()
                    percent_complete = (current_frame / total_frames) * 100
                    if checkpoints and percent_complete >= checkpoints[0]:
                        st.sidebar.success(f"Video Processing is {checkpoints.pop(0)}% completed.")
                    current_frame += 1
                    if current_frame < total_frames + 1: #  videos can be very very long

                        if success:
                            _display_detected_frames(video_writer,
                                                        conf,
                                                        model,
                                                        image,
                                                        classes_to_detect,
                                                        is_display_tracker,
                                                        tracker
                                                        )
                            # video_row.append(new_frame)
                        
                        else:
                            video_writer.release()
                            vid_cap.release()
                            break
                    else:
                        ""
                        video_writer.release()
                        vid_cap.release()
                        break
                # clip = mpy.ImageSequenceClip(video_row, fps=frames_per_second)
                # clip.write_videofile(pathToWriteVideo)
                # st_video = open(pathToWriteVideo,'rb')
                # video_bytes = st_video.read()
                # st.video(video_bytes)
                st.video(pathToWriteVideo.name)
                result_video = open(pathToWriteVideo.name, "rb")
                st.download_button(label="Download video file", data=result_video,file_name='video_clip.mp4')
                st.write("Detected Video") 
            except Exception as e:
                st.sidebar.error("Error loading video: " + str(e))

    else:
        st.sidebar.error("No classes Selected")
    
