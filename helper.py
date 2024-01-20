from ultralytics import YOLO
import time
import os
import tempfile
import streamlit as st
import cv2
from pytube import YouTube

import settings

# def download_youtube_video_to_tempfile(url):
#     """
#     Downloads a YouTube video to a temporary file.

#     Parameters:
#         url (str): The URL of the YouTube video.

#     Returns:
#         str: The path to the temporary file.
#     """

#     # Create a YouTube object
#     yt = YouTube(url)

#     # Select the highest resolution stream available
#     video_stream = yt.streams.filter(file_extension="mp4", res=720).first()
#     if not video_stream:
#         raise Exception("No suitable video stream found")

#     # Create a temporary file
#     temp_video_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)

#     print("Downloading video...")
#     # Download the video directly to the temporary file
#     video_stream.stream_to_buffer(temp_video_file)

#     # Close the file (necessary before trying to access it on some systems)
#     temp_video_file.close()

#     print(f"Video downloaded to {temp_video_file.name}")
#     return temp_video_file.name


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


# def play_uploaded_video(conf, model):
    
#     video_data = st.sidebar.file_uploader("Upload video", ['mp4','mov', 'avi'])
#     if video_data:
        
#         temp_file_1 = tempfile.NamedTemporaryFile(delete=False,suffix='.mp4')
#         temp_file_1.write(video_data.getbuffer())
#         is_display_tracker, tracker = display_tracker_options()
#         selected_options = st.sidebar.multiselect("Choose The classes of objects     you interested in", list(settings.CLASS_NAMES.keys()))
#         if selected_options:
#             message = "You selected:", ", ".join(selected_options)
#             st.sidebar.success(message[0] + " " + message[1])
#             classes_to_detect = [settings.CLASS_NAMES[key] for key in selected_options]
#             if st.sidebar.button('Detect Objects'):
#                 st.sidebar.success("Processing Youtube Video......")
#                 try:
#                     vid_cap_yt = cv2.VideoCapture(temp_file_1.name)
#                     file_out_yt = tempfile.NamedTemporaryFile(suffix='.mp4')
#                     pathToWriteVideo = file_out_yt.name
#                     fourcc = cv2.VideoWriter_fourcc(*'mpv4')
#                     width = int(vid_cap_yt.get(cv2.CAP_PROP_FRAME_WIDTH))
#                     height = int(vid_cap_yt.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                     frames_per_second = vid_cap_yt.get(cv2.CAP_PROP_FPS)
#                     total_frames = int(vid_cap_yt.get(cv2.CAP_PROP_FRAME_COUNT))
#                     if total_frames > 2500:
#                         total_frames = 2500
#                     checkpoints = [25, 50, 75, 100]
#                     current_frame = 0
#                     video_writer_yt = cv2.VideoWriter(pathToWriteVideo, fourcc , fps=float(frames_per_second), frameSize=(width, height), isColor=True)
#                     while (vid_cap_yt.isOpened()):
#                         success, image = vid_cap_yt.read()
#                         percent_complete = (current_frame / total_frames) * 100
#                         if checkpoints and percent_complete >= checkpoints[0]:
#                             st.sidebar.success(f"Video Processing is {checkpoints.pop(0)}% completed.")
#                         current_frame += 1
#                         if current_frame < total_frames + 1: # youtube videos can be very very long
    
#                             if success:
#                                 _display_detected_frames(video_writer_yt,
#                                                          conf,
#                                                          model,
#                                                          image,
#                                                          classes_to_detect,
#                                                          is_display_tracker,
#                                                          tracker
#                                                             )
                            
#                             else:
#                                 video_writer_yt.release()
#                                 vid_cap_yt.release()
#                                 break
#                         else:
#                             ""
#                             video_writer_yt.release()
#                             vid_cap_yt.release()
#                             break
                    
#                     # st_video = open('result_youtube.mp4','rb')
#                     # video_bytes = st_video.read()
#                     # st.video(video_bytes)
#                     # st.write("Detected Video") 
#                     result_video_yt = open(pathToWriteVideo, "rb")
#                     st.download_button(label="Download Youtube Results", data=result_video_yt,file_name='ytube_results.mp4')
#                 except Exception as e:
#                     st.sidebar.error("Error loading video: " + str(e))
#         else:
#             st.sidebar.error("No classes Selected")
#     else:
#         st.sidebar.error("Please Upload a Video")

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
                file_out = tempfile.NamedTemporaryFile(suffix='.mp4')
                pathToWriteVideo = file_out.name
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
                # file_size = os.path.getsize(pathToWriteVideo)
                # print("vujvycbvhgjhf")
                # print(file_size)
                # st.video(pathToWriteVideo)
                result_video = open(pathToWriteVideo, "rb")
                st.download_button(label="Download Results", data=result_video,file_name='results.mp4')
                # st.write("Detected Video") 
            except Exception as e:
                st.sidebar.error("Error loading video: " + str(e))

    else:
        st.sidebar.error("No classes Selected")
    
