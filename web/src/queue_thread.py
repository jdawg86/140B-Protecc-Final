import cv2
from queue import Queue
from threading import Thread
import time
import face_recognition
import glob
from os.path import join
import numpy as np
import PIL

video=cv2.VideoCapture(0)
baseline_image=None

# A thread that produces data
def motion_detection(out_q):
    while True:
        time.sleep(1)
        global baseline_image
        global video
        # Produce some data
        check, frame = video.read()
        status=0
        gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray_frame=cv2.GaussianBlur(gray_frame,(25,25),0)
        color_frame = frame

        if baseline_image is None or out_q.qsize() > 1000:
            baseline_image=gray_frame
            continue

        delta=cv2.absdiff(baseline_image,gray_frame)
        min = 50
        threshold=cv2.threshold(delta, min, 255, cv2.THRESH_BINARY)[1]
        (contours,_)=cv2.findContours(threshold,cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)

        box_frame = None
        motion = False
        for contour in contours:
            if cv2.contourArea(contour) > 10000:
                (x, y, w, h)=cv2.boundingRect(contour)
                box_frame = cv2.rectangle(frame,(x, y), (x+w, y+h), (0,255,0), 1)
                #
                frame_dict = {'gray_frame' : gray_frame,
                              'delta_frame' : delta,
                              'thresh_frame' : threshold,
                              'color_frame' : color_frame,
                              'box_frame' : box_frame}
                motion = True
        if(motion):
            out_q.put(frame_dict)
            timestr = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(f"public/motion_cap/motion_{timestr}.jpg", frame_dict['color_frame'])
            # print("Motion!")
            # print("out_q: " + str(out_q.qsize()))

# A thread that consumes data
def facial_recognition(in_q, known_faces, known_faces_enc):
    while True:
        # Get some data
        # print("in_q: " + str(in_q.qsize()))
        frames = in_q.get()
        # Process the data
        # color_frame = np.asarray(frames['color_frame'])
        color_frame = frames['color_frame']
        small_frame = cv2.resize(frames['color_frame'], (0, 0), fx=0.25, fy=0.25)


        compare_enc = face_recognition.face_encodings(small_frame)
        name = ""
        if(len(compare_enc) != 0):
            # print("found matching face: ")
            for found in compare_enc:
                matches = face_recognition.compare_faces(known_faces_enc, found)
                if True in matches:
                    first_match_index = matches.index(True)
                    name = str(known_faces[first_match_index])
                print("found matching face: " + str(name))
        else:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(f"public/unknown_faces/unknown_{timestr}.jpg", color_frame)
        # Indicate completion
        in_q.task_done()

    # Create the shared queue and launch both threads
if __name__ == "__main__":
    known_faces_enc = []

    image_types = ('*.img', '*.jpg', '*.jpeg', '*.png')
    image_paths = []
    known_face_names = []
    for files in image_types:
        image_paths.extend(glob.glob((join("public/known_faces/", files))))


    for p in image_paths:
        print(p)
        face_img = face_recognition.load_image_file(p)
        if (len(face_img) == 0):
            print("No Faces Found in " + str(p) )
            continue
        known_face_names.append(p)
        known_faces_enc.append(face_recognition.face_encodings(face_img)[0])    # [0] first face
    q = Queue()
    facial_recog = Thread(target = facial_recognition, args =(q, known_face_names, known_faces_enc, ))
    motion = Thread(target = motion_detection, args =(q, ))
    motion.start()
    facial_recog.start()
    print("Threads Started")
    # Wait for all produced items to be consumed
    q.join()
