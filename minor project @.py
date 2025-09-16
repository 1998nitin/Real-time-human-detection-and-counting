import cv2
import imutils
import numpy as np
import argparse


def detect(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    person = 1
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'person {person}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        person += 1

    cv2.putText(frame, 'Status: Detecting ', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f'Total Persons: {person-1}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow('output', frame)

    return frame

def detectByPathVideo(path, output_path, face_cascade):
    video = cv2.VideoCapture(path)
    check, frame = video.read()
    if not check:
        print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
        return

    # Define the codec and create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(output_path, fourcc, 10, (frame.shape[1], frame.shape[0]))

    print('Detecting people...')
    while video.isOpened():
        check, frame =  video.read()

        if check:
            frame = imutils.resize(frame, width=min(800, frame.shape[1]))
            frame = detect(frame, face_cascade)
            
            # Save each frame to the output video
            writer.write(frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:
            break
    video.release()
    writer.release()
    cv2.destroyAllWindows()

def detectByCamera(output_path, face_cascade):   
    video = cv2.VideoCapture(0)

    # Define the codec and create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(output_path, fourcc, 10, (800, 800))

    print('Detecting people...')
    while True:
        check, frame = video.read()

        frame = detect(frame, face_cascade)
        # Save each frame to the output video
        writer.write(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    video.release()
    writer.release()
    cv2.destroyAllWindows()

def detectByPathImage(path, output_path, face_cascade):
    image = cv2.imread(path)

    image = imutils.resize(image, width=min(800, image.shape[1])) 

    result_image = detect(image, face_cascade)

    if output_path is not None:
        cv2.imwrite(output_path, result_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def humanDetector(method, args, face_cascade):
    if method == 'camera':
        detectByCamera(args['output'], face_cascade)
    elif method == 'video':
        detectByPathVideo(args['video'], args['output'], face_cascade)
    elif method == 'image':
        detectByPathImage(args['image'], args['output'], face_cascade)
    else:
        print('Invalid method selected.')

def argsParser(): 
    arg_parse = argparse.ArgumentParser()         
    arg_parse.add_argument("-v", "--video", default="video.mp4", help="path to Video File ")  
    arg_parse.add_argument("-i", "--image", default="minor.jpg", help="path to Image File ")  
    arg_parse.add_argument("-c", "--camera", default=True, help="Set true if you want to use the camera.")  
    arg_parse.add_argument("-o", "--output", type=str, default="C:/Users/ABCD/Desktop/Output/output.jpg", help="path to optional output file")  # Corrected the default value
    args = vars(arg_parse.parse_args())

    return args


if __name__ == "__main__":
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    args = argsParser()

    # Ask the user for the method choice
    method_choice = input("Choose method (camera, video, image): ").lower()

    # Add a switch case for the user's choice
    humanDetector(method_choice, args, face_cascade)

