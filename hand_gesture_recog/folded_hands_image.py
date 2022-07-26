import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose

# Setup the Pose function for images - independently for the images standalone processing.
pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Setup the Pose function for videos - for video processing.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)

# Initialize mediapipe drawing class - to draw the landmarks points.
mp_drawing = mp.solutions.drawing_utils

import cv2

cap = cv2.VideoCapture('http://127.0.0.1:8080/video')
ret,frame = cap.read() # return a single frame in variable `frame`

#while(True):
    #cv2.imshow('img1',frame) #display the captured image
    #if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y' 
cv2.imwrite('c1.png',frame)
cv2.destroyAllWindows()
#        break

cap.release()

def calc_angle(a1,b1,c1,d,e,f):
            a = np.array([a1, b1]) # First coord
            b = np.array([c1, d]) # Second coord
            c = np.array([e, f]) # Third coord
            
            radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            
            if angle > 180.0:
                angle = 360-angle

            print(angle) 

def detectPose(image_pose, pose, draw=False, display=False):
          original_image = image_pose.copy()
    
          image_in_RGB = cv2.cvtColor(image_pose, cv2.COLOR_BGR2RGB)
    
          resultant = pose.process(image_in_RGB)

          if resultant.pose_landmarks and draw:    

              mp_drawing.draw_landmarks(image=original_image, landmark_list=resultant.pose_landmarks,
                                        connections=mp_pose.POSE_CONNECTIONS,
                                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255),
                                                                               thickness=10, circle_radius=20),
                                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(49,125,237),
                                                                               thickness=10, circle_radius=10))

          if display:
              plt.figure(figsize=[22,22])
              plt.subplot(121);plt.imshow(image_pose[:,:,::-1]);plt.title("Input Image");plt.axis('off');
              plt.subplot(122);plt.imshow(original_image[:,:,::-1]);plt.title("Pose detected Image");plt.axis('off');
          else:
              return original_image, results

          # Save landmarks.
          if resultant.pose_landmarks is not None:
        # Check the number of landmarks and take pose landmarks.
              assert len(resultant.pose_landmarks.landmark) == 33, 'Unexpected number of predicted pose landmarks: {}'.format(len(resultant.pose_landmarks.landmark))
              pose_landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in resultant.pose_landmarks.landmark]

        # Map pose landmarks from [0, 1] range to absolute coordinates to get
        # correct aspect ratio.
              frame_height, frame_width = original_image.shape[:2]
              print(
                  f'Image Size: ('
                  f'{frame_width}, '
                  f'{frame_height})'
                  )
              pose_landmarks *= np.array([frame_width, frame_height, frame_width])

       
              if not resultant.pose_landmarks:
                  return
              print(
                  f'Nose coordinates: ('
                  f'{resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * frame_width}, '
                  f'{resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * frame_height})'
                  )
              print(
                  f'Left Shoulder coordinates: ('
                  f'{resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame_width}, '
                  f'{resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height})'
                  )
              print(
                  f'Left Wrist coordinates: ('
                  f'{resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * frame_width}, '
                  f'{resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * frame_height})'
                  )
              print(
                  f'Left Pinky coordinates: ('
                  f'{resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY].x * frame_width}, '
                  f'{resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY].y * frame_height})'
                  )
              print(
                  f'Left Elbow coordinates: ('
                  f'{resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * frame_width}, '
                  f'{resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * frame_height})'
                  )
              print(
                  f'Right Shoulder coordinates: ('
                  f'{resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame_width}, '
                  f'{resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height})'
                  )
              print(
                  f'Right Wrist coordinates: ('
                  f'{resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame_width}, '
                  f'{resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame_height})'
                  )
              print(
                  f'Right Pinky oordinates: ('
                  f'{resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY].x * frame_width}, '
                  f'{resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY].y * frame_height})'
                  )
              print(
                  f'Right Elbow coordinates: ('
                  f'{resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * frame_width}, '
                  f'{resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * frame_height})'
                  )
                  
              
              lsX = resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame_width
              lsY = resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height
              lwX = resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * frame_width
              lwY = resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * frame_height
              leX = resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * frame_width
              leY = resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * frame_height

              rsX = resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame_width
              rsY = resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height
              rwX = resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame_width
              rwY = resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame_height
              reX = resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * frame_width
              reY = resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * frame_height

              calc_angle(lsX, lsY, leX, leY, lwX,lwY)
              calc_angle(rsX, rsY, reX, reY, rwX,rwY)



# Here we will read our image from the specified path to detect the pose
image_path = 'c1.png'
output = cv2.imread(image_path)
detectPose(output, pose_image, draw=True, display=True)