import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import cv2, queue, threading, time
import paho.mqtt.client as mqttClient
 
def on_connect(client, userdata, flags, rc):
 
    if rc == 0:
 
        print("Connected to broker")
 
        global Connected                #Use global variable
        Connected = True                #Signal connection 
 
    else:
 
        print("Connection failed")
 
Connected = False   #global variable for the state of the connection
 
broker_address= "192.168.0.104"
port = 1883
#user = "yourUser"
#password = "yourPassword"
 
client = mqttClient.Client("Python")               #create new instance
#client.username_pw_set(user, password=password)    #set username and password
client.on_connect= on_connect                      #attach function to callback
client.connect(broker_address, port=port)          #connect to broker
 
client.loop_start()        #start the loop
 
while Connected != True:    #Wait for connection
    time.sleep(0.1)

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose

# Setup the Pose function for images - independently for the images standalone processing.
pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Setup the Pose function for videos - for video processing.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)

# Initialize mediapipe drawing class - to draw the landmarks points.
mp_drawing = mp.solutions.drawing_utils

# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()

cap = VideoCapture(('http://127.0.0.1:8080/video'))
#cap = cv2.VideoCapture('http://127.0.0.1:8080/video')

#if (cap.isOpened() == False):
#    print("Error opening video stream or file")

def calc_angle(a1,b1,c1,d,e,f):
            a = np.array([a1, b1]) # First coord
            b = np.array([c1, d]) # Second coord
            c = np.array([e, f]) # Third coord
            
            radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            
            if angle > 180.0:
                angle = 360-angle

            return angle

def detectPose(image_pose, pose, draw=False, display=False):
          original_image = image_pose.copy()
    
          image_in_RGB = cv2.cvtColor(image_pose, cv2.COLOR_BGR2RGB)
    
          resultant = pose.process(image_in_RGB)
        # Save landmarks.
          if resultant.pose_landmarks is not None:

        # Map pose landmarks from [0, 1] range to absolute coordinates to get
        # correct aspect ratio.
              frame_height, frame_width = original_image.shape[:2]

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

              lpX = resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY].x * frame_width
              lpY = resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY].y * frame_height
              rpX = resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY].x * frame_width
              rpY = resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY].y * frame_height

              la = calc_angle(lsX, lsY, leX, leY, lwX,lwY)
              ra = calc_angle(rsX, rsY, reX, reY, rwX,rwY)

              if(la < 50 and ra < 50 and abs(lpX - rpX) < 100 and abs(lpY - rpY) < 100 ):                
                
                print("Folded hands detected !")
                return 1

              else:
                print("--")
                return 0
                


# Read until video is completed
'''while(cap.isOpened()):
    ret = True
    frame = cap.read()
    if ret == True:'''
while(True):
        frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        retcode = detectPose(image, pose_image, draw=False, display=False)
        if(retcode == 1):
          client.publish("python/test","Folded hands detected !")
          time.sleep(5)
          client.publish("python/test","Done")
          

        
    # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
client.disconnect()
client.loop_stop()

