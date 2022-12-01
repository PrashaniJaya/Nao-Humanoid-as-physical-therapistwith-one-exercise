import numpy as np
import cv2
import matplotlib as plt
from naoqi import ALProxy
import time
import math

a = []
# Specify the paths for the 2 files for coco
protoFile = "pose_deploy_linevec.prototxt"
weightsFile = "pose_iter_440000.caffemodel"

# Specify the paths for the 2 files for body_25
#protoFile = "pose_deploy.prototxt"
#weightsFile = "pose_iter_584000.caffemodel"

# net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

thr = 0.2

BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]



def pose_estimation(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0 / 255, (width, height), (0, 0, 0), swapRB=False, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]
    assert (len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponding body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > thr else None)
        # coord(points)
        # if we print here it will not work because append occurs meaning data keeps adding so there will be an error

    #a = points
    #x1 = points[2][1]
    #print(x1)
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert (partFrom in BODY_PARTS)
        assert (partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    t, _ = net.getPerfProfile()
    freq = cv2.getTickFrequency() / 1000
    cv2.putText(frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    return frame, points


def distance(cx1, cy1, cx2, cy2):
    # euclidean distance method to find distance between 2 points
    if cx1 == None: cx1 = (0, 0)
    if cy1 == None: cx1 = (0, 0)
    if cx2 == None: cx1 = (0, 0)
    if cy2 == None: cx1 = (0, 0)
    return math.sqrt(abs(cx2 - cx1) ** 2 + abs(cy2 - cy1) ** 2)

def angle(p0x, p0y, p1x, p1y, p2x, p2y):
    if p0x == None: cx1 = (0, 0)
    if p0y == None: cx1 = (0, 0)
    if p1x == None: cx1 = (0, 0)
    if p1y == None: cx1 = (0, 0)
    if p2x == None: cx1 = (0, 0)
    if p2y == None: cx1 = (0, 0)
    try:
        a = (p1x-p0x)**2 + (p1y-p0y)**2
        b = (p1x-p2x)**2 + (p1y-p2y)**2
        c = (p2x-p0x)**2 + (p2y-p0y)**2
        angle = math.acos((a+b-c) / math.sqrt(4*a*b) ) * 180/math.pi
    except:
        return 0
    return int(angle)

tts = ALProxy("ALTextToSpeech", "192.168.0.158", 9559)
tts.say("Hello everybody today we will be doing the T stretch exercise, so now i will analyse your position, please lift your arms to the shoulder level until your body forms a T position")

# get NAOqi module proxy
videoDevice = ALProxy('ALVideoDevice', "192.168.0.158", 9559)


# subscribe top camera
AL_kTopCamera = 0
AL_kQVGA = 1     # 320x240 http://doc.aldebaran.com/2-1/family/robots/video_robot.html#cameraresolution-mt9m114
AL_kBGRColorSpace = 13   # Buffer contains triplet on the format 0xRRGGBB, equivalent to three unsigned char
# ALVideoDeviceProxy::subscribeCameras(Name,  CameraIndexes,  Resolutions,  ColorSpaces,  Fps)
captureDevice = videoDevice.subscribeCamera("test", AL_kTopCamera, AL_kQVGA, AL_kBGRColorSpace, 30) # 30

# create image
width = 320
height = 240
image = np.zeros((height, width, 3), np.uint8)

while True:

    # get image
    result = videoDevice.getImageRemote(captureDevice)

    if result == None:
        print 'cannot capture.'
    elif result[6] == None:
        print 'no image data string.'
    else:

        # translate value to mat
        values = map(ord, list(result[6]))
        i = 0
        for y in range(0, height):
            for x in range(0, width):
                image.itemset((y, x, 0), values[i + 0])
                image.itemset((y, x, 1), values[i + 1])
                image.itemset((y, x, 2), values[i + 2])
                i += 3

        # show image
        #heck, frame = image.read()
        estimated_image, coordinates = pose_estimation(image)
        cv2.imshow("nao-top-camera-320x240", estimated_image)

        # now i calculate the angles and distances for T stretch pose
        threlbow = 80
        threlbownose = 86
        # this is for the distances
        # nose
        x1 = coordinates[0][0]
        y1 = coordinates[0][1]
        # neck
        x2 = coordinates[1][0]
        y2 = coordinates[1][1]
        # shoulder 2
        s1 = coordinates[2][0]
        s2 = coordinates[2][1]
        # elbow
        x3 = coordinates[3][0]
        y3 = coordinates[3][1]
        # wrist 4
        w1 = coordinates[4][0]
        w2 = coordinates[4][1]


        # opposite = distance(x1, y1, x2, y2)
        adjacent = distance(x2, y2, x3, y3)
        hypotenuse = distance(x3, y3, x1, y1)
        #angles
        anglehand = angle(s1, s2, x3, y3, w1, w2)
        # theory_hypotenuse = math.sqrt(opposite ** 2 + adjacent ** 2)
        theoryresult = int((threlbownose * adjacent) / threlbow)
        print("hypotenuse, theoryresult, anglehand")
        print(hypotenuse, theoryresult, anglehand)

        #plt.imshow(cv2.cvtColor(estimated_image, cv2.COLOR_BGR2RGB))
        time.sleep(2)

        #time.sleep(3)
        #if hypotenuse in range(theoryresult -10, theoryresult+10) and anglehand in range(178, 182):
           #  tts.say("wow, you are doing a good job")
        if hypotenuse < theoryresult -10:
            tts.say("could you lower your hand a little")
            print("could you lower your hand a little")
        elif hypotenuse > theoryresult +10:
            tts.say("could you lift your hand a little")
            print("could you lift your hand a little")
        elif anglehand < 165:
            tts.say("make sure not to bend your hand")
            print("make sure not to bend your hand")
        else:
            print("good")
            tts.say("wow, you are doing a good job")



    # exit by [ESC]
    if cv2.waitKey(33) == 27:
        break

#image.release()

cv2.destroyAllWindows()

