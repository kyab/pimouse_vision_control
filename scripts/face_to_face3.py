#!/usr/bin/env python
#encoding: utf8
import rospy, cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class FaceToFace3():
    def __init__(self):
        sub = rospy.Subscriber("/cv_camera/image_raw", Image, self.get_image)
        self.bridge = CvBridge()
        self.image_org = None

        self.pub = rospy.Publisher("face", Image, queue_size=1)

    def get_image(self, img):
        try:
            self.image_org = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

    def monitor(self, rect,org):
        if rect is not None:
            cv2.rectangle(org, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), (0, 255, 255), 4)

        self.pub.publish(self.bridge.cv2_to_imgmsg(org, "bgr8"))


    def detect_face(self):
        if self.image_org is None:
            return None

        org = self.image_org

        gimg = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
        classsfier = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(classsfier)
        face = cascade.detectMultiScale(
            gimg, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1), flags=cv2.CASCADE_FIND_BIGGEST_OBJECT)

        if len(face) == 0:
            self.monitor(None, org)
            return None

        r = face[0]
        self.monitor(r, org)
        return r

if __name__ == "__main__":
    rospy.init_node("face_to_face2")
    fd = FaceToFace3()

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rospy.loginfo(fd.detect_face())
        rate.sleep()
