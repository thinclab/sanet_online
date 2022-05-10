#!/usr/bin/env python3

from copy import copy
import numpy as np
import rospy
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image, CameraInfo
import sys
import rospkg
rospack = rospkg.RosPack()  # get an instance of RosPack with the default search paths
path = rospack.get_path('sanet_online')   # get the file path for sanet_onionsorting
sys.path.append(path)
from other_scripts.human_sanet import TestStateAction
from time import time
rgb_mem = None
dataset = 'human'
sa = TestStateAction()

def grabrgb(msg):

    global rgb_mem, sa
    # print("We're in callback!")
    if msg is not None:
        t0 = time()
        rgb_mem = copy(msg)
        rgb_mem = np.frombuffer(rgb_mem.data, dtype=np.uint8).reshape(rgb_mem.height, rgb_mem.width, -1).astype('float32')     # Added by Prasanth Suresh
        state_frame_list, action_frame_list = [[rgb_mem], [rgb_mem]*5]
        sa.main(state_frame_list, action_frame_list)
        print("Total time taken: ", time() - t0)
    return

def main():
    try:
        rospy.init_node("sanet_online")
        rate = rospy.Rate(10)
        rospy.loginfo("Sanet online node started")
        rospy.Subscriber("/kinect2/hd/image_color", Image, grabrgb)       
        # rate.sleep()
    except rospy.ROSInterruptException:
        print(rospy.ROSInterruptException)
        rospy.signal_shutdown()
    except KeyboardInterrupt:
        print(KeyboardInterrupt)
        rospy.signal_shutdown()
    rospy.spin()


if __name__ == '__main__':
    main()