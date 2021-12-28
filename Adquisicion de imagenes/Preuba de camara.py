# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 18:52:31 2021

@author: Johan Cuervo
"""

import PySpin
import sys


system = PySpin.System.GetInstance()

version = system.GetLibraryVersion()
print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

cam_list = system.GetCameras()

print(cam_list[0])
num_cameras = cam_list.GetSize()






#system.ReleaseInstance()