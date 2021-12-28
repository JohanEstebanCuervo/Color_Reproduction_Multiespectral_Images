# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 16:49:42 2021

@author: Johan Cuervo
"""

import serial
import time
import warnings
import PySpin
import sys

class TriggerType:
    SOFTWARE = 1
    HARDWARE = 2

CHOSEN_TRIGGER = TriggerType.HARDWARE

def configure_trigger(cam,nodemap):
    """
    This function configures the camera to use a trigger. First, trigger mode is
    ensured to be off in order to select the trigger source. Trigger mode is
    then enabled, which has the camera capture only a single image upon the
    execution of the chosen trigger.

     :param cam: Camera to configure trigger for.
     :type cam: CameraPtr
     :return: True if successful, False otherwise.
     :rtype: bool
    """

    print('*** CONFIGURING TRIGGER ***\n')

    print('Note that if the application / user software triggers faster than frame time, the trigger may be dropped / skipped by the camera.\n')
    print('If several frames are needed per trigger, a more reliable alternative for such case, is to use the multi-frame mode.\n\n')

    if CHOSEN_TRIGGER == TriggerType.SOFTWARE:
        print('Software trigger chosen...')
    elif CHOSEN_TRIGGER == TriggerType.HARDWARE:
        print('Hardware trigger chose...')

    try:
        result = True
        # Turn off auto exposure
        node_exposure_auto = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
        if not PySpin.IsAvailable(node_exposure_auto) or not PySpin.IsWritable(node_exposure_auto):
            print('\nUnable to set Exposure Auto (enumeration retrieval). Aborting...\n')
            return False

        entry_exposure_auto_off = node_exposure_auto.GetEntryByName('Off')
        if not PySpin.IsAvailable(entry_exposure_auto_off) or not PySpin.IsReadable(entry_exposure_auto_off):
            print('\nUnable to set Exposure Auto (entry retrieval). Aborting...\n')
            return False

        exposure_auto_off = entry_exposure_auto_off.GetValue()

        node_exposure_auto.SetIntValue(exposure_auto_off)

        # Set Exposure Time to less than 1/50th of a second (5000 us is used as an example)
        node_exposure_time = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
        if not PySpin.IsAvailable(node_exposure_time) or not PySpin.IsWritable(node_exposure_time):
            print('\nUnable to set Exposure Time (float retrieval). Aborting...\n')
            return False

        node_exposure_time.SetValue(8000)
        
        # Turn off Gain
        node_gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
        if not PySpin.IsAvailable(node_gain_auto) or not PySpin.IsWritable(node_gain_auto):
            print('\nUnable to set Gain Auto (enumeration retrieval). Aborting...\n')
            return False

        entry_gain_auto_off = node_gain_auto.GetEntryByName('Off')
        if not PySpin.IsAvailable(entry_gain_auto_off) or not PySpin.IsReadable(entry_gain_auto_off):
            print('\nUnable to set Exposure Auto (entry retrieval). Aborting...\n')
            return False

        gain_auto_off = entry_gain_auto_off.GetValue()

        node_gain_auto.SetIntValue(gain_auto_off)

        node_gain = PySpin.CFloatPtr(nodemap.GetNode('Gain'))
        if not PySpin.IsAvailable(node_gain) or not PySpin.IsWritable(node_gain):
            print('\nUnable to set Exposure Time (float retrieval). Aborting...\n')
            return False

        node_gain.SetValue(10000)
        
        # Ensure trigger mode off
        # The trigger must be disabled in order to configure whether the source
        # is software or hardware.
        if cam.TriggerMode.GetAccessMode() != PySpin.RW:
            print('Unable to disable trigger mode (node retrieval). Aborting...')
            return False

        cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)

        print('Trigger mode disabled...')
        
        # Set TriggerSelector to FrameStart
        # For this example, the trigger selector should be set to frame start.
        # This is the default for most cameras.
        if cam.TriggerSelector.GetAccessMode() != PySpin.RW:
            print('Unable to get trigger selector (node retrieval). Aborting...')
            return False
            
        cam.TriggerSource.SetValue(PySpin.TriggerSelector_FrameStart)

        print('Trigger selector set to frame start...')
        
        # Select trigger source
        # The trigger source must be set to hardware or software while trigger
		# mode is off.
        if cam.TriggerSource.GetAccessMode() != PySpin.RW:
            print('Unable to get trigger source (node retrieval). Aborting...')
            return False

        if CHOSEN_TRIGGER == TriggerType.SOFTWARE:
            cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)
            print('Trigger source set to software...')
        elif CHOSEN_TRIGGER == TriggerType.HARDWARE:
            cam.TriggerSource.SetValue(PySpin.TriggerSource_Line2) #line 0,1,2
            print('Trigger source set to hardware...')

        # Turn trigger mode on
        # Once the appropriate trigger source has been set, turn trigger mode
        # on in order to retrieve images using the trigger.
        cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
        print('Trigger mode turned back on...')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result

def grab_next_image_by_trigger(cam,Led):
    """
    This function acquires an image by executing the trigger node.

    :param cam: Camera to acquire images from.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True
        # Use trigger to capture image
        # The software trigger only feigns being executed by the Enter key;
        # what might not be immediately apparent is that there is not a
        # continuous stream of images being captured; in other examples that
        # acquire images, the camera captures a continuous stream of images.
        # When an image is retrieved, it is plucked from the stream.

        if CHOSEN_TRIGGER == TriggerType.SOFTWARE:
            # Get user input
            input('Press the Enter key to initiate software trigger.')

            # Execute software trigger
            if cam.TriggerSoftware.GetAccessMode() != PySpin.WO:
                print('Unable to execute trigger. Aborting...')
                return False

            cam.TriggerSoftware.Execute()

            # TODO: Blackfly and Flea3 GEV cameras need 2 second delay after software trigger

        elif CHOSEN_TRIGGER == TriggerType.HARDWARE:
            cam.BeginAcquisition()

            print('Acquiring images...')
            try:

                #  Retrieve next received image
                #
                #  *** NOTES ***
                #  Capturing an image houses images on the camera buffer. Trying
                #  to capture an image that does not exist will hang the camera.
                #
                #  *** LATER ***
                #  Once an image from the buffer is saved and/or no longer
                #  needed, the image must be released in order to keep the
                #  buffer from filling up.
                image_result = cam.GetNextImage(500)

                #  Ensure image completion
                #
                #  *** NOTES ***
                #  Images can easily be checked for completion. This should be
                #  done whenever a complete image is expected or required.
                #  Further, check image status for a little more insight into
                #  why an image is incomplete.
                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())

                else:

                    #  Print image information; height and width recorded in pixels
                    #
                    #  *** NOTES ***
                    #  Images have quite a bit of available metadata including
                    #  things such as CRC, image status, and offset values, to
                    #  name a few.
                    width = image_result.GetWidth()
                    height = image_result.GetHeight()
                    print('Grabbed Image %d, width = %d, height = %d' % (i, width, height))

                    #  Convert image to mono 8
                    #
                    #  *** NOTES ***
                    #  Images can be converted between pixel formats by using
                    #  the appropriate enumeration value. Unlike the original
                    #  image, the converted one does not need to be released as
                    #  it does not affect the camera buffer.
                    #
                    #  When converting images, color processing algorithm is an
                    #  optional parameter.
                    image_converted = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)

                    # Create a unique filename
                    
                    filename = 'Acquisition_'+Led[2]+'.bmp'

                    #  Save image
                    #
                    #  *** NOTES ***
                    #  The standard practice of the examples is to use device
                    #  serial numbers to keep images of one device from
                    #  overwriting those of another.
                    image_converted.Save(filename)
                    print(image_result)
                    #cv2.imwrite(filename, image_result)
                    print('Image saved at %s' % filename)

                    #  Release image
                    #
                    #  *** NOTES ***
                    #  Images retrieved directly from the camera (i.e. non-converted
                    #  images) need to be released in order to keep from filling the
                    #  buffer.
                    image_result.Release()
                    print('')

            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                return False
    
            #  End acquisition
            #
            #  *** NOTES ***
            #  Ending acquisition appropriately helps ensure that devices clean up
            #  properly and do not need to be power-cycled to maintain integrity.
            cam.EndAcquisition()
            
            
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result


def print_device_info(nodemap):
    """
    This function prints the device information of the camera from the transport
    layer; please see NodeMapInfo example for more in-depth comments on printing
    device information from the nodemap.

    :param nodemap: Transport layer device nodemap.
    :type nodemap: INodeMap
    :returns: True if successful, False otherwise.
    :rtype: bool
    """

    print('*** DEVICE INFORMATION ***\n')

    try:
        result = True
        node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

        if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                print('%s: %s' % (node_feature.GetName(),
                                  node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))

        else:
            print('Device control information not available.')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result


Error_check=0
Lista_Leds = ['M02N','M03N','M04N','M05N','M06N','M07N','M08N','M09N','M0AN','M0BN','M0CN','M0DN','M0EN','M0FN']

comunicacion = serial.Serial('COM16',57600)


# Retrieve singleton reference to system object
system = PySpin.System.GetInstance()

# Get current library version
version = system.GetLibraryVersion()
print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

# Retrieve list of cameras from the system
cam_list = system.GetCameras()

num_cameras = cam_list.GetSize()

print('Number of cameras detected: %d' % num_cameras)

# Finish if there are no cameras
if num_cameras == 0:
    # Clear camera list before releasing system
    
    cam_list.Clear()

    # Release system instance
    system.ReleaseInstance()

    print('Not enough cameras!')
    input('Done! Press Enter to exit...')

# Run example on each camera
for i, cam in enumerate(cam_list):
    
    cam.Init()
    
    # Retrieve GenICam nodemap
    nodemap = cam.GetNodeMap()
    
    
    #configure_trigger(cam,nodemap)
    print('Running example for camera %d...' % i)
    print('Camera %d example complete... \n' % i)
    for Led in Lista_Leds:
        
        try:
            bandera=0
            iteraciones=0
            
            while bandera==0 and iteraciones<3:
                
                comunicacion.write(Led.encode('utf-8'))
                time.sleep(1e-3)
            
                Check=''
                if comunicacion.inWaiting()==1:
                    print(2)
                    Check = comunicacion.read()
                    
                if Check == b'O':
                    print("Comando "+Led +" Aceptado")
                    bandera=1
                
                print(Check)
                iteraciones+=1
            
            if(bandera==0):
                warnings.warn('Error en Comunicación')
                warnings.simplefilter('No se recibe respuesta del puerto Serial', UserWarning)
            comunicacion.write('W'.encode('utf-8'))
            
            time.sleep(1e-3)
            
            Check=''
            if comunicacion.inWaiting()==1:
                Check = comunicacion.read()
                grab_next_image_by_trigger(cam,Led)
                #grab_next_image_by_trigger(cam,Led)
                
            if Check == b'O':
                print("Comando W Aceptado")
                
            
            time.sleep(1)
        
        except:
            print("Comunicación Fracasada")
            Error_check=1
            break
    

if Error_check==1:
    print("El programa finalizo con errores")
    
else:
    print("El programa finalizo correctamente")
    

comunicacion.close()
del cam


# Release reference to camera
# NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
# cleaned up when going out of scope.
# The usage of del is preferred to assigning the variable to None.
