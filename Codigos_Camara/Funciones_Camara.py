# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 22:33:42 2021

@author: Johan Cuervo
"""
import PySpin
import sys
import serial
import time
import warnings


class TriggerType:
    SOFTWARE = 1
    HARDWARE = 2


CHOSEN_TRIGGER = TriggerType.HARDWARE


def setear_config_cam(
    cam,
    nodemap,
    VGamma=1.25,
    VExposureTime=8000,
    VGain=0,
    VSharpness=1800,
    VBlackLevel=0.7,
):

    # Turn off auto exposure
    node_exposure_auto = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
    if not PySpin.IsAvailable(node_exposure_auto) or not PySpin.IsWritable(
        node_exposure_auto
    ):
        print("\nUnable to set Exposure Auto (enumeration retrieval). Aborting...\n")
        return False

    entry_exposure_auto_off = node_exposure_auto.GetEntryByName("Off")
    if not PySpin.IsAvailable(entry_exposure_auto_off) or not PySpin.IsReadable(
        entry_exposure_auto_off
    ):
        print("\nUnable to set Exposure Auto (entry retrieval). Aborting...\n")
        return False

    exposure_auto_off = entry_exposure_auto_off.GetValue()

    node_exposure_auto.SetIntValue(exposure_auto_off)

    # Set Exposure Time to less than 1/50th of a second (5000 us is used as an example)
    node_exposure_time = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
    if not PySpin.IsAvailable(node_exposure_time) or not PySpin.IsWritable(
        node_exposure_time
    ):
        print("\nUnable to set Exposure Time (float retrieval). Aborting...\n")
        return False

    node_exposure_time.SetValue(VExposureTime)

    # Turn off Gain
    node_gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode("GainAuto"))
    if not PySpin.IsAvailable(node_gain_auto) or not PySpin.IsWritable(node_gain_auto):
        print("\nUnable to set Gain Auto (enumeration retrieval). Aborting...\n")
        return False

    entry_gain_auto_off = node_gain_auto.GetEntryByName("Off")
    if not PySpin.IsAvailable(entry_gain_auto_off) or not PySpin.IsReadable(
        entry_gain_auto_off
    ):
        print("\nUnable to set Gain Auto (entry retrieval). Aborting...\n")
        return False

    gain_auto_off = entry_gain_auto_off.GetValue()

    node_gain_auto.SetIntValue(gain_auto_off)

    node_gain = PySpin.CFloatPtr(nodemap.GetNode("Gain"))
    if not PySpin.IsAvailable(node_gain) or not PySpin.IsWritable(node_gain):
        print("\nUnable to set Gain (float retrieval). Aborting...\n")
        return False

    node_gain.SetValue(VGain)

    # Turn off Sharpness

    node_Sharpness_auto = PySpin.CEnumerationPtr(nodemap.GetNode("SharpnessAuto"))
    if not PySpin.IsAvailable(node_Sharpness_auto) or not PySpin.IsWritable(
        node_Sharpness_auto
    ):
        print("\nUnable to set Sharpness Auto (enumeration retrieval). Aborting...\n")
        return False

    entry_Sharpness_auto_off = node_Sharpness_auto.GetEntryByName("Off")
    if not PySpin.IsAvailable(entry_Sharpness_auto_off) or not PySpin.IsReadable(
        entry_Sharpness_auto_off
    ):
        print("\nUnable to set Sharpness Auto (entry retrieval). Aborting...\n")
        return False

    Sharpness_auto_off = entry_Sharpness_auto_off.GetValue()

    node_Sharpness_auto.SetIntValue(Sharpness_auto_off)

    node_Sharpness = PySpin.CIntegerPtr(nodemap.GetNode("Sharpness"))
    if not PySpin.IsAvailable(node_Sharpness) or not PySpin.IsWritable(node_Sharpness):
        print("\nUnable to set Sharpness Time (Integer retrieval). Aborting...\n")
        return False

    node_Sharpness.SetValue(VSharpness)

    # Turn off BlackLevel

    node_BlackLevel = PySpin.CFloatPtr(nodemap.GetNode("BlackLevel"))
    if not PySpin.IsAvailable(node_BlackLevel) or not PySpin.IsWritable(
        node_BlackLevel
    ):
        print("\nUnable to set BlackLevel (float retrieval). Aborting...\n")
        return False

    node_BlackLevel.SetValue(VBlackLevel)

    # Turn off Gamma

    node_Gamma = PySpin.CFloatPtr(nodemap.GetNode("Gamma"))
    if not PySpin.IsAvailable(node_Gamma) or not PySpin.IsWritable(node_Gamma):
        print("\nUnable to set Gamma Time (Integer retrieval). Aborting...\n")
        return False

    node_Gamma.SetValue(VGamma)


def configure_crown(comunicacion, time_sleep):
    result = True
    intensidades_leds = [
        "J1090K",
        "J2090K",
        "J3090K",
        "J4090K",
        "J5090K",
        "J6090K",
        "J7090K",
        "J8090K",
        "J9090K",
        "JA090K",
        "JB090K",
        "JC090K",
        "JD080K",
        "JE010K",
        "JF010K",
    ]

    for Led in intensidades_leds:
        bandera = 0
        iteraciones = 0

        while bandera == 0 and iteraciones < 6:

            comunicacion.write(Led.encode("utf-8"))
            time.sleep(time_sleep)

            Check = ""
            if comunicacion.inWaiting() == 1:

                Check = comunicacion.read()

            if Check == b"O":
                bandera = 1

            iteraciones += 1

    if bandera == 0:
        print("Error al configurar el PWM de la corona")
        result = False

    return result


def configure_trigger(cam):
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

    print("*** CONFIGURING TRIGGER ***\n")

    print(
        "Note that if the application / user software triggers faster than frame time, the trigger may be dropped / skipped by the camera.\n"
    )
    print(
        "If several frames are needed per trigger, a more reliable alternative for such case, is to use the multi-frame mode.\n\n"
    )

    if CHOSEN_TRIGGER == TriggerType.SOFTWARE:
        print("Software trigger chosen...")
    elif CHOSEN_TRIGGER == TriggerType.HARDWARE:
        print("Hardware trigger chose...")

    try:
        result = True

        # Ensure trigger mode off
        # The trigger must be disabled in order to configure whether the source
        # is software or hardware.
        if cam.TriggerMode.GetAccessMode() != PySpin.RW:
            print("Unable to disable trigger mode (node retrieval). Aborting...")
            return False

        cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)

        print("Trigger mode disabled...")

        # Set TriggerSelector to FrameStart
        # For this example, the trigger selector should be set to frame start.
        # This is the default for most cameras.
        ## if cam.TriggerSelector.GetAccessMode() != PySpin.RW:
        ##     print('Unable to get trigger selector (node retrieval). Aborting...')
        ##     return False

        cam.TriggerSource.SetValue(PySpin.TriggerSelector_FrameStart)

        print("Trigger selector set to frame start...")

        # Select trigger source
        # The trigger source must be set to hardware or software while trigger
        # mode is off.
        if cam.TriggerSource.GetAccessMode() != PySpin.RW:
            print("Unable to get trigger source (node retrieval). Aborting...")
            return False

        if CHOSEN_TRIGGER == TriggerType.SOFTWARE:
            cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)
            print("Trigger source set to software...")
        elif CHOSEN_TRIGGER == TriggerType.HARDWARE:
            cam.TriggerSource.SetValue(PySpin.TriggerSource_Line2)  # line 0,1,2
            print("Trigger source set to hardware...")

        # Turn trigger mode on
        # Once the appropriate trigger source has been set, turn trigger mode
        # on in order to retrieve images using the trigger.
        cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
        print("Trigger mode turned back on...")

    except PySpin.SpinnakerException as ex:
        print("Error: %s" % ex)
        return False

    return result


def run_single_camera(cam):
    """
    This function acts as the body of the example; please see NodeMapInfo example
    for more in-depth comments on setting up cameras.

    :param cam: Camera to run on.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True

        # Retrieve TL device nodemap and print device information
        nodemap_tldevice = cam.GetTLDeviceNodeMap()

        # result &= print_device_info(nodemap_tldevice)

        # Initialize camera
        cam.Init()

        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()

        # Configure trigger
        if setear_config_cam(cam, nodemap) is False:
            return False

        if configure_trigger(cam) is False:
            return False

        if configure_buffer(cam, nodemap) is False:
            return False

        # Acquire images
        # result &= acquire_images(cam)

        # Reset trigger
        # result &= reset_trigger(cam)

        # Deinitialize camera
        cam.DeInit()

    except PySpin.SpinnakerException as ex:
        print("Error: %s" % ex)
        result = False

    return result


def configure_buffer(cam, nodemap):
    try:
        result = True

        node_acquisition_mode = PySpin.CEnumerationPtr(
            nodemap.GetNode("AcquisitionMode")
        )
        if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(
            node_acquisition_mode
        ):
            print(
                "Unable to set acquisition mode to continuous (node retrieval). Aborting..."
            )
            return False

        # Retrieve entry node from enumeration mode
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName(
            "Continuous"
        )
        if not PySpin.IsAvailable(
            node_acquisition_mode_continuous
        ) or not PySpin.IsReadable(node_acquisition_mode_continuous):
            print(
                "Unable to set acquisition mode to continuous (entry retrieval). Aborting..."
            )
            return False

        # Retrieve integer value from entry node
        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        print("Acquisition mode set to continuous...")

        # Retrieve Stream Parameters device nodemap
        s_node_map = cam.GetTLStreamNodeMap()

        # Retrieve Buffer Handling Mode Information
        handling_mode = PySpin.CEnumerationPtr(
            s_node_map.GetNode("StreamBufferHandlingMode")
        )
        if not PySpin.IsAvailable(handling_mode) or not PySpin.IsWritable(
            handling_mode
        ):
            print("Unable to set Buffer Handling mode (node retrieval). Aborting...\n")
            return False

        handling_mode_entry = PySpin.CEnumEntryPtr(handling_mode.GetCurrentEntry())
        if not PySpin.IsAvailable(handling_mode_entry) or not PySpin.IsReadable(
            handling_mode_entry
        ):
            print("Unable to set Buffer Handling mode (Entry retrieval). Aborting...\n")
            return False

        # Set stream buffer Count Mode to manual
        stream_buffer_count_mode = PySpin.CEnumerationPtr(
            s_node_map.GetNode("StreamBufferCountMode")
        )
        if not PySpin.IsAvailable(stream_buffer_count_mode) or not PySpin.IsWritable(
            stream_buffer_count_mode
        ):
            print("Unable to set Buffer Count Mode (node retrieval). Aborting...\n")
            return False

        stream_buffer_count_mode_manual = PySpin.CEnumEntryPtr(
            stream_buffer_count_mode.GetEntryByName("Manual")
        )
        if not PySpin.IsAvailable(
            stream_buffer_count_mode_manual
        ) or not PySpin.IsReadable(stream_buffer_count_mode_manual):
            print(
                "Unable to set Buffer Count Mode entry (Entry retrieval). Aborting...\n"
            )
            return False

        stream_buffer_count_mode.SetIntValue(stream_buffer_count_mode_manual.GetValue())
        print("Stream Buffer Count Mode set to manual...")

        # Retrieve and modify Stream Buffer Count
        buffer_count = PySpin.CIntegerPtr(s_node_map.GetNode("StreamBufferCountManual"))
        if not PySpin.IsAvailable(buffer_count) or not PySpin.IsWritable(buffer_count):
            print("Unable to set Buffer Count (Integer node retrieval). Aborting...\n")
            return False

        # Display Buffer Info
        print(
            "\nDefault Buffer Handling Mode: %s" % handling_mode_entry.GetDisplayName()
        )
        print("Default Buffer Count: %d" % buffer_count.GetValue())
        print("Maximum Buffer Count: %d" % buffer_count.GetMax())

        buffer_count.SetValue(1)

        print("Buffer count now set to: %d" % buffer_count.GetValue())

        handling_mode_entry = handling_mode.GetEntryByName("NewestOnly")
        handling_mode.SetIntValue(handling_mode_entry.GetValue())
        print(
            "\n\nBuffer Handling Mode has been set to %s"
            % handling_mode_entry.GetDisplayName()
        )

    except PySpin.SpinnakerException as ex:
        print("Error configurando el buffer: %s" % ex)
        result = False

    return result


def acquire_images(cam, nodemap, nodemap_tldevice):
    """
    This function cycles through the four different buffer handling modes.
    It saves three images for three of the buffer handling modes
    (NewestFirst, OldestFirst, and OldestFirstOverwrite).  For NewestOnly,
    it saves one image.

    :param cam: Camera instance to grab images from.
    :param nodemap: Device nodemap.
    :type cam: CameraPtr
    :type nodemap: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True
        print("\n*** IMAGE ACQUISITION ***\n")

        node_acquisition_mode = PySpin.CEnumerationPtr(
            nodemap.GetNode("AcquisitionMode")
        )
        if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(
            node_acquisition_mode
        ):
            print(
                "Unable to set acquisition mode to continuous (node retrieval). Aborting..."
            )
            return False

        # Retrieve entry node from enumeration mode
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName(
            "Continuous"
        )
        if not PySpin.IsAvailable(
            node_acquisition_mode_continuous
        ) or not PySpin.IsReadable(node_acquisition_mode_continuous):
            print(
                "Unable to set acquisition mode to continuous (entry retrieval). Aborting..."
            )
            return False

        # Retrieve integer value from entry node
        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        print("Acquisition mode set to continuous...")

        # Retrieve device serial number for filename
        device_serial_number = ""
        node_device_serial_number = PySpin.CStringPtr(
            nodemap_tldevice.GetNode("DeviceSerialNumber")
        )
        if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(
            node_device_serial_number
        ):
            device_serial_number = node_device_serial_number.GetValue()
            print("Device serial number retrieved as %s..." % device_serial_number)

        # Retrieve Stream Parameters device nodemap
        s_node_map = cam.GetTLStreamNodeMap()

        # Retrieve Buffer Handling Mode Information
        handling_mode = PySpin.CEnumerationPtr(
            s_node_map.GetNode("StreamBufferHandlingMode")
        )
        if not PySpin.IsAvailable(handling_mode) or not PySpin.IsWritable(
            handling_mode
        ):
            print("Unable to set Buffer Handling mode (node retrieval). Aborting...\n")
            return False

        handling_mode_entry = PySpin.CEnumEntryPtr(handling_mode.GetCurrentEntry())
        if not PySpin.IsAvailable(handling_mode_entry) or not PySpin.IsReadable(
            handling_mode_entry
        ):
            print("Unable to set Buffer Handling mode (Entry retrieval). Aborting...\n")
            return False

        # Set stream buffer Count Mode to manual
        stream_buffer_count_mode = PySpin.CEnumerationPtr(
            s_node_map.GetNode("StreamBufferCountMode")
        )
        if not PySpin.IsAvailable(stream_buffer_count_mode) or not PySpin.IsWritable(
            stream_buffer_count_mode
        ):
            print("Unable to set Buffer Count Mode (node retrieval). Aborting...\n")
            return False

        stream_buffer_count_mode_manual = PySpin.CEnumEntryPtr(
            stream_buffer_count_mode.GetEntryByName("Manual")
        )
        if not PySpin.IsAvailable(
            stream_buffer_count_mode_manual
        ) or not PySpin.IsReadable(stream_buffer_count_mode_manual):
            print(
                "Unable to set Buffer Count Mode entry (Entry retrieval). Aborting...\n"
            )
            return False

        stream_buffer_count_mode.SetIntValue(stream_buffer_count_mode_manual.GetValue())
        print("Stream Buffer Count Mode set to manual...")

        # Retrieve and modify Stream Buffer Count
        buffer_count = PySpin.CIntegerPtr(s_node_map.GetNode("StreamBufferCountManual"))
        if not PySpin.IsAvailable(buffer_count) or not PySpin.IsWritable(buffer_count):
            print("Unable to set Buffer Count (Integer node retrieval). Aborting...\n")
            return False

        # Display Buffer Info
        print(
            "\nDefault Buffer Handling Mode: %s" % handling_mode_entry.GetDisplayName()
        )
        print("Default Buffer Count: %d" % buffer_count.GetValue())
        print("Maximum Buffer Count: %d" % buffer_count.GetMax())

        buffer_count.SetValue(NUM_BUFFERS)

        print("Buffer count now set to: %d" % buffer_count.GetValue())
        print(
            "\nCamera will be triggered %d times in a row before %d images will be retrieved"
            % (NUM_TRIGGERS, (NUM_LOOPS - NUM_TRIGGERS))
        )

        for x in range(0, 4):
            if x == 0:
                handling_mode_entry = handling_mode.GetEntryByName("NewestFirst")
                handling_mode.SetIntValue(handling_mode_entry.GetValue())
                print(
                    "\n\nBuffer Handling Mode has been set to %s"
                    % handling_mode_entry.GetDisplayName()
                )
            elif x == 1:
                handling_mode_entry = handling_mode.GetEntryByName("NewestOnly")
                handling_mode.SetIntValue(handling_mode_entry.GetValue())
                print(
                    "\n\nBuffer Handling Mode has been set to %s"
                    % handling_mode_entry.GetDisplayName()
                )
            elif x == 2:
                handling_mode_entry = handling_mode.GetEntryByName("OldestFirst")
                handling_mode.SetIntValue(handling_mode_entry.GetValue())
                print(
                    "\n\nBuffer Handling Mode has been set to %s"
                    % handling_mode_entry.GetDisplayName()
                )
            elif x == 3:
                handling_mode_entry = handling_mode.GetEntryByName(
                    "OldestFirstOverwrite"
                )
                handling_mode.SetIntValue(handling_mode_entry.GetValue())
                print(
                    "\n\nBuffer Handling Mode has been set to %s"
                    % handling_mode_entry.GetDisplayName()
                )

            # Begin capturing images
            cam.BeginAcquisition()

            # Sleep for one second; only necessary when using non-BFS/ORX cameras on startup
            if x == 0:
                time.sleep(1)

            try:
                # Software Trigger the camera then  save images
                for loop_cnt in range(0, NUM_LOOPS):
                    if loop_cnt < NUM_TRIGGERS:
                        # Retrieve the next image from the trigger
                        result &= grab_next_image_by_trigger(nodemap)
                        print("\nCamera triggered. No image grabbed")
                    else:
                        print(
                            "\nNo trigger. Grabbing image %d"
                            % (loop_cnt - NUM_TRIGGERS)
                        )
                        result_image = cam.GetNextImage(500)

                        if result_image.IsIncomplete():
                            print(
                                "Image incomplete with image status %s ...\n"
                                % result_image.GetImageStatus()
                            )

                    if loop_cnt >= NUM_TRIGGERS:
                        # Retrieve Frame ID
                        print("Frame ID: %d" % result_image.GetFrameID())

                        # Create a unique filename
                        if device_serial_number:
                            filename = "%s-%s-%d.jpg" % (
                                handling_mode_entry.GetSymbolic(),
                                device_serial_number,
                                (loop_cnt - NUM_TRIGGERS),
                            )
                        else:
                            filename = "%s-%d.jpg" % (
                                handling_mode_entry.GetSymbolic(),
                                (loop_cnt - NUM_TRIGGERS),
                            )

                        # Save image
                        result_image.Save(filename)
                        print("Image saved at %s" % filename)

                        # Release image
                        result_image.Release()

                    # To control the framerate, have the application pause for 250ms.
                    time.sleep(0.25)

            except PySpin.SpinnakerException as ex:
                print("Error: %s" % ex)
                if handling_mode_entry.GetSymbolic() == "NewestOnly":
                    print(
                        "Error should occur when grabbing image 1 with handling mode set to NewestOnly"
                    )
                result = False

            # End acquisition
            cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print("Error: %s" % ex)
        result = False

    return result


def grab_next_image_by_trigger(cam, Led):
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
            input("Press the Enter key to initiate software trigger.")

            # Execute software trigger
            if cam.TriggerSoftware.GetAccessMode() != PySpin.WO:
                print("Unable to execute trigger. Aborting...")
                return False

            cam.TriggerSoftware.Execute()

            # TODO: Blackfly and Flea3 GEV cameras need 2 second delay after software trigger

        elif CHOSEN_TRIGGER == TriggerType.HARDWARE:

            print("Acquiring images...")
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
                    print(
                        "Image incomplete with image status %d ..."
                        % image_result.GetImageStatus()
                    )

                else:

                    #  Print image information; height and width recorded in pixels
                    #
                    #  *** NOTES ***
                    #  Images have quite a bit of available metadata including
                    #  things such as CRC, image status, and offset values, to
                    #  name a few.
                    width = image_result.GetWidth()
                    height = image_result.GetHeight()
                    print("Grabbed Image width = %d, height = %d" % (width, height))

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
                    image_converted = image_result.Convert(
                        PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR
                    )

                    # Create a unique filename

                    filename = "Acquisition_" + Led[2] + ".bmp"

                    #  Save image
                    #
                    #  *** NOTES ***
                    #  The standard practice of the examples is to use device
                    #  serial numbers to keep images of one device from
                    #  overwriting those of another.
                    image_converted.Save(filename)
                    print(image_result)
                    # cv2.imwrite(filename, image_result)
                    print("Image saved at %s" % filename)

                    #  Release image
                    #
                    #  *** NOTES ***
                    #  Images retrieved directly from the camera (i.e. non-converted
                    #  images) need to be released in order to keep from filling the
                    #  buffer.
                    image_result.Release()
                    print("")

            except PySpin.SpinnakerException as ex:
                print("Error imagen no guardada: %s" % ex)
                return False

            #  End acquisition
            #
            #  *** NOTES ***
            #  Ending acquisition appropriately helps ensure that devices clean up
            #  properly and do not need to be power-cycled to maintain integrity.

    except PySpin.SpinnakerException as ex:
        print("Error imagen no guardada: %s" % ex)
        return False

    return result


def main():
    """
    Example entry point; please see Enumeration example for more in-depth
    comments on preparing and cleaning up the system.

    :return: True if successful, False otherwise.
    :rtype: bool
    """
    result = True
    time_sleep = 0.1

    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Get current library version
    version = system.GetLibraryVersion()
    print(
        "Library version: %d.%d.%d.%d"
        % (version.major, version.minor, version.type, version.build)
    )

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()

    num_cameras = cam_list.GetSize()

    print("Number of cameras detected: %d" % num_cameras)

    # Finish if there are no cameras
    if num_cameras == 0:
        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        print("Not enough cameras!")
        input("Done! Press Enter to exit...")
        return False

    # Run example on each camera
    cam = cam_list[0]

    result &= run_single_camera(cam)

    ###############################################################################

    Error_check = 0
    Lista_Leds = [
        "M01N",
        "M02N",
        "M03N",
        "M04N",
        "M05N",
        "M06N",
        "M07N",
        "M08N",
        "M09N",
        "M0AN",
        "M0BN",
        "M0CN",
        "M0DN",
        "M0EN",
        "M0FN",
    ]

    comunicacion = serial.Serial("COM16", 57600)

    if configure_crown(comunicacion, time_sleep):
        result = True

    else:
        return False

    cam.BeginAcquisition()
    for Led in Lista_Leds:

        try:
            bandera = 0
            iteraciones = 0

            while bandera == 0 and iteraciones < 6:

                comunicacion.write(Led.encode("utf-8"))
                time.sleep(time_sleep)

                Check = ""
                if comunicacion.inWaiting() == 1:
                    print(2)
                    Check = comunicacion.read()

                if Check == b"O":
                    print("Comando " + Led + " Aceptado")
                    bandera = 1

                print(Check)
                iteraciones += 1

            if bandera == 0:
                warnings.warn("Error en Comunicación")
                warnings.simplefilter(
                    "No se recibe respuesta del puerto Serial", UserWarning
                )
            comunicacion.write("W".encode("utf-8"))

            time.sleep(time_sleep)

            Check = ""
            if comunicacion.inWaiting() == 1:
                print("imagen adquirida")
                Check = comunicacion.read()
                grab_next_image_by_trigger(cam, Led)
                # grab_next_image_by_trigger(cam,Led)

            if Check == b"O":
                print("Comando W Aceptado")

            time.sleep(time_sleep)

        except:
            print("Comunicación Fracasada")
            Error_check = 1
            break

    cam.EndAcquisition()

    # Release reference to camera
    # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
    # cleaned up when going out of scope.
    # The usage of del is preferred to assigning the variable to None.
    del cam

    # Clear camera list before releasing system
    cam_list.Clear()

    # Release system instance
    system.ReleaseInstance()

    input("Done! Press Enter to exit...")
    return result


if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
