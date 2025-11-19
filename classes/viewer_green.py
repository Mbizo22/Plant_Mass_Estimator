#!/usr/bin/env python3

import numpy as np
import cv2
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel
from pylibfreenect2 import CpuPacketPipeline

def main():
    # Enable Info level logging like Protonect
    logger = createConsoleLogger(LoggerLevel.Info)
    setGlobalLogger(logger)
    
    # Use CPU pipeline like Protonect
    pipeline = CpuPacketPipeline()
    fn = Freenect2(pipeline)
    
    num_devices = fn.enumerateDevices()
    if num_devices == 0:
        print("No Kinect device connected!")
        return
    
    serial = fn.getDeviceSerialNumber(1)
    print(f"Found Kinect device with serial: {serial}")
    
    # Create and set up device
    device = fn.openDevice(serial)
    
    # Check if device opened successfully
    if not device:
        print("Failed to open device!")
        return
    
    print(f"Device firmware: {device.getFirmwareVersion()}")
    
    # Create listener for RGB frames only (like Protonect -nodepth)
    listener = SyncMultiFrameListener(FrameType.Color)
    
    # Register listeners
    device.setColorFrameListener(listener)
    
    # Start the device
    print("Starting device...")
    try:
        device.startStreams(rgb=True, depth=False)  # Explicit stream control
        print("Device started successfully!")
    except Exception as e:
        print(f"Error starting device: {e}")
        try:
            # Try the simple start method
            started = device.start()
            if started:
                print("Device started with simple start()")
            else:
                print("Failed to start device with simple start()")
                device.close()
                return
        except Exception as e2:
            print(f"Both start methods failed: {e2}")
            device.close()
            return
    
    # NOTE: We need to call registration. Here we use the default parameters.
    registration = Registration(device.getIrCameraParams(),
                              device.getColorCameraParams())
    
    undistorted = Frame(512, 424, 4)
    registered = Frame(512, 424, 4)
    
    print("Starting RGB stream only...")
    print("Press 'q' to quit, 'ESC' to exit")
    
    try:
        frame_count = 0
        while True:
            frames = listener.waitForNewFrame()  # Remove timeout parameter
            
            # Get the color frame using integer constant (1 = Color)
            try:
                color = frames[1]  # FrameType.Color = 1
                if color:
                    # Convert frames to numpy arrays
                    # Color frame (BGRX format)
                    color_array = color.asarray()
                    color_bgr = cv2.cvtColor(color_array, cv2.COLOR_BGRA2BGR)
                    
                    # Resize for display
                    color_display = cv2.resize(color_bgr, (640, 480))
                    
                    # Display the frame
                    cv2.imshow('RGB Stream', color_display)
                    
                    frame_count += 1
                    if frame_count % 30 == 0:  # Print every 30 frames
                        print(f"Received {frame_count} RGB frames")
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        break
                else:
                    print("No RGB frame received")
                    
            except Exception as e:
                print(f"Error processing frame: {e}")
            
            listener.release(frames)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        device.stop()
        device.close()
        cv2.destroyAllWindows()
        print("Cleanup complete")

if __name__ == "__main__":
    main()
