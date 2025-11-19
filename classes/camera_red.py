#!/usr/bin/env python3
"""
Simple Camera Host for Kinect Capture
Captures RGB and Depth data from Kinect and saves to current directory
Allows selection of specific Kinect device by serial number
"""

import time
import numpy as np
import cv2
from datetime import datetime
import sys
import os
import argparse
from pathlib import Path

# Try to import Kinect libraries
FREENECT2_AVAILABLE = False
USING_PYLIBFREENECT2 = False

try:
    from pylibfreenect2 import Freenect2, SyncMultiFrameListener
    from pylibfreenect2 import FrameType, Registration, Frame, CpuPacketPipeline
    USING_PYLIBFREENECT2 = True
    FREENECT2_AVAILABLE = True
    print("Using pylibfreenect2 library")
except ImportError:
    try:
        from freenect2 import Device, FrameType
        FREENECT2_AVAILABLE = True
        USING_PYLIBFREENECT2 = False
        print("Using freenect2 library (device selection not supported)")
    except ImportError:
        FREENECT2_AVAILABLE = False
        USING_PYLIBFREENECT2 = False

class SimpleCameraHost:
    def __init__(self, target_serial="006158144547", filename="default", count=1):
        self.device = None
        self.listener = None
        self.registration = None
        self.undistorted = None
        self.registered = None
        self.kinect_working = False
        self.capture_count = 0
        self.target_serial = target_serial
        self.freenect2 = None  # Add this line
        
        # Store the filename and count from args
        self.filename = filename
        self.count = count
        
        # Use current directory for saving
        self.save_directory = os.path.join(os.getcwd(), "data_collection")
        os.makedirs(self.save_directory, exist_ok = True)
        print(f"📁 Save directory: {self.save_directory}")
    
    def list_kinect_devices(self):
        """List all available Kinect devices with their serial numbers"""
        if not FREENECT2_AVAILABLE:
            print("❌ Kinect libraries not available")
            return []
        
        devices = []
        
        if USING_PYLIBFREENECT2:
            freenect2 = Freenect2()
            num_devices = freenect2.enumerateDevices()
            
            print(f"🔍 Found {num_devices} Kinect device(s):")
            
            for i in range(num_devices):
                serial = freenect2.getDeviceSerialNumber(i)
                devices.append(serial)
                print(f"  Device {i}: Serial {serial}")
        else:
            # For freenect2 library, we'll need to open devices to get serials
            print("🔍 Scanning for Kinect devices...")
            try:
                device = Device()
                # Assuming single device for now with freenect2
                devices.append("unknown_serial")
                print(f"  Device 0: Serial unknown_serial (freenect2 library limitation)")
                device.close()
            except Exception as e:
                print(f"❌ Error scanning devices: {e}")
        
        return devices
    
    def find_device_by_serial(self, target_serial):
        """Find device index by serial number"""
        if not USING_PYLIBFREENECT2:
            print("Serial selection not fully supported with freenect2 library")
            return 0  # Return first device
        
        freenect2 = Freenect2()
        num_devices = freenect2.enumerateDevices()
        
        for i in range(num_devices):
            serial = freenect2.getDeviceSerialNumber(i)
            
            # Convert bytes to string if needed
            if isinstance(serial, bytes):
                serial_str = serial.decode('utf-8')
            else:
                serial_str = str(serial)
            
            print(f"Checking device {i}: {serial_str} against target: {target_serial}")
            
            if serial_str == target_serial:
                print(f"Found target device at index {i}: Serial {serial_str}")
                return i
        
        print(f"Device with serial {target_serial} not found")
        return None
    
    def init_kinect(self):
        """Initialize Kinect with specific serial number"""
        if not FREENECT2_AVAILABLE:
            print("❌ Kinect libraries not available")
            return False
        
        # List available devices
        available_devices = self.list_kinect_devices()
        if not available_devices:
            print("❌ No Kinect devices found")
            return False
        
        if not USING_PYLIBFREENECT2:
            # For freenect2, we can't select by serial easily
            print(f"⚠️  Using freenect2 library - opening first available device")
            print(f"   Target serial: {self.target_serial} (selection not supported)")
            self.device = Device()
            self.kinect_working = True
            return True
        else:
            # For pylibfreenect2, we can select by serial
            device_index = self.find_device_by_serial(self.target_serial)
            if device_index is None:
                print(f"❌ Cannot find device with serial {self.target_serial}")
                print(f"   Available devices: {available_devices}")
                return False
            
            self.freenect2 = Freenect2()  # Store as instance variable
            pipeline = CpuPacketPipeline()
            
            # Open specific device by serial (convert to bytes)
            target_serial_bytes = self.target_serial.encode('utf-8')
            self.device = self.freenect2.openDevice(target_serial_bytes, pipeline)
            
            if not self.device:
                print(f"❌ Failed to open Kinect device with serial {self.target_serial}")
                return False
            
            # Verify we got the right device
            actual_serial = self.device.getSerialNumber()
            if isinstance(actual_serial, bytes):
                actual_serial = actual_serial.decode('utf-8')
            print(f"✅ Opened device with serial: {actual_serial}")
            
            types = FrameType.Color | FrameType.Depth
            self.listener = SyncMultiFrameListener(types)
            
            self.device.setColorFrameListener(self.listener)
            self.device.setIrAndDepthFrameListener(self.listener)
            
            # Try to start device - ignore return value if logs show success
            print("🔄 Starting device...")
            start_result = self.device.start()
            print(f"Start method returned: {start_result}")
            
            # Continue even if start() returns False, as the logs show it actually starts
            if start_result is None or start_result == False:
                print("⚠️  Start returned False/None, but continuing based on device logs")
            
            # Give device time to fully initialize
            time.sleep(2)
            
            self.registration = Registration(
                self.device.getIrCameraParams(),
                self.device.getColorCameraParams()
            )
            
            self.undistorted = Frame(512, 424, 4)
            self.registered = Frame(512, 424, 4)
            
            self.kinect_working = True
            print("✅ Device initialization completed")
            return True
    
    def capture_frames(self):
        """Capture RGB and Depth frames"""
        if not self.kinect_working:
            print("❌ Kinect not initialized")
            return None, None
        
        if not USING_PYLIBFREENECT2:
            frames = {}
            with self.device.running():
                for frame_type, frame in self.device:
                    frames[frame_type] = frame
                    if FrameType.Color in frames and FrameType.Depth in frames:
                        break
            
            if FrameType.Color in frames and FrameType.Depth in frames:
                rgb_frame = frames[FrameType.Color]
                depth_frame = frames[FrameType.Depth]
                
                rgb_array = np.array(rgb_frame.to_array()).reshape((rgb_frame.height, rgb_frame.width, 4))[:, :, :3]
                depth_array = np.array(depth_frame.to_array()).reshape((depth_frame.height, depth_frame.width))
                
                return rgb_array, depth_array
        else:            
            # Use the correct FrameType values (usually integers)
            try:
                frames = self.listener.waitForNewFrame()
                rgb_frame   = frames[FrameType.Color]
                depth_frame = frames[FrameType.Depth]

                self.registration.apply(rgb_frame, depth_frame,
                                        self.undistorted, self.registered)

                rgb_array   = rgb_frame.asarray()[:, :, :3].copy()
                depth_array = depth_frame.asarray().copy()

                self.listener.release(frames)
                return rgb_array, depth_array
            except Exception as e:
                print(f"❌ Error accessing frames: {e}")
                return None, None
    
    def filter_depth_data(self, depth_data):
        """Clean up depth data"""
        filtered_depth = depth_data.copy()
        filtered_depth[filtered_depth < 500] = 0
        filtered_depth[filtered_depth > 6000] = 0
        filtered_depth = cv2.medianBlur(filtered_depth.astype(np.uint16), 3)
        return filtered_depth
    
    def save_capture(self, rgb_data, depth_data):
        """Save RGB and Depth data to current directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.capture_count += 1
        
        save_path = Path(self.save_directory)
        filtered_depth = self.filter_depth_data(depth_data)
        
        # Generate file paths with serial number in filename
        rgb_jpg = save_path / f"{self.filename}_rgb_plant_{self.count}.jpg"
        rgb_npy = save_path / f"{self.filename}_rgb_plant_{self.count}.npy"
        depth_orig_npy = save_path / f"{self.filename}_depth_plant_{self.count}.npy"
        depth_orig_jpg = save_path / f"{self.filename}_depth_plant_{self.count}.jpg"
        #depth_filt_npy = save_path / f"{self.filename}_filtered_plant_{self.count}.npy"
        #depth_filt_jpg = save_path / f"{self.filename}_filtered_plant_{self.count}.jpg"
        
        try:
            # Save RGB files
            cv2.imwrite(str(rgb_jpg), rgb_data)
            np.save(str(rgb_npy), rgb_data)
            
            # Save original depth files
            np.save(str(depth_orig_npy), depth_data)
            depth_display_orig = ((depth_data.astype(np.float32) / 8000.0) * 255).astype(np.uint8)
            depth_colored_orig = cv2.applyColorMap(depth_display_orig, cv2.COLORMAP_JET)
            cv2.imwrite(str(depth_orig_jpg), depth_colored_orig)
            
            """
            # Save filtered depth files
            np.save(str(depth_filt_npy), filtered_depth)
            depth_display_filt = ((filtered_depth.astype(np.float32) / 8000.0) * 255).astype(np.uint8)
            depth_colored_filt = cv2.applyColorMap(depth_display_filt, cv2.COLORMAP_JET)
            cv2.imwrite(str(depth_filt_jpg), depth_colored_filt)
            """
            
            print(f"✅ Saved capture {self.capture_count} to current directory:")
            print(f"  RGB: {rgb_jpg.name} / {rgb_npy.name}")
            print(f"  Depth Original: {depth_orig_jpg.name} / {depth_orig_npy.name}")
            #print(f"  Depth Filtered: {depth_filt_jpg.name} / {depth_filt_npy.name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to save files: {e}")
            return False
    
    def cleanup(self):
        """Cleanup all resources"""
        if self.device:
            if USING_PYLIBFREENECT2:
                self.device.stop()
            self.device.close()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Kinect Camera Host with filename and count parameters')
    parser.add_argument('filename', nargs = '?', default = 'default', help = 'Filename to include in saved files')
    parser.add_argument('count', nargs = '?', type = int, default = 1, help = 'Count number for file naming')
    parser.add_argument('--serial', default = "006158144547", help = 'Target Kinect serial number')
    
    return parser.parse_args()
    
def main():
    # Parse command line arguments
    args = parse_arguments()
    
    print(f"Initializing Camera Host for Kinect Serial: {args.serial}")
    print(f"Using filename: {args.filename}")
    print(f"For Plant Number: {args.count}")
    
    camera = SimpleCameraHost(target_serial = args.serial, filename = args.filename, count = args.count)
    
    print("🎥 Initializing Kinect...")
    if not camera.init_kinect():
        print("❌ Failed to initialize Kinect")
        camera.cleanup()
        return
    
    print("📸 Capturing frames...")
    rgb_data, depth_data = camera.capture_frames()
    
    if rgb_data is not None and depth_data is not None:
        if camera.save_capture(rgb_data, depth_data):
            print("✅ Capture completed successfully")
        else:
            print("❌ Failed to save capture files")
    else:
        print("❌ Capture failed - no data received from Kinect")
    
    camera.cleanup()

if __name__ == "__main__":
    main()
