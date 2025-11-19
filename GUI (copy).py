""" Integration file import """
from integration import Integration
import numpy as np
from classes.random_forest_class import RandomForestRegressor, DecisionTreeRegressor
import sys
import os
import socket
import time
import struct

# Minimal imports first
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
except ImportError as e:
    print("Error importing tkinter: {}".format(e))
    sys.exit(1)

import json
import threading
import subprocess
import tempfile
import glob
import serial.tools.list_ports

# Optional imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available")

class PlantAnalysisGUI:
    def __init__(self, root):
        self.root = root
        
        # CRITICAL: Override system theme to make text visible
        self.root.option_add('*foreground', 'black')
        self.root.option_add('*background', 'white')
    
        self.root.title("Plant Analysis System")
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        root.minsize(800, 600)
        self.root.configure(bg='#f0f0f0')
        
        # State variables
        self.selected_plant = tk.StringVar()
        self.is_processing = False
        self.process_thread = None
        
        # Set default directory to reconstruction_output
        default_dir = os.path.join(os.getcwd(), "reconstruction_output")
        self.base_directory = tk.StringVar(value=default_dir)
        self.current_plant_number = None
        self.plant_count = 1
        
        # Arduino connection state
        self.arduino_connected = False
        self.arduino_port = None
        self.arduino_check_thread = None
        self.checking_arduino = True
        
        # Host connection state
        self.host_connected = False
        self.host_socket = None
        self.host_ip = "192.168.200.1"
        self.host_port = 8888
        self.host_check_thread = None
        self.checking_host = True
        self.waiting_for_host_response = False
        self.host_response_received = None
        
        # File transfer state
        self.receiving_files = False
        self.num_files_to_receive = 0
        self.files_received = 0
        
        # Main container
        self.setup_ui()
        
        # Load plants from default directory on launch
        self.load_available_plants()
        
        # Start Arduino connection monitoring
        self.start_arduino_monitoring()
        
        # Start Host connection monitoring
        self.start_host_monitoring()
        
    def setup_ui(self):
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50')
        title_frame.pack(fill='x', padx=0, pady=0)
        
        title_label = tk.Label(
            title_frame, 
            text="Plant Analysis System", 
            font=('Arial', 20, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(side='left', pady=15, padx=20)
        
        # Connection status indicators frame
        status_indicators = tk.Frame(title_frame, bg='#2c3e50')
        status_indicators.pack(side='right', pady=15, padx=20)
        
        # Arduino status indicator
        arduino_frame = tk.Frame(status_indicators, bg='#2c3e50')
        arduino_frame.pack(side='left', padx=10)
        
        tk.Label(arduino_frame, text="Arduino:", font=('Arial', 10, 'bold'),
                bg='#2c3e50', fg='white').pack(side='left', padx=5)
        
        self.arduino_led = tk.Canvas(arduino_frame, width=20, height=20, 
                                     bg='#2c3e50', highlightthickness=0)
        self.arduino_led.pack(side='left', padx=5)
        self.arduino_indicator = self.arduino_led.create_oval(2, 2, 18, 18, 
                                                              fill='red', outline='darkred')
        
        self.arduino_status_label = tk.Label(arduino_frame, text="Disconnected",
                                             font=('Arial', 9), bg='#2c3e50', 
                                             fg='#e74c3c')
        self.arduino_status_label.pack(side='left', padx=5)
        
        # Host connection status indicator
        host_frame = tk.Frame(status_indicators, bg='#2c3e50')
        host_frame.pack(side='left', padx=10)
        
        tk.Label(host_frame, text="Host:", font=('Arial', 10, 'bold'),
                bg='#2c3e50', fg='white').pack(side='left', padx=5)
        
        self.host_led = tk.Canvas(host_frame, width=20, height=20, 
                                  bg='#2c3e50', highlightthickness=0)
        self.host_led.pack(side='left', padx=5)
        self.host_indicator = self.host_led.create_oval(2, 2, 18, 18, 
                                                        fill='red', outline='darkred')
        
        self.host_status_label = tk.Label(host_frame, text="Disconnected",
                                          font=('Arial', 9), bg='#2c3e50', 
                                          fg='#e74c3c')
        self.host_status_label.pack(side='left', padx=5)
        
        # Main content area
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Create PanedWindow for resizable panels
        paned = tk.PanedWindow(main_container, orient=tk.HORIZONTAL, 
                               sashwidth=5, bg='#f0f0f0', relief='flat')
        paned.pack(fill='both', expand=True)
        
        # Left panel - Controls
        left_panel = tk.Frame(paned, bg='white', relief='raised', borderwidth=1)
        paned.add(left_panel, minsize=300)
        
        # Right panel - Results
        right_panel = tk.Frame(paned, bg='white', relief='raised', borderwidth=1)
        paned.add(right_panel, minsize=400)
        
        # Setup panels
        self.setup_control_panel(left_panel)
        self.setup_results_panel(right_panel)
    
    # ==================================================================================
    # Host Connection Monitoring
    # ==================================================================================
    
    def start_host_monitoring(self):
        """Start the host connection monitoring thread"""
        self.host_check_thread = threading.Thread(target=self.monitor_host_connection, daemon=True)
        self.host_check_thread.start()
        self.log_status("Started host connection monitoring ({}:{})".format(self.host_ip, self.host_port))
    
    def monitor_host_connection(self):
        """Continuously monitor and attempt host connection"""
        while self.checking_host:
            if not self.host_connected:
                self.attempt_host_connection()
                time.sleep(3)
            else:
                time.sleep(5)
    
    def attempt_host_connection(self):
        """Attempt to connect to the host"""
        try:
            temp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            temp_socket.settimeout(2)
            temp_socket.connect((self.host_ip, self.host_port))
            
            self.host_socket = temp_socket
            self.host_socket.settimeout(None)
            self.host_connected = True
            
            self.update_host_status(True)
            self.log_status("Connected to host at {}:{}".format(self.host_ip, self.host_port))
            
            listen_thread = threading.Thread(target=self.listen_to_host, daemon=True)
            listen_thread.start()
            
        except (socket.timeout, socket.error, ConnectionRefusedError):
            if self.host_socket:
                try:
                    self.host_socket.close()
                except:
                    pass
            self.host_socket = None
            self.host_connected = False
    
    def listen_to_host(self):
        """Listen for messages from the host"""
        while self.host_connected and self.checking_host:
            try:
                if self.host_socket:
                    # Check if we are in file receive mode or not
                    if self.receiving_files and self.files_received < self.num_files_to_receive:
                        continue
                        
                    self.host_socket.settimeout(1.0)
                    try:
                        data = self.host_socket.recv(1024).decode('utf-8')
                        if data:
                            self.log_status("Received from host: {}".format(data))
                            
                            # Handle file transfer protocol
                            # Handle file transfer protocol
                            if data == "START_TRANSFER":
                                self.handle_host_message(data)
                            elif data.startswith("NUM_FILES:"):
                                self.handle_host_message(data)
                                # Start receiving files in separate thread
                                receive_thread = threading.Thread(target=self.receive_all_files, daemon=True)
                                receive_thread.start()
                            elif data == "END_TRANSFER":
                                self.handle_host_message(data)
                            else:
                                self.handle_host_message(data)
                        else:
                            self.log_status("Host closed connection")
                            self.host_connected = False
                            break
                    except socket.timeout:
                        continue
                    except UnicodeDecodeError:
                        continue
            except socket.error as e:
                self.log_status("Host connection error: {}".format(str(e)))
                self.host_connected = False
                break
            except Exception as e:
                self.log_status("Error in host listener: {}".format(str(e)))
                self.host_connected = False
                break
        
        if self.host_socket:
            try:
                self.host_socket.close()
            except:
                pass
        self.host_socket = None
        self.update_host_status(False)
        self.log_status("Host listener thread ended")
    
    def handle_host_message(self, message):
        """Handle messages received from the host"""
        if message == "Complete":
            self.log_status("Host analysis complete")
            self.host_response_received = "Complete"
            self.waiting_for_host_response = False
        elif message == "Failed":
            self.log_status("Host analysis failed")
            self.host_response_received = "Failed"
            self.waiting_for_host_response = False
        elif message == "START_TRANSFER":
            self.log_status("Starting file transfer from host...")
            self.receiving_files = True
            self.files_received = 0
        elif message.startswith("NUM_FILES:"):
            self.num_files_to_receive = int(message.split(":")[1])
            self.log_status(f"Expecting {self.num_files_to_receive} files from host")
        elif message == "END_TRANSFER":
            self.log_status(f"File transfer complete: {self.files_received}/{self.num_files_to_receive} files received")
            self.receiving_files = False
            self.root.after(1000, self.load_available_plants)  # Refresh plant list
        elif message == "start_capture":
            self.log_status("Host requested capture start")
        elif message == "status_request":
            self.send_message_to_host("ready")
        elif message == "Plant":
            self.send_message_to_host(f"{self.plant_count}")
        else:
            if not self.receiving_files:
                self.log_status("Received unknown message from host: {}".format(message))
    
    def send_message_to_host(self, message):
        """Send a message to the host"""
        if self.host_connected and self.host_socket:
            try:
                self.host_socket.send(message.encode('utf-8'))
                self.log_status("Sent to host: {}".format(message))
                return True
            except socket.error as e:
                self.log_status("Failed to send message to host: {}".format(str(e)))
                self.host_connected = False
                return False
        else:
            self.log_status("Cannot send message - host not connected")
            return False
    
    def update_host_status(self, connected):
        """Update the host connection status indicator in UI"""
        def update():
            if connected:
                self.host_led.itemconfig(self.host_indicator, fill='#27ae60', outline='#229954')
                self.host_status_label.config(text="Connected", fg='#27ae60')
            else:
                self.host_led.itemconfig(self.host_indicator, fill='red', outline='darkred')
                self.host_status_label.config(text="Disconnected", fg='#e74c3c')
        
        self.root.after(0, update)
    
    # ==================================================================================
    # File Transfer Functions
    # ==================================================================================
    
    def receive_file(self):
        """
        Receive a single file from the host
        Protocol: [FILENAME_LENGTH][FILENAME][FILE_SIZE][FILE_DATA]
        """
        try:
            # Receive filename length (4 bytes)
            filename_length_data = self.recvall(4)
            if not filename_length_data:
                return False
            filename_length = struct.unpack('!I', filename_length_data)[0]
            
            # Receive filename
            filename_data = self.recvall(filename_length)
            if not filename_data:
                return False
            filename = filename_data.decode('utf-8')
            
            # Receive file size (8 bytes)
            filesize_data = self.recvall(8)
            if not filesize_data:
                return False
            filesize = struct.unpack('!Q', filesize_data)[0]
            
            self.log_status(f"Receiving: {filename} ({filesize} bytes)")
            
            # Create output directory
            output_dir = os.path.join(os.getcwd(), "reconstruction_output")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.log_status(f"Created directory: {output_dir}")
            
            # Receive file data
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'wb') as f:
                received = 0
                last_log = 0
                while received < filesize:
                    chunk_size = min(4096, filesize - received)
                    chunk = self.recvall(chunk_size)
                    if not chunk:
                        self.log_status(f"Error: Connection lost during file transfer")
                        return False
                    f.write(chunk)
                    received += len(chunk)
                    
                    # Progress (log every 10%)
                    progress = (received / filesize) * 100
                    if progress - last_log >= 10:
                        self.log_status(f"  Progress: {progress:.0f}%")
                        last_log = progress
            
            self.log_status(f"✓ Received: {filename}")
            self.files_received += 1
            return True
            
        except Exception as e:
            self.log_status(f"Error receiving file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def recvall(self, n):
        """Helper function to receive exactly n bytes from socket"""
        data = bytearray()
        while len(data) < n:
            packet = self.host_socket.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)
    
    def receive_all_files(self):
        """Receive all files from the host"""
        self.log_status(f"Receiving {self.num_files_to_receive} files...")
        
        for i in range(self.num_files_to_receive):
            self.log_status(f"File {i+1}/{self.num_files_to_receive}")
            if not self.receive_file():
                self.log_status(f"Failed to receive file {i+1}/{self.num_files_to_receive}")
                break
        
        self.log_status(f"File reception complete: {self.files_received}/{self.num_files_to_receive} files")
    
    # ==================================================================================
    # Arduino Connection Monitoring
    # ==================================================================================
    
    def start_arduino_monitoring(self):
        """Start the Arduino connection monitoring thread"""
        self.arduino_check_thread = threading.Thread(target=self.monitor_arduino_connection, daemon=True)
        self.arduino_check_thread.start()
        self.log_status("Started Arduino connection monitoring")
    
    def monitor_arduino_connection(self):
        """Continuously monitor Arduino connection"""
        while self.checking_arduino:
            ports = list(serial.tools.list_ports.comports())
            arduino_found = False
            
            for port in ports:
                if 'Arduino' in port.description or 'ACM' in port.device or 'USB' in port.device:
                    if not self.arduino_connected or self.arduino_port != port.device:
                        self.arduino_port = port.device
                        self.arduino_connected = True
                        arduino_found = True
                        self.update_arduino_status(True)
                        self.log_status("Arduino connected on {}".format(port.device))
                    else:
                        arduino_found = True
                    break
            
            if not arduino_found and self.arduino_connected:
                self.arduino_connected = False
                self.arduino_port = None
                self.update_arduino_status(False)
                self.log_status("Arduino disconnected")
            
            time.sleep(2)
    
    def update_arduino_status(self, connected):
        """Update the Arduino connection status indicator"""
        def update():
            if connected:
                self.arduino_led.itemconfig(self.arduino_indicator, fill='#27ae60', outline='#229954')
                self.arduino_status_label.config(text="Connected", fg='#27ae60')
            else:
                self.arduino_led.itemconfig(self.arduino_indicator, fill='red', outline='darkred')
                self.arduino_status_label.config(text="Disconnected", fg='#e74c3c')
        
        self.root.after(0, update)
    
    # ==================================================================================
    # UI Setup Methods
    # ==================================================================================
    
    def create_scrollable_frame(self, parent):
        """Create a scrollable frame"""
        canvas = tk.Canvas(parent, bg='white', highlightthickness=0)
        scrollbar = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        return scrollable_frame
    
    def setup_control_panel(self, parent):
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        
        scrollable = self.create_scrollable_frame(parent)
        
        # Directory selection
        dir_frame = tk.LabelFrame(scrollable, text="Data Directory", font=('Arial', 10, 'bold'),
                                  bg='white', fg='#2c3e50', padx=10, pady=10)
        dir_frame.pack(fill='x', padx=10, pady=10)
        
        self.dir_entry = tk.Entry(dir_frame, textvariable=self.base_directory, 
                                  fg='#2c3e50', bg='white')
        self.dir_entry.pack(side='left', fill='x', expand=True, padx=5)
        
        browse_btn = tk.Button(dir_frame, text="Browse", command=self.browse_directory,
                               bg='#3498db', fg='white', padx=10)
        browse_btn.pack(side='left')
        
        # Plant selection
        selection_frame = tk.LabelFrame(scrollable, text="Plant Selection", 
                                       font=('Arial', 10, 'bold'),
                                       bg='white', fg='#2c3e50', padx=10, pady=10)
        selection_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(selection_frame, text="Select Plant:", bg='white', 
                fg='#2c3e50', font=('Arial', 10)).pack(anchor='w', pady=5)
        
        self.plant_dropdown = ttk.Combobox(
            selection_frame, 
            textvariable=self.selected_plant,
            state='readonly',
            font=('Arial', 10)
        )
        self.plant_dropdown.pack(fill='x', pady=5)
        self.plant_dropdown.bind('<<ComboboxSelected>>', self.on_plant_selected)
        
        refresh_btn = tk.Button(selection_frame, text="Refresh Plant List",
                               command=self.load_available_plants,
                               bg='#95a5a6', fg='white', padx=10, pady=5)
        refresh_btn.pack(fill='x', pady=5)
        
        # Process Control
        control_frame = tk.LabelFrame(scrollable, text="Process Control",
                                     font=('Arial', 10, 'bold'),
                                     bg='white', fg='#2c3e50', padx=10, pady=10)
        control_frame.pack(fill='x', padx=10, pady=10)
        
        self.start_btn = tk.Button(
            control_frame, 
            text="START ANALYSIS",
            command=self.start_processing,
            bg='#27ae60',
            fg='white',
            font=('Arial', 11, 'bold'),
            padx=20,
            pady=10
        )
        self.start_btn.pack(fill='x', pady=5)
        
        self.pause_btn = tk.Button(
            control_frame,
            text="PAUSE",
            command=self.pause_processing,
            bg='#f39c12',
            fg='white',
            font=('Arial', 11, 'bold'),
            padx=20,
            pady=10,
            state='disabled'
        )
        self.pause_btn.pack(fill='x', pady=5)
        
        self.cancel_btn = tk.Button(
            control_frame,
            text="CANCEL",
            command=self.cancel_processing,
            bg='#e74c3c',
            fg='white',
            font=('Arial', 11, 'bold'),
            padx=20,
            pady=10,
            state='disabled'
        )
        self.cancel_btn.pack(fill='x', pady=5)
        
        # Status
        status_frame = tk.LabelFrame(scrollable, text="Status", font=('Arial', 10, 'bold'),
                                    bg='white', fg='#2c3e50', padx=10, pady=10)
        status_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.status_text = tk.Text(status_frame, height=8,
                                   font=('Courier', 9), bg='#ecf0f1',
                                   fg='#2c3e50', relief='sunken', borderwidth=2)
        self.status_text.pack(fill='both', expand=True)
        
        scrollbar = tk.Scrollbar(status_frame, command=self.status_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.status_text.config(yscrollcommand=scrollbar.set)
        
        self.log_status("System initialized. Ready.")
    
    def setup_results_panel(self, parent):
        """Setup the results display panel"""
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        
        scrollable = self.create_scrollable_frame(parent)
        
        # Results header
        header = tk.Label(scrollable, text="Analysis Results", 
                         font=('Arial', 16, 'bold'),
                         bg='white', fg='#2c3e50')
        header.pack(pady=20)
        
        # Placeholder for results
        self.results_frame = tk.Frame(scrollable, bg='white')
        self.results_frame.pack(fill='both', expand=True, padx=20)
        
        no_results_label = tk.Label(self.results_frame, 
                                    text="No plant selected\nSelect a plant to view results",
                                    font=('Arial', 12),
                                    bg='white', fg='#7f8c8d')
        no_results_label.pack(pady=50)
        
        self.detail_fields = {}
    
    # ==================================================================================
    # Application Methods
    # ==================================================================================
    
    def browse_directory(self):
        """Browse for directory"""
        directory = filedialog.askdirectory(initialdir=self.base_directory.get())
        if directory:
            self.base_directory.set(directory)
            self.load_available_plants()
            self.log_status("Directory changed to: {}".format(directory))
    
    def load_available_plants(self):
        """Load available plant data files"""
        base_dir = self.base_directory.get()
        if not os.path.exists(base_dir):
            self.log_status("Directory does not exist: {}".format(base_dir))
            return
        
        # Look for plant files
        plant_files = glob.glob(os.path.join(base_dir, "plant_*")) + \
                      glob.glob(os.path.join(base_dir, "*_plant_*"))
        plant_numbers = []
        
        for file in plant_files:
            try:
                # Extract plant number from filename
                parts = os.path.basename(file).split('_')
                for i, part in enumerate(parts):
                    if part == 'plant' and i + 1 < len(parts):
                        num_part = parts[i + 1].split('.')[0]
                        if num_part.isdigit():
                            plant_numbers.append(int(num_part))
                            break
            except:
                continue
        
        plant_numbers = sorted(set(plant_numbers))
        plant_options = ["Plant {}".format(n) for n in plant_numbers]
        
        self.plant_dropdown['values'] = plant_options
        if plant_options:
            self.plant_dropdown.current(0)
            self.log_status("Loaded {} plants".format(len(plant_options)))
        else:
            self.log_status("No plant data found in directory")
    
    def on_plant_selected(self, event=None):
        """Handle plant selection"""
        selected = self.selected_plant.get()
        if selected:
            plant_num = selected.split()[-1]
            self.current_plant_number = int(plant_num)
            self.log_status("Selected: {}".format(selected))
    
    def start_processing(self):
        """Start the analysis process"""
        if not self.arduino_connected:
            response = messagebox.askyesno("Arduino Not Connected",
                "Arduino is not connected. Continue anyway?")
            if not response:
                return
        
        if not self.host_connected:
            response = messagebox.askyesno("Host Not Connected",
                "Host is not connected. Continue anyway?")
            if not response:
                return
        
        self.is_processing = True
        self.start_btn.config(state='disabled')
        self.pause_btn.config(state='normal')
        self.cancel_btn.config(state='normal')
        
        self.log_status("Starting new plant analysis...")
        
        self.process_thread = threading.Thread(
            target=self.run_analysis,
            daemon=True
        )
        self.process_thread.start()
    
    def run_analysis(self):
        """Run the analysis pipeline"""
        try:
            # Movement control will be added soon
            
            # Send "Start" message to host
            self.log_status("Sending 'Start' message to host...")
            if not self.send_message_to_host("Start"):
                self.log_status("Failed to send Start message - host not connected")
                self.root.after(0, lambda: messagebox.showerror("Error", "Cannot communicate with host"))
                self.root.after(0, lambda: self.start_btn.config(state='normal'))
                self.root.after(0, lambda: self.pause_btn.config(state='disabled'))
                self.root.after(0, lambda: self.cancel_btn.config(state='disabled'))
                return
            
            # Wait for response from host (either "Complete" or "Failed")
            self.log_status("Waiting for host response...")
            self.waiting_for_host_response = True
            self.host_response_received = None
            
            # Wait up to 180 seconds for response (processing can take time)
            timeout = 240
            elapsed = 0
            while self.waiting_for_host_response and elapsed < timeout:
                time.sleep(0.5)
                elapsed += 0.5
            
            # Check if we got a response
            if not self.waiting_for_host_response:
                # Response received
                if self.host_response_received == "Complete":
                    self.log_status("Host reported: Complete")
                    self.plant_count += 1
                    self.log_status("Analysis complete!")
                    self.root.after(0, lambda: messagebox.showinfo("Success", "Analysis completed successfully"))
                elif self.host_response_received == "Failed":
                    self.log_status("Host reported: Failed")
                    self.root.after(0, lambda: messagebox.showerror("Error", "Host reported analysis failed"))
                else:
                    self.log_status("Received unexpected response: {}".format(self.host_response_received))
            #else:
            #    # Timeout
            #    self.log_status("Timeout waiting for host response")
            #    self.root.after(0, lambda: messagebox.showerror("Error", "Timeout waiting for host response"))
            
            # Re-enable buttons
            self.root.after(0, lambda: self.start_btn.config(state='normal'))
            self.root.after(0, lambda: self.pause_btn.config(state='disabled'))
            self.root.after(0, lambda: self.cancel_btn.config(state='disabled'))
            
        except Exception as e:
            self.log_status("Analysis failed: {}".format(str(e)))
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, lambda: self.start_btn.config(state='normal'))
            self.root.after(0, lambda: self.pause_btn.config(state='disabled'))
            self.root.after(0, lambda: self.cancel_btn.config(state='disabled'))
    
    def pause_processing(self):
        """Pause the processing"""
        self.log_status("Pause requested")
        pass
    
    def cancel_processing(self):
        """Cancel the processing"""
        self.is_processing = False
        self.log_status("Processing cancelled")
        self.start_btn.config(state='normal')
        self.pause_btn.config(state='disabled')
        self.cancel_btn.config(state='disabled')
    
    def log_status(self, message):
        """Log a status message"""
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.status_text.insert('end', "[{}] {}\n".format(timestamp, message))
            self.status_text.see('end')
        except:
            pass
    
    def cleanup(self):
        """Cleanup resources before closing"""
        self.checking_arduino = False
        self.checking_host = False
        
        if self.arduino_check_thread:
            self.arduino_check_thread.join(timeout=1)
        
        if self.host_check_thread:
            self.host_check_thread.join(timeout=1)
        
        if self.host_socket:
            try:
                self.host_socket.close()
            except:
                pass

def main():
    print("Starting Plant Analysis GUI...")
    try:
        root = tk.Tk()
        print("Tkinter initialized")
        app = PlantAnalysisGUI(root)
        print("GUI created successfully")
        
        def on_closing():
            app.cleanup()
            root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()
    except Exception as e:
        print("Fatal error: {}".format(str(e)))
        import traceback
        traceback.print_exc()
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
