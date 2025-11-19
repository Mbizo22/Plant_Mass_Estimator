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
import serial
import serial.tools.list_ports
import shutil

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
        self.arduino_serial = None
        self.arduino_check_thread = None
        self.checking_arduino = True
        
        # Motor control state - track button press states
        self.motor_states = {
            'left_forward': False,
            'left_reverse': False,
            'both_forward': False,
            'both_reverse': False,
            'right_forward': False,
            'right_reverse': False
        }
        
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
        
        # Progress tracking
        self.progress_var = tk.DoubleVar()
        
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
        self.setup_details_panel(right_panel)
    
    def setup_control_panel(self, parent):
        """Setup the control panel with motor joystick controls"""
        # Create scrollable frame
        canvas = tk.Canvas(parent, bg='white', highlightthickness=0)
        scrollbar = tk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # ==================== MOTOR CONTROL SECTION ====================
        motor_section = tk.LabelFrame(
            scrollable_frame,
            text="Motor Control - Joystick System",
            font=('Arial', 12, 'bold'),
            bg='white',
            fg='#2c3e50',
            padx=15,
            pady=15
        )
        motor_section.pack(fill='x', padx=10, pady=10)
        
        # Instructions
        instructions = tk.Label(
            motor_section,
            text="Press and hold buttons to move motors\nRelease to stop",
            font=('Arial', 9, 'italic'),
            bg='white',
            fg='#7f8c8d'
        )
        instructions.pack(pady=(0, 10))
        
        # Create container for three joystick arrays
        joystick_container = tk.Frame(motor_section, bg='white')
        joystick_container.pack(fill='x', pady=5)
        
        # Configure grid columns to be equal width
        joystick_container.grid_columnconfigure(0, weight=1, uniform='group1')
        joystick_container.grid_columnconfigure(1, weight=1, uniform='group1')
        joystick_container.grid_columnconfigure(2, weight=1, uniform='group1')
        
        # LEFT MOTOR CONTROL ARRAY
        left_frame = tk.LabelFrame(
            joystick_container,
            text="Left Motor",
            font=('Arial', 10, 'bold'),
            bg='white',
            fg='#e74c3c',
            padx=10,
            pady=10
        )
        left_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        
        # Left Forward Button
        self.left_fwd_btn = tk.Button(
            left_frame,
            text="▲\nFORWARD",
            font=('Arial', 10, 'bold'),
            bg='#27ae60',
            fg='white',
            activebackground='#229954',
            relief='raised',
            bd=3,
            height=3
        )
        self.left_fwd_btn.pack(fill='x', pady=3)
        self.left_fwd_btn.bind('<ButtonPress-1>', lambda e: self.on_motor_press('left_forward'))
        self.left_fwd_btn.bind('<ButtonRelease-1>', lambda e: self.on_motor_release('left_forward'))
        
        # Left Stop Button
        left_stop_btn = tk.Button(
            left_frame,
            text="■\nSTOP",
            font=('Arial', 10, 'bold'),
            bg='#e74c3c',
            fg='white',
            activebackground='#c0392b',
            relief='raised',
            bd=3,
            height=2,
            command=lambda: self.send_arduino_command('s')
        )
        left_stop_btn.pack(fill='x', pady=3)
        
        # Left Reverse Button
        self.left_rev_btn = tk.Button(
            left_frame,
            text="▼\nREVERSE",
            font=('Arial', 10, 'bold'),
            bg='#f39c12',
            fg='white',
            activebackground='#d68910',
            relief='raised',
            bd=3,
            height=3
        )
        self.left_rev_btn.pack(fill='x', pady=3)
        self.left_rev_btn.bind('<ButtonPress-1>', lambda e: self.on_motor_press('left_reverse'))
        self.left_rev_btn.bind('<ButtonRelease-1>', lambda e: self.on_motor_release('left_reverse'))
        
        # BOTH MOTORS CONTROL ARRAY (CENTER)
        both_frame = tk.LabelFrame(
            joystick_container,
            text="Both Motors",
            font=('Arial', 10, 'bold'),
            bg='white',
            fg='#3498db',
            padx=10,
            pady=10
        )
        both_frame.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')
        
        # Both Forward Button
        self.both_fwd_btn = tk.Button(
            both_frame,
            text="▲▲\nFORWARD",
            font=('Arial', 11, 'bold'),
            bg='#27ae60',
            fg='white',
            activebackground='#229954',
            relief='raised',
            bd=3,
            height=3
        )
        self.both_fwd_btn.pack(fill='x', pady=3)
        self.both_fwd_btn.bind('<ButtonPress-1>', lambda e: self.on_motor_press('both_forward'))
        self.both_fwd_btn.bind('<ButtonRelease-1>', lambda e: self.on_motor_release('both_forward'))
        
        # Both Stop Button
        both_stop_btn = tk.Button(
            both_frame,
            text="■■\nSTOP ALL",
            font=('Arial', 11, 'bold'),
            bg='#e74c3c',
            fg='white',
            activebackground='#c0392b',
            relief='raised',
            bd=3,
            height=2,
            command=lambda: self.send_arduino_command('s')
        )
        both_stop_btn.pack(fill='x', pady=3)
        
        # Both Reverse Button
        self.both_rev_btn = tk.Button(
            both_frame,
            text="▼▼\nREVERSE",
            font=('Arial', 11, 'bold'),
            bg='#f39c12',
            fg='white',
            activebackground='#d68910',
            relief='raised',
            bd=3,
            height=3
        )
        self.both_rev_btn.pack(fill='x', pady=3)
        self.both_rev_btn.bind('<ButtonPress-1>', lambda e: self.on_motor_press('both_reverse'))
        self.both_rev_btn.bind('<ButtonRelease-1>', lambda e: self.on_motor_release('both_reverse'))
        
        # RIGHT MOTOR CONTROL ARRAY
        right_frame = tk.LabelFrame(
            joystick_container,
            text="Right Motor",
            font=('Arial', 10, 'bold'),
            bg='white',
            fg='#9b59b6',
            padx=10,
            pady=10
        )
        right_frame.grid(row=0, column=2, padx=5, pady=5, sticky='nsew')
        
        # Right Forward Button
        self.right_fwd_btn = tk.Button(
            right_frame,
            text="▲\nFORWARD",
            font=('Arial', 10, 'bold'),
            bg='#27ae60',
            fg='white',
            activebackground='#229954',
            relief='raised',
            bd=3,
            height=3
        )
        self.right_fwd_btn.pack(fill='x', pady=3)
        self.right_fwd_btn.bind('<ButtonPress-1>', lambda e: self.on_motor_press('right_forward'))
        self.right_fwd_btn.bind('<ButtonRelease-1>', lambda e: self.on_motor_release('right_forward'))
        
        # Right Stop Button
        right_stop_btn = tk.Button(
            right_frame,
            text="■\nSTOP",
            font=('Arial', 10, 'bold'),
            bg='#e74c3c',
            fg='white',
            activebackground='#c0392b',
            relief='raised',
            bd=3,
            height=2,
            command=lambda: self.send_arduino_command('s')
        )
        right_stop_btn.pack(fill='x', pady=3)
        
        # Right Reverse Button
        self.right_rev_btn = tk.Button(
            right_frame,
            text="▼\nREVERSE",
            font=('Arial', 10, 'bold'),
            bg='#f39c12',
            fg='white',
            activebackground='#d68910',
            relief='raised',
            bd=3,
            height=3
        )
        self.right_rev_btn.pack(fill='x', pady=3)
        self.right_rev_btn.bind('<ButtonPress-1>', lambda e: self.on_motor_press('right_reverse'))
        self.right_rev_btn.bind('<ButtonRelease-1>', lambda e: self.on_motor_release('right_reverse'))
        
        # Emergency Stop Button (Full Width)
        emergency_stop = tk.Button(
            motor_section,
            text="⚠ EMERGENCY STOP ALL ⚠",
            font=('Arial', 12, 'bold'),
            bg='#c0392b',
            fg='white',
            activebackground='#a93226',
            relief='raised',
            bd=4,
            height=2,
            command=lambda: self.send_arduino_command('s')
        )
        emergency_stop.pack(fill='x', pady=(10, 5))
        
        # ==================== Progress Bar Section =================
        progress_section = tk.LabelFrame(
            scrollable_frame,
            text="Processing Progress",
            font=('Arial', 12, 'bold'),
            bg='white',
            fg='#2c3e50',
            padx=15,
            pady=15
        )
        progress_section.pack(fill='x', padx=10, pady=10)

        # Progress bar
        self.progress_bar = ttk.Progressbar(
            progress_section,
            variable=self.progress_var,
            maximum=100,
            length=300
        )
        self.progress_bar.pack(fill='x', pady=(0, 10))

        # Progress label
        self.progress_label = tk.Label(
            progress_section,
            text="Ready",
            font=('Arial', 10),
            bg='white',
            fg='#2c3e50',
            anchor='w'
        )
        self.progress_label.pack(fill='x')
        
        # ==================== ANALYSIS SECTION ====================
        analysis_section = tk.LabelFrame(
            scrollable_frame,
            text="Analysis Control",
            font=('Arial', 12, 'bold'),
            bg='white',
            fg='#2c3e50',
            padx=15,
            pady=15
        )
        analysis_section.pack(fill='x', padx=10, pady=10)
        
        # Directory selection
        dir_frame = tk.Frame(analysis_section, bg='white')
        dir_frame.pack(fill='x', pady=5)
        
        tk.Label(dir_frame, text="Output Directory:", font=('Arial', 10),
                bg='white').pack(anchor='w')
        
        dir_entry_frame = tk.Frame(dir_frame, bg='white')
        dir_entry_frame.pack(fill='x', pady=5)
        
        dir_entry = tk.Entry(dir_entry_frame, textvariable=self.base_directory,
                           font=('Arial', 9), state='readonly')
        dir_entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        browse_btn = tk.Button(dir_entry_frame, text="Browse...",
                              command=self.select_directory,
                              font=('Arial', 9), bg='#ecf0f1')
        browse_btn.pack(side='right')
        
        # Start/Cancel buttons
        button_frame = tk.Frame(analysis_section, bg='white')
        button_frame.pack(fill='x', pady=10)
        
        self.start_btn = tk.Button(
            button_frame,
            text="Start Analysis",
            command=self.start_processing,
            font=('Arial', 12, 'bold'),
            bg='#27ae60',
            fg='white',
            activebackground='#229954',
            relief='raised',
            bd=3,
            height=2
        )
        self.start_btn.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        self.cancel_btn = tk.Button(
            button_frame,
            text="Cancel",
            command=self.cancel_processing,
            font=('Arial', 12, 'bold'),
            bg='#e74c3c',
            fg='white',
            activebackground='#c0392b',
            relief='raised',
            bd=3,
            height=2,
            state='disabled'
        )
        self.cancel_btn.pack(side='right', fill='x', expand=True, padx=(5, 0))
        
        # ==================== STATUS SECTION ====================
        status_section = tk.LabelFrame(
            scrollable_frame,
            text="System Status",
            font=('Arial', 12, 'bold'),
            bg='white',
            fg='#2c3e50',
            padx=15,
            pady=15
        )
        status_section.pack(fill='both', expand=True, padx=10, pady=10)
        
        status_frame = tk.Frame(status_section, bg='white')
        status_frame.pack(fill='both', expand=True)
        
        self.status_text = tk.Text(
            status_frame,
            height=10,
            font=('Courier', 9),
            wrap='word',
            bg='#ecf0f1',
            relief='sunken',
            bd=2
        )
        self.status_text.pack(side='left', fill='both', expand=True)
        
        status_scroll = tk.Scrollbar(status_frame, command=self.status_text.yview)
        status_scroll.pack(side='right', fill='y')
        self.status_text.config(yscrollcommand=status_scroll.set)
    
    def setup_details_panel(self, parent):
        """Setup the details/results panel"""
        # Create scrollable frame
        canvas = tk.Canvas(parent, bg='white', highlightthickness=0)
        scrollbar = tk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Title
        title = tk.Label(
            scrollable_frame,
            text="Plant Analysis Results",
            font=('Arial', 16, 'bold'),
            bg='white',
            fg='#2c3e50'
        )
        title.pack(pady=15)
        
        # Plant selection
        selection_frame = tk.LabelFrame(
            scrollable_frame,
            text="Select Plant",
            font=('Arial', 11, 'bold'),
            bg='white',
            fg='#2c3e50',
            padx=15,
            pady=10
        )
        selection_frame.pack(fill='x', padx=15, pady=10)
        
        plant_select_frame = tk.Frame(selection_frame, bg='white')
        plant_select_frame.pack(fill='x')
        
        self.plant_dropdown = ttk.Combobox(
            plant_select_frame,
            textvariable=self.selected_plant,
            state='readonly',
            font=('Arial', 10)
        )
        self.plant_dropdown.pack(side='left', fill='x', expand=True, padx=(0, 5))
        self.plant_dropdown.bind('<<ComboboxSelected>>', self.on_plant_selected)
        
        refresh_btn = tk.Button(
            plant_select_frame,
            text="Refresh",
            command=self.load_available_plants,
            font=('Arial', 9),
            bg='#ecf0f1'
        )
        refresh_btn.pack(side='right')
        
        # Results display
        results_frame = tk.LabelFrame(
            scrollable_frame,
            text="Analysis Data",
            font=('Arial', 11, 'bold'),
            bg='white',
            fg='#2c3e50',
            padx=15,
            pady=10
        )
        results_frame.pack(fill='both', expand=True, padx=15, pady=10)
        
        # Create fields for results
        self.detail_fields = {}
        fields = [
            ('Volume (m³)', 'volume'),
            ('Height (cm)', 'height'),
            ('Width (cm)', 'width'),
            ('Depth (cm)', 'depth'),
            ('Surface Area (m²)', 'surface_area'),
            ('Biomass - RF (kg)', 'biomass_rf'),
            ('Processing Time (s)', 'processing_time')
        ]
        
        for i, (label_text, key) in enumerate(fields):
            field_frame = tk.Frame(results_frame, bg='white')
            field_frame.pack(fill='x', pady=5)
            
            label = tk.Label(
                field_frame,
                text="{}:".format(label_text),
                font=('Arial', 10, 'bold'),
                bg='white',
                anchor='w',
                width=18
            )
            label.pack(side='left')
            
            value_label = tk.Label(
                field_frame,
                text="N/A",
                font=('Arial', 10),
                bg='#ecf0f1',
                anchor='w',
                relief='sunken',
                bd=1,
                padx=10,
                pady=5
            )
            value_label.pack(side='left', fill='x', expand=True)
            
            self.detail_fields[key] = value_label
        
        # Action buttons
        action_frame = tk.Frame(scrollable_frame, bg='white')
        action_frame.pack(fill='x', padx=15, pady=15)
        
        # View buttons for 3D files
        view_final_btn = tk.Button(
            action_frame,
            text="Open Final Mesh",
            command=self.view_final_mesh,
            font=('Arial', 10),
            bg='#3498db',
            fg='white',
            activebackground='#2980b9',
            relief='raised',
            bd=2
        )
        view_final_btn.pack(fill='x', pady=3)
        
        view_cloud_btn = tk.Button(
            action_frame,
            text="Open Merged Point Cloud",
            command=self.view_merged_cloud,
            font=('Arial', 10),
            bg='#9b59b6',
            fg='white',
            activebackground='#8e44ad',
            relief='raised',
            bd=2
        )
        view_cloud_btn.pack(fill='x', pady=3)
        
        export_btn = tk.Button(
            action_frame,
            text="Export Results",
            command=self.export_results,
            font=('Arial', 10),
            bg='#95a5a6',
            fg='white',
            activebackground='#7f8c8d',
            relief='raised',
            bd=2
        )
        export_btn.pack(fill='x', pady=3)
    
    # ==================================================================================
    # Motor Control Methods
    # ==================================================================================
    
    def on_motor_press(self, motor_action):
        """Handle motor button press - start continuous movement"""
        if not self.arduino_connected:
            self.log_status("Cannot control motors - Arduino not connected")
            return
        
        # Set the state for this motor action
        self.motor_states[motor_action] = True
        
        # Map motor actions to Arduino commands
        command_map = {
            'left_forward': 'cl',   # Continuous left forward
            'left_reverse': 'bl',   # Continuous left backward/reverse
            'both_forward': 'c',    # Continuous both forward
            'both_reverse': 'b',    # Continuous both backward/reverse
            'right_forward': 'cr',  # Continuous right forward
            'right_reverse': 'br'   # Continuous right backward/reverse
        }
        
        command = command_map.get(motor_action)
        if command:
            self.send_arduino_command(command)
            self.log_status("Motor control: {} started".format(motor_action.replace('_', ' ').title()))
    
    def on_motor_release(self, motor_action):
        """Handle motor button release - stop movement"""
        if not self.arduino_connected:
            return
        
        # Clear the state for this motor action
        self.motor_states[motor_action] = False
        
        # Send stop command
        self.send_arduino_command('s')
        self.log_status("Motor control: {} stopped".format(motor_action.replace('_', ' ').title()))
    
    def send_arduino_command(self, command):
        """Send command to Arduino via serial"""
        if not self.arduino_connected or not self.arduino_serial:
            self.log_status("Cannot send command - Arduino not connected")
            return False
        
        try:
            self.arduino_serial.write((command + '\n').encode())
            self.arduino_serial.flush()
            self.log_status("Sent to Arduino: {}".format(command))
            return True
        except Exception as e:
            self.log_status("Error sending Arduino command: {}".format(str(e)))
            # Connection lost, close serial
            try:
                self.arduino_serial.close()
            except:
                pass
            self.arduino_serial = None
            self.arduino_connected = False
            self.update_arduino_status(False)
            return False
    
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
                        # Try to open serial connection
                        try:
                            if self.arduino_serial:
                                try:
                                    self.arduino_serial.close()
                                except:
                                    pass
                            
                            self.arduino_serial = serial.Serial(port.device, 9600, timeout=1)
                            time.sleep(2)  # Wait for Arduino to reset after opening port
                            
                            self.arduino_port = port.device
                            self.arduino_connected = True
                            arduino_found = True
                            self.update_arduino_status(True)
                            self.log_status("Arduino connected on {}".format(port.device))
                        except (serial.SerialException, OSError) as e:
                            self.log_status("Failed to open Arduino on {}: {}".format(port.device, str(e)))
                            continue
                    else:
                        arduino_found = True
                    break
            
            if not arduino_found and self.arduino_connected:
                if self.arduino_serial:
                    try:
                        self.arduino_serial.close()
                    except:
                        pass
                    self.arduino_serial = None
                self.arduino_connected = False
                self.arduino_port = None
                self.update_arduino_status(False)
                self.log_status("Arduino disconnected")
            
            time.sleep(2)
    
    def update_arduino_status(self, connected):
        """Update Arduino connection status indicator"""
        def update():
            if connected:
                self.arduino_led.itemconfig(self.arduino_indicator, fill='green', outline='darkgreen')
                self.arduino_status_label.config(text="Connected", fg='#27ae60')
            else:
                self.arduino_led.itemconfig(self.arduino_indicator, fill='red', outline='darkred')
                self.arduino_status_label.config(text="Disconnected", fg='#e74c3c')
        
        self.root.after(0, update)
    
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
    
    def update_host_status(self, connected):
        """Update host connection status indicator"""
        def update():
            if connected:
                self.host_led.itemconfig(self.host_indicator, fill='green', outline='darkgreen')
                self.host_status_label.config(text="Connected", fg='#27ae60')
            else:
                self.host_led.itemconfig(self.host_indicator, fill='red', outline='darkred')
                self.host_status_label.config(text="Disconnected", fg='#e74c3c')
        
        self.root.after(0, update)
    
    def listen_to_host(self):
        """Listen for messages from the host"""
        while self.host_connected and self.checking_host:
            try:
                if self.host_socket:
                    # CRITICAL FIX: Sleep while file transfer is in progress
                    if self.receiving_files:
                        time.sleep(0.1)  # Sleep briefly and check again
                        continue
                    
                    self.host_socket.settimeout(1.0)
                    try:
                        data = self.host_socket.recv(1024).decode('utf-8')
                        if data:
                            self.log_status("Received from host: {}".format(data))
                            
                            # Handle file transfer protocol
                            if data == "START_TRANSFER":
                                self.process_host_message(data)
                            elif data.startswith("NUM_FILES:"):
                                self.process_host_message(data)
                            elif data == "END_TRANSFER":
                                self.process_host_message(data)
                            else:
                                self.process_host_message(data)
                        else:
                            self.log_status("Host closed connection")
                            self.host_connected = False
                            break
                    except socket.timeout:
                        continue
                    except UnicodeDecodeError:
                        # Binary data received (file content) - skip decoding
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
    
    def process_host_message(self, message):
        """Handle messages received from the host"""
        if message == "Complete":
            self.log_status("Host reported: Complete")
            self.log_status("Analysis complete!")
            self.root.after(0, lambda: messagebox.showinfo("Success", "Analysis completed successfully"))
            self.plant_count += 1
            self.waiting_for_host_response = False
            # Re-enable buttons
            self.root.after(0, lambda: self.start_btn.config(state='normal'))
            self.root.after(0, lambda: self.cancel_btn.config(state='disabled'))
        elif message == "Failed":
            self.log_status("Host reported: Failed")
            self.root.after(0, lambda: messagebox.showerror("Error", "Host reported analysis failed"))
            self.host_response_received = "Failed"
            self.waiting_for_host_response = False
            # Re-enable buttons
            self.root.after(0, lambda: self.start_btn.config(state='normal'))
            self.root.after(0, lambda: self.cancel_btn.config(state='disabled'))
        elif message == "START_TRANSFER":
            self.log_status("Starting file transfer from host...")
            self.files_received = 0
        elif message.startswith("NUM_FILES:"):
            self.num_files_to_receive = int(message.split(":")[1])
            self.log_status(f"Expecting {self.num_files_to_receive} files from host")
            # CRITICAL: Set receiving_files flag BEFORE starting thread
            self.receiving_files = True  # Move this here
            # Start receiving files in separate thread
            receive_thread = threading.Thread(target=self.receive_all_files, daemon=True)
            receive_thread.start()
        elif message == "END_TRANSFER":
            self.log_status(f"File transfer complete: {self.files_received}/{self.num_files_to_receive} files received")
            self.receiving_files = False
            self.root.after(1000, self.load_available_plants)  # Refresh plant list
            self.send_message_to_host("Transfer Complete")
        elif message == "start_capture":
            self.log_status("Host requested capture start")
        elif message == "status_request":
            self.send_message_to_host("ready")
        elif message == "Plant":
            self.send_message_to_host(f"{self.plant_count}")
        elif message.startswith("PROGRESS:"):
            # Handle the progress updates
            parts = message.split(":", 2)
            if len(parts) == 3:
                try:
                    percentage = int(parts[1]) if parts[1] != 'None' else 0
                    progress_message = parts[2]
                    self.update_progress(progress_message, percentage)
                except (ValueError, IndexError):
                    pass
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
    
    # ==================================================================================
    # File Transfer Functions
    # ==================================================================================
    
    def receive_file(self):
        """
        Receive a single file from the host
        Protocol: [FILENAME_LENGTH][FILENAME][FILE_SIZE][FILE_DATA]
        """
        try:
            # CRITICAL: Remove timeout for file transfer
            self.host_socket.settimeout(None)
            
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
            
            # Create output directory - use reconstruction_output in current directory
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
        # This allows listen_to_host() to resume and receive the END_TRANSFER message
        self.receiving_files = False
        
        self.send_message_to_host("Received")
        self.log_status(f"File reception complete: {self.files_received}/{self.num_files_to_receive} files")
    
    # ==================================================================================
    # Plant Selection and Data Loading
    # ==================================================================================
    
    def select_directory(self):
        """Select output directory"""
        directory = filedialog.askdirectory(initialdir=self.base_directory.get())
        if directory:
            self.base_directory.set(directory)
            self.load_available_plants()
    
    def load_available_plants(self):
        """Load list of available plants from output directory"""
        try:
            base_dir = self.base_directory.get()
            
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
                self.log_status("Created directory: {}".format(base_dir))
            
            if not os.path.isdir(base_dir):
                self.log_status("Invalid directory")
                return
            
            # Scan for any files containing 'plant_' to extract plant numbers
            plant_numbers = set()
            for filename in os.listdir(base_dir):
                if 'plant_' in filename:
                    parts = filename.split('plant_')
                    if len(parts) > 1:
                        # Extract the number after 'plant_'
                        num_str = parts[1].split('.')[0].split('_')[0]
                        try:
                            plant_numbers.add(int(num_str))
                        except ValueError:
                            pass
            
            plant_options = ['Plant {}'.format(n) for n in sorted(plant_numbers)]
            self.plant_dropdown['values'] = plant_options
            
            if plant_options:
                self.plant_dropdown.current(0)
                self.log_status("Found {} plant(s)".format(len(plant_options)))
                self.on_plant_selected(None)
            else:
                self.plant_dropdown.set('')
                self.log_status("No plant files found in {}".format(base_dir))
        except Exception as e:
            self.log_status("Error loading plants: {}".format(str(e)))
    
    def on_plant_selected(self, event):
        """Handle plant selection"""
        plant = self.selected_plant.get()
        if not plant:
            return
        try:
            self.current_plant_number = int(plant.split()[-1])
            self.log_status("Selected: {}".format(plant))
            self.load_plant_data()
        except ValueError:
            self.log_status("Error: Invalid plant selection")
    
    def load_plant_data(self):
        """Load plant data from files"""
        if self.current_plant_number is None:
            return
        
        try:
            base_dir = self.base_directory.get()
            plant_num = self.current_plant_number
            
            # Load reconstruction stats file
            stats_file = os.path.join(base_dir, 
                                     'reconstruction_stats_plant_{}.txt'.format(plant_num))
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    stats_content = f.read()
                self.parse_stats_file(stats_content)
            else:
                # Reset all fields to N/A if no stats file
                for key in self.detail_fields:
                    self.detail_fields[key].config(text='N/A')
                self.log_status("No reconstruction stats file found for Plant {}".format(plant_num))
            
            self.log_status("Plant data loaded successfully")
        except Exception as e:
            self.log_status("Error loading plant data: {}".format(str(e)))
    
    def parse_stats_file(self, content):
        """Parse statistics file to extract values"""
        lines = content.split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if 'volume' in key and 'bbox' not in key:
                    val = value.split()[0] if value else "N/A"
                    self.detail_fields['volume'].config(text=val)
                elif 'height' in key:
                    val = value.split()[0] if value else "N/A"
                    self.detail_fields['height'].config(text=val)
                elif 'width' in key:
                    val = value.split()[0] if value else "N/A"
                    self.detail_fields['width'].config(text=val)
                elif 'depth' in key:
                    val = value.split()[0] if value else "N/A"
                    self.detail_fields['depth'].config(text=val)
                elif 'surface area' in key:
                    val = value.split()[0] if value else "N/A"
                    self.detail_fields['surface_area'].config(text=val)
                elif 'rf biomass' in key:
                    val = value.replace('kg', '').replace('g', '').strip()
                    self.detail_fields['biomass_rf'].config(text=val)
                elif 'total execution time' in key:
                    val = value.split()[0] if value else "N/A"
                    self.detail_fields['processing_time'].config(text=val)
    
    def view_final_mesh(self):
        """Open final mesh in external viewer"""
        if self.current_plant_number is None:
            messagebox.showwarning("Warning", "Please select a plant first")
            return
        filename = 'final_mesh_plant_{}.ply'.format(self.current_plant_number)
        self.open_file(filename)
    
    def view_merged_cloud(self):
        """Open merged point cloud in external viewer"""
        if self.current_plant_number is None:
            messagebox.showwarning("Warning", "Please select a plant first")
            return
        filename = 'merged_point_cloud_plant_{}.ply'.format(self.current_plant_number)
        self.open_file(filename)
    
    def open_file(self, filename):
        """Open file with external viewer"""
        base_dir = self.base_directory.get()
        filepath = os.path.join(base_dir, filename)
        
        if not os.path.exists(filepath):
            messagebox.showerror("Error", "File not found: {}".format(filename))
            return
        
        self.log_status("Opening {}...".format(filename))
        
        viewers = ['meshlab', 'cloudcompare', 'pcl_viewer', 'xdg-open']
        
        opened = False
        for viewer in viewers:
            try:
                subprocess.Popen([viewer, filepath], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
                self.log_status("Opened with {}".format(viewer))
                opened = True
                break
            except (FileNotFoundError, OSError):
                continue
        
        if not opened:
            messagebox.showinfo("Viewer Not Found",
                "No 3D viewer found.\nInstall: meshlab, cloudcompare, or pcl_viewer\n\nFile: {}".format(filepath))
            self.log_status("No viewer available")
    
    def export_results(self):
        """Export results to JSON file"""
        if self.current_plant_number is None:
            messagebox.showwarning("Warning", "Please select a plant first")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="plant_{}_results.json".format(self.current_plant_number)
        )
        
        if filename:
            try:
                results = {}
                for key, label in self.detail_fields.items():
                    results[key] = label.cget('text')
                
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=4)
                
                self.log_status("Results exported to {}".format(filename))
                messagebox.showinfo("Success", "Results exported successfully")
            except Exception as e:
                messagebox.showerror("Error", "Export failed: {}".format(str(e)))
    
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
            # Send "Start" message to host
            self.log_status("Sending 'Start' message to host...")
            if not self.send_message_to_host("Start"):
                self.log_status("Failed to send Start message - host not connected")
                self.root.after(0, lambda: messagebox.showerror("Error", "Cannot communicate with host"))
                self.root.after(0, lambda: self.start_btn.config(state='normal'))
                self.root.after(0, lambda: self.cancel_btn.config(state='disabled'))
                return
            
            # Future: Add rotation control and image capture coordination here
            
        except Exception as e:
            self.log_status("Analysis failed: {}".format(str(e)))
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, lambda: self.start_btn.config(state='normal'))
            self.root.after(0, lambda: self.cancel_btn.config(state='disabled'))
    
    def cancel_processing(self):
        """Cancel the processing"""
        self.is_processing = False
        self.log_status("Processing cancelled")
        self.start_btn.config(state='normal')
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
    
    def update_progress(self, message, percentage):
        """Update progress bar and label - thread-safe"""
        # Update progress bar
        if percentage >= 0:
            self.root.after(0, lambda: self.progress_var.set(percentage))
            self.root.after(0, lambda: self.progress_label.config(
                text=message,
                fg='#2c3e50'
            ))
        else:  # Error state (-1)
            self.root.after(0, lambda: self.progress_label.config(
                text=message,
                fg='#e74c3c'
            ))
        
        # Also log to status
        self.log_status(message)
    
    def cleanup(self):
        """Cleanup resources before closing"""
        self.checking_arduino = False
        self.checking_host = False
        
        # Stop any active motors
        if self.arduino_connected:
            self.send_arduino_command('s')
        
        if self.arduino_serial:
            try:
                self.arduino_serial.close()
            except:
                pass
        
        if self.arduino_check_thread:
            self.arduino_check_thread.join(timeout=1)
        
        if self.host_check_thread:
            self.host_check_thread.join(timeout=1)
        
        if self.host_socket:
            try:
                self.host_socket.close()
            except:
                pass
                
        # Clear all directories for fresh startup
        self.cleanup_directories()
        
    def cleanup_directories(self):
        """
        Delete all files in reconstruction_output and data_collection
        """
        directories_to_clean = [
            os.path.join(os.getcwd(), "reconstruction_output"),
            os.path.join(os.getcwd(), "data_collection")
        ]
        
        for directory in directories_to_clean:
            if os.path.exists(directory):
                try:
                    # Remove all files in the directory
                    for filename in os.listdir(directory):
                        file_path = os.path.join(directory, filename)
                        try:
                            if os.path.isfile(file_path) or os.path.islink(file_path):
                                os.unlink(file_path)
                                print(f"Deleted file: {file_path}")
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                                print(f"Deleted directory: {file_path}")
                        except Exception as e:
                            print(f"Failed to delete {file_path}: {e}")
                    
                    print(f"Cleaned directory: {directory}")
                except Exception as e:
                    print(f"Error cleaning directory {directory}: {e}")

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
