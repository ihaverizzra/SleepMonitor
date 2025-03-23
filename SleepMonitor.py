import cv2
import numpy as np
import pyaudio
import wave
import threading
import time
import os
import datetime
from datetime import datetime
import librosa
import statistics
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import tkinter as tk
from tkinter import ttk, messagebox
import json

class SleepMonitor:
    def __init__(self):
        # Configuration settings
        self.recording = False
        self.sound_threshold = 0.1  # Adjust based on microphone sensitivity
        self.movement_threshold = 50  # Adjust based on camera and lighting
        self.frame_history = []
        self.movement_scores = []
        self.sound_scores = []
        self.timestamps = []
        self.sleep_start_time = None
        self.sleep_end_time = None
        
        # Fan noise and other settings
        self.fan_is_on = False
        self.fan_noise_profile = None
        self.calibration_complete = False
        self.bedtime_reminder = None
        self.wake_time_goal = None
        self.snore_detection = True
        
        # Today's date for folder naming
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.folder_path = os.path.join(os.path.expanduser("~/Desktop"), f"Sleep {self.today}")
        
        # Create folder if it doesn't exist
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
            os.makedirs(os.path.join(self.folder_path, "Audio"))
            os.makedirs(os.path.join(self.folder_path, "Video"))
            
        # Audio settings
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.chunk = 1024
        self.audio_file_counter = 0
        
        # Video settings
        self.video_file_counter = 0
        self.fps = 5  # Lower fps to reduce file size
        self.resolution = (640, 480)
        
        # Sleep analysis 
        self.is_asleep = False
        self.sleep_state_history = []
        self.consecutive_still_frames = 0
        self.consecutive_still_threshold = 20  # About 4 seconds at 5 fps
        
        # Sleep trends data
        self.sleep_history = []
        self.load_sleep_history()

    def load_sleep_history(self):
        """Load previous sleep history if available"""
        history_file = os.path.join(os.path.expanduser("~/Desktop"), "sleep_history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    self.sleep_history = json.load(f)
                print(f"Loaded sleep history with {len(self.sleep_history)} previous records")
            except Exception as e:
                print(f"Error loading sleep history: {e}")
                self.sleep_history = []
        else:
            self.sleep_history = []
    
    def save_sleep_history(self, new_record):
        """Save sleep history to file"""
        self.sleep_history.append(new_record)
        
        # Keep only the last 30 records
        if len(self.sleep_history) > 30:
            self.sleep_history = self.sleep_history[-30:]
        
        history_file = os.path.join(os.path.expanduser("~/Desktop"), "sleep_history.json")
        try:
            with open(history_file, 'w') as f:
                json.dump(self.sleep_history, f)
            print(f"Saved sleep history with {len(self.sleep_history)} records")
        except Exception as e:
            print(f"Error saving sleep history: {e}")
            
    def calibrate_fan_noise(self):
        """Calibrate the system to recognize fan noise"""
        print("Calibrating fan noise. Please ensure your fan is on...")
        
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        # Collect 5 seconds of fan noise
        frames = []
        for i in range(0, int(self.rate / self.chunk * 5)):
            data = stream.read(self.chunk, exception_on_overflow=False)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # Create a temporary file to analyze
        temp_file = os.path.join(self.folder_path, "fan_calibration_temp.wav")
        wf = wave.open(temp_file, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(audio.get_sample_size(self.audio_format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        # Analyze the fan noise characteristics
        try:
            y, sr = librosa.load(temp_file)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            
            # Store the fan noise profile
            self.fan_noise_profile = {
                'centroid_mean': float(np.mean(spectral_centroid)),
                'centroid_std': float(np.std(spectral_centroid)),
                'bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'bandwidth_std': float(np.std(spectral_bandwidth)),
                'amplitude_mean': float(np.mean(np.abs(y))),
                'amplitude_std': float(np.std(np.abs(y)))
            }
            
            # Clean up temp file
            os.remove(temp_file)
            
            self.calibration_complete = True
            print("Fan noise calibration complete.")
            return True
        except Exception as e:
            print(f"Error during fan noise calibration: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return False
    
    def is_fan_noise(self, audio_data):
        """Check if audio matches the fan noise profile"""
        if not self.fan_is_on or not self.calibration_complete or self.fan_noise_profile is None:
            return False
            
        try:
            # Convert audio data to float array for librosa
            audio_float = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Extract audio features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_float, sr=self.rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_float, sr=self.rate)[0]
            
            # Compare with fan noise profile
            centroid_mean = np.mean(spectral_centroid)
            bandwidth_mean = np.mean(spectral_bandwidth)
            amplitude_mean = np.mean(np.abs(audio_float))
            
            # Check if audio features are within the fan noise profile
            centroid_diff = abs(centroid_mean - self.fan_noise_profile['centroid_mean'])
            bandwidth_diff = abs(bandwidth_mean - self.fan_noise_profile['bandwidth_mean'])
            amplitude_ratio = amplitude_mean / self.fan_noise_profile['amplitude_mean']
            
            # Define tolerance thresholds
            if (centroid_diff < self.fan_noise_profile['centroid_std'] * 2 and
                bandwidth_diff < self.fan_noise_profile['bandwidth_std'] * 2 and
                0.5 < amplitude_ratio < 2.0):
                return True
            
            return False
        except Exception as e:
            print(f"Error in fan noise detection: {e}")
            return False
            
    def detect_snoring(self, audio_data):
        """Detect snoring patterns in audio"""
        if not self.snore_detection:
            return False
            
        try:
            # Convert audio data to float array for librosa
            audio_float = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Calculate zero crossing rate and spectral rolloff
            zcr = librosa.feature.zero_crossing_rate(audio_float)[0]
            rolloff = librosa.feature.spectral_rolloff(y=audio_float, sr=self.rate)[0]
            
            # Snoring typically has low ZCR and low rolloff
            zcr_mean = np.mean(zcr)
            rolloff_mean = np.mean(rolloff)
            
            # Check if audio has snoring characteristics
            # Snoring typically has rhythmic pattern, low frequency, and moderate amplitude
            if (zcr_mean < 0.1 and  # Low zero crossing rate
                rolloff_mean < self.rate/4 and  # Low frequency content
                0.01 < np.mean(np.abs(audio_float)) < 0.2):  # Moderate amplitude
                
                # Check for rhythmic pattern
                envelope = np.abs(audio_float)
                envelope_smooth = gaussian_filter1d(envelope, sigma=20)
                peaks, _ = librosa.util.peak_pick(envelope_smooth, 
                                                 pre_max=20, post_max=20, 
                                                 pre_avg=20, post_avg=20, 
                                                 delta=0.01, wait=20)
                
                # If we find rhythmic peaks that could be snoring
                if len(peaks) > 2:
                    peak_intervals = np.diff(peaks)
                    if np.std(peak_intervals) / np.mean(peak_intervals) < 0.3:  # Consistent rhythm
                        return True
            
            return False
        except Exception as e:
            print(f"Error in snore detection: {e}")
            return False
                    
    def start_monitoring(self):
        """Start the sleep monitoring process"""
        self.recording = True
        
        # Start audio and video monitoring threads
        audio_thread = threading.Thread(target=self.monitor_audio)
        video_thread = threading.Thread(target=self.monitor_video)
        
        audio_thread.start()
        video_thread.start()
        
        print(f"Sleep monitoring started. Data will be saved to: {self.folder_path}")
        print("Press Ctrl+C to stop monitoring and generate sleep report.")
        
        try:
            while self.recording:
                time.sleep(1)
        except KeyboardInterrupt:
            self.recording = False
            print("Stopping monitoring...")
            
        # Wait for threads to finish
        audio_thread.join()
        video_thread.join()
        
        # Generate and save report
        self.generate_sleep_report()
        
    def monitor_audio(self):
        """Monitor audio and record when sound exceeds threshold"""
        audio = pyaudio.PyAudio()
        
        # Open stream
        stream = audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        is_recording = False
        frames = []
        quiet_duration = 0
        snoring_detected = False
        
        print("Audio monitoring started")
        
        while self.recording:
            data = stream.read(self.chunk, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # Calculate volume level
            volume_norm = np.abs(audio_data).mean() / 32768.0
            
            current_time = datetime.now()
            self.timestamps.append(current_time)
            self.sound_scores.append(volume_norm)
            
            # Check if this is fan noise when fan is on
            is_fan = self.is_fan_noise(data) if self.fan_is_on else False
            
            # Check for snoring
            is_snoring = self.detect_snoring(data)
            if is_snoring and not snoring_detected:
                snoring_detected = True
                snore_filename = os.path.join(
                    self.folder_path, 
                    "Audio", 
                    f"snoring_{current_time.strftime('%H-%M-%S')}.wav"
                )
                print(f"Snoring detected and recording at {current_time.strftime('%H:%M:%S')}")
            elif not is_snoring:
                snoring_detected = False
            
            # Detect if currently asleep based on sound
            if volume_norm < self.sound_threshold or is_fan:
                if not self.is_asleep and self.consecutive_still_frames >= self.consecutive_still_threshold:
                    if self.sleep_start_time is None:
                        self.sleep_start_time = current_time
                    self.is_asleep = True
                    self.sleep_state_history.append((current_time, True))
            else:
                if self.is_asleep and not is_snoring:  # Don't count snoring as waking up
                    self.is_asleep = False
                    self.sleep_state_history.append((current_time, False))
            
            # Record unusual sounds, but ignore fan noise
            if (volume_norm > self.sound_threshold and not is_fan) or is_snoring:
                if not is_recording:
                    is_recording = True
                    frames = []
                    quiet_duration = 0
                frames.append(data)
            elif is_recording:
                frames.append(data)
                quiet_duration += 1
                
                # Stop recording after 2 seconds of quiet
                if quiet_duration > int(self.rate / self.chunk * 2):
                    is_recording = False
                    if len(frames) > 0:
                        self.save_audio(frames, current_time, is_snoring)
                    frames = []
                    quiet_duration = 0
                    
        # Close audio stream
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
    def save_audio(self, frames, timestamp, is_snoring=False):
        """Save audio frames to a WAV file"""
        file_prefix = "snoring" if is_snoring else "audio"
        filename = os.path.join(
            self.folder_path, 
            "Audio", 
            f"{file_prefix}_{timestamp.strftime('%H-%M-%S')}.wav"
        )
        
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.audio_format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        self.audio_file_counter += 1
        
    def monitor_video(self):
        """Monitor video for movement"""
        cap = cv2.VideoCapture(0)
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        # Get video writer settings
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_out = None
        last_frame = None
        is_recording = False
        
        # Variables for movement detection
        still_duration = 0
        
        print("Video monitoring started")
        
        while self.recording:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture video")
                break
                
            current_time = datetime.now()
            
            # Convert to grayscale for motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            # Calculate motion
            if last_frame is not None:
                frame_delta = cv2.absdiff(last_frame, gray)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                movement_score = np.sum(thresh) / 255
                
                self.movement_scores.append(movement_score)
                
                # Detect significant movement
                if movement_score > self.movement_threshold:
                    # Movement detected
                    if not is_recording:
                        video_filename = os.path.join(
                            self.folder_path,
                            "Video",
                            f"movement_{current_time.strftime('%H-%M-%S')}.avi"
                        )
                        video_out = cv2.VideoWriter(
                            video_filename, 
                            fourcc, 
                            self.fps,
                            self.resolution
                        )
                        is_recording = True
                        self.consecutive_still_frames = 0
                    
                    if video_out is not None:
                        video_out.write(frame)
                    
                    still_duration = 0
                else:
                    # No significant movement
                    self.consecutive_still_frames += 1
                    
                    if is_recording:
                        still_duration += 1
                        video_out.write(frame)
                        
                        # Stop recording after 5 seconds of stillness
                        if still_duration > self.fps * 5:
                            is_recording = False
                            video_out.release()
                            video_out = None
                            still_duration = 0
                            self.video_file_counter += 1
            
            # Update last frame
            last_frame = gray
            
            # Brief sleep to control frame rate
            time.sleep(1/self.fps)
        
        # Clean up
        if video_out is not None:
            video_out.release()
        cap.release()
        
    def generate_sleep_report(self):
        """Generate a sleep quality report"""
        # Determine sleep duration and pattern
        awake_periods = []
        sleep_periods = []
        
        # Smooth the data for better analysis
        if len(self.movement_scores) > 0:
            smooth_movement = gaussian_filter1d(self.movement_scores, sigma=10)
            
            # Detect sleep/wake cycles based on movement
            is_awake = True
            period_start = self.timestamps[0]
            
            for i in range(len(smooth_movement)):
                if is_awake and smooth_movement[i] < self.movement_threshold / 2:
                    # Transition to sleep
                    if i > 0:  # Not the first timestamp
                        awake_periods.append((period_start, self.timestamps[i]))
                    period_start = self.timestamps[i]
                    is_awake = False
                elif not is_awake and smooth_movement[i] > self.movement_threshold:
                    # Transition to awake
                    sleep_periods.append((period_start, self.timestamps[i]))
                    period_start = self.timestamps[i]
                    is_awake = True
            
            # Add the final period
            if is_awake:
                awake_periods.append((period_start, self.timestamps[-1]))
            else:
                sleep_periods.append((period_start, self.timestamps[-1]))
        
        # Calculate sleep metrics
        if self.sleep_start_time is None and len(sleep_periods) > 0:
            self.sleep_start_time = sleep_periods[0][0]
        
        if len(sleep_periods) > 0:
            total_sleep_duration = sum([(end - start).total_seconds() / 3600 for start, end in sleep_periods])
            sleep_efficiency = total_sleep_duration / ((self.timestamps[-1] - self.timestamps[0]).total_seconds() / 3600) * 100
            
            # Calculate average movement during sleep
            sleep_movement_scores = []
            for start, end in sleep_periods:
                start_idx = self.timestamps.index(min(self.timestamps, key=lambda x: abs((x - start).total_seconds())))
                end_idx = self.timestamps.index(min(self.timestamps, key=lambda x: abs((x - end).total_seconds())))
                sleep_movement_scores.extend(self.movement_scores[start_idx:end_idx+1])
            
            if sleep_movement_scores:
                avg_sleep_movement = sum(sleep_movement_scores) / len(sleep_movement_scores)
            else:
                avg_sleep_movement = 0
            
            # Calculate number of awakenings
            num_awakenings = len(sleep_periods) - 1 if len(sleep_periods) > 0 else 0
            
            # Count snoring episodes
            snoring_files = [f for f in os.listdir(os.path.join(self.folder_path, "Audio")) if f.startswith("snoring")]
            snoring_episodes = len(snoring_files)
            
            # Calculate sleep quality score (0-100)
            sleep_quality = max(0, min(100, 100 - (num_awakenings * 5) - (avg_sleep_movement / self.movement_threshold * 30) - (snoring_episodes * 2)))
            
            # Save to sleep history
            sleep_record = {
                "date": self.today,
                "start_time": self.sleep_start_time.strftime('%H:%M:%S') if self.sleep_start_time else "Unknown",
                "duration_hours": round(total_sleep_duration, 2),
                "efficiency": round(sleep_efficiency, 2),
                "quality_score": round(sleep_quality, 2),
                "awakenings": num_awakenings,
                "snoring_episodes": snoring_episodes
            }
            self.save_sleep_history(sleep_record)
            
            # Save the report
            report_path = os.path.join(self.folder_path, "sleep_report.txt")
            with open(report_path, "w") as f:
                f.write("=== SLEEP REPORT ===\n\n")
                f.write(f"Date: {self.today}\n")
                if self.sleep_start_time:
                    f.write(f"Sleep Start Time: {self.sleep_start_time.strftime('%H:%M:%S')}\n")
                else:
                    f.write("Sleep Start Time: Could not determine\n")
                
                f.write(f"Total Sleep Duration: {total_sleep_duration:.2f} hours\n")
                f.write(f"Sleep Efficiency: {sleep_efficiency:.2f}%\n")
                f.write(f"Number of Awakenings: {num_awakenings}\n")
                f.write(f"Snoring Episodes: {snoring_episodes}\n")
                f.write(f"Sleep Quality Score (0-100): {sleep_quality:.2f}\n\n")
                
                f.write("=== SLEEP TIMELINE ===\n")
                for i, (start, end) in enumerate(sleep_periods):
                    duration = (end - start).total_seconds() / 60
                    f.write(f"Sleep Period {i+1}: {start.strftime('%H:%M:%S')} - {end.strftime('%H:%M:%S')} ({duration:.2f} minutes)\n")
                
                f.write("\n=== RECORDINGS SUMMARY ===\n")
                f.write(f"Audio Recordings: {self.audio_file_counter}\n")
                f.write(f"Video Recordings: {self.video_file_counter}\n")
                
                f.write("\n=== TIPS FOR BETTER SLEEP ===\n")
                
                # Provide customized sleep advice
                if num_awakenings > 3:
                    f.write("- You had several awakenings during the night. Consider reducing caffeine intake in the afternoon and evening.\n")
                
                if snoring_episodes > 0:
                    f.write("- Snoring was detected. Try sleeping on your side or using a different pillow height.\n")
                
                if sleep_efficiency < 80:
                    f.write("- Your sleep efficiency was below optimal levels. Try to maintain a consistent sleep schedule.\n")
                
                if self.sleep_start_time and self.sleep_start_time.hour > 0:
                    f.write("- Consider going to bed earlier to improve your sleep duration and quality.\n")
                
            # Generate sleep graph
            self.generate_sleep_graph()
            self.generate_trend_graph()
            
            print(f"Sleep report generated and saved to: {report_path}")
            print(f"Sleep Quality Score: {sleep_quality:.2f}/100")
            if self.sleep_start_time:
                print(f"You fell asleep at approximately: {self.sleep_start_time.strftime('%H:%M:%S')}")
            print(f"Total Sleep Duration: {total_sleep_duration:.2f} hours")
            if snoring_episodes > 0:
                print(f"Snoring episodes detected: {snoring_episodes}")
        else:
            print("Not enough data collected to generate a meaningful sleep report.")
    
    def generate_sleep_graph(self):
        """Generate a graph showing movement and sound patterns"""
        # Convert timestamps to hours since start
        if not self.timestamps:
            return
            
        start_time = self.timestamps[0]
        hours = [(t - start_time).total_seconds() / 3600 for t in self.timestamps]
        
        # Create the graph
        plt.figure(figsize=(12, 8))
        
        # Plot movement
        plt.subplot(2, 1, 1)
        plt.plot(hours, self.movement_scores, 'b-', alpha=0.5, label='Raw Movement')
        
        # Add smoothed movement line
        smooth_movement = gaussian_filter1d(self.movement_scores, sigma=10)
        plt.plot(hours, smooth_movement, 'r-', linewidth=2, label='Smoothed Movement')
        
        plt.axhline(y=self.movement_threshold, color='g', linestyle='--', label='Movement Threshold')
        plt.title('Sleep Movement Pattern')
        plt.xlabel('Hours Since Start')
        plt.ylabel('Movement Intensity')
        plt.legend()
        
        # Plot sound
        plt.subplot(2, 1, 2)
        plt.plot(hours, self.sound_scores, 'g-', alpha=0.5, label='Sound Level')
        
        # Add smoothed sound line
        smooth_sound = gaussian_filter1d(self.sound_scores, sigma=10)
        plt.plot(hours, smooth_sound, 'm-', linewidth=2, label='Smoothed Sound')
        
        plt.axhline(y=self.sound_threshold, color='r', linestyle='--', label='Sound Threshold')
        plt.title('Sleep Sound Pattern')
        plt.xlabel('Hours Since Start')
        plt.ylabel('Sound Level')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path, "sleep_graph.png"))
        
    def generate_trend_graph(self):
        """Generate a graph showing sleep trends over time"""
        if len(self.sleep_history) < 2:
            return
            
        # Extract data for trends
        dates = [record.get('date', '') for record in self.sleep_history]
        quality_scores = [record.get('quality_score', 0) for record in self.sleep_history]
        durations = [record.get('duration_hours', 0) for record in self.sleep_history]
        
        plt.figure(figsize=(12, 8))
        
        # Plot sleep quality trend
        plt.subplot(2, 1, 1)
        plt.plot(dates[-14:], quality_scores[-14:], 'b-o', linewidth=2)
        plt.title('Sleep Quality Trend')
        plt.xlabel('Date')
        plt.ylabel('Sleep Quality Score')
        plt.ylim(0, 100)
        plt.grid(True)
        plt.xticks(rotation=45)
        
        # Plot sleep duration trend
        plt.subplot(2, 1, 2)
        plt.plot(dates[-14:], durations[-14:], 'g-o', linewidth=2)
        plt.title('Sleep Duration Trend')
        plt.xlabel('Date')
        plt.ylabel('Hours of Sleep')
        plt.ylim(0, 12)
        plt.grid(True)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path, "sleep_trend_graph.png"))

class SleepMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sleep Monitor")
        self.root.geometry("600x600")
        self.root.resizable(True, True)
        
        self.monitor = SleepMonitor()
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Sleep Monitor", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=10)
        
        # Fan checkbox
        self.fan_var = tk.BooleanVar(value=False)
        fan_check = ttk.Checkbutton(
            settings_frame, 
            text="I sleep with a fan on", 
            variable=self.fan_var,
            command=self.toggle_fan
        )
        fan_check.grid(row=0, column=0, sticky="w", pady=5)
        
        # Calibrate fan button
        self.calibrate_btn = ttk.Button(
            settings_frame,
            text="Calibrate Fan Noise",
            command=self.calibrate_fan,
            state=tk.DISABLED
        )
        self.calibrate_btn.grid(row=0, column=1, padx=10, pady=5)
        
        # Snore detection checkbox
        self.snore_var = tk.BooleanVar(value=True)
        snore_check = ttk.Checkbutton(
            settings_frame, 
            text="Detect snoring", 
            variable=self.snore_var,
            command=self.toggle_snore_detection
        )
        snore_check.grid(row=1, column=0, sticky="w", pady=5)
        
        # Sensitivity settings
        sensitivity_frame = ttk.LabelFrame(main_frame, text="Sensitivity Settings", padding="10")
        sensitivity_frame.pack(fill=tk.X, pady=10)
        
        # Sound sensitivity
        ttk.Label(sensitivity_frame, text="Sound Sensitivity:").grid(row=0, column=0, sticky="w", pady=5)
        self.sound_sensitivity = ttk.Scale(
            sensitivity_frame, 
            from_=0.05, 
            to=0.5, 
            orient="horizontal",
            length=200,
            command=self.update_sound_sensitivity
        )
        self.sound_sensitivity.set(0.1)
        self.sound_sensitivity.grid(row=0, column=1, padx=10, pady=5)
        
        # Movement sensitivity
        ttk.Label(sensitivity_frame, text="Movement Sensitivity:").grid(row=1, column=0, sticky="w", pady=5)
        self.movement_sensitivity = ttk.Scale(
            sensitivity_frame, 
            from_=20, 
            to=100, 
            orient="horizontal",
            length=200,
            command=self.update_movement_sensitivity
        )
        self.movement_sensitivity.set(50)
        self.movement_sensitivity.grid(row=1, column=1, padx=10, pady=5)
        
        # Bedtime reminder
        reminder_frame = ttk.LabelFrame(main_frame, text="Bedtime Reminder", padding="10")
        reminder_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(reminder_frame, text="Remind me to go to bed at:").grid(row=0, column=0, sticky="w", pady=5)
        
        # Time selector
        time_frame = ttk.Frame(reminder_frame)
        time_frame.grid(row=0, column=1, padx=10, pady=5)
        
        self.hour_var = tk.StringVar()
        hour_spin = ttk.Spinbox(
            time_frame, 
            from_=1, 
            to=12, 
            width=2,
            textvariable=self.hour_var
        )
        hour_spin.set(10)
        hour_spin.pack(side=tk.LEFT)
        
        ttk.Label(time_frame, text=":").pack(side=tk.LEFT)
        
        self.minute_var = tk.StringVar()
        minute_spin = ttk.Spinbox(
            time_frame, 
            from_=0, 
            to=59, 
            width=2,
            textvariable=self.minute_var
        )
        minute_spin.set("00")
        minute_spin.pack(side=tk.LEFT)
        
        self.am_pm_var = tk.StringVar(value="PM")
        am_pm = ttk.Combobox(
            time_frame, 
            values=["AM", "PM"],
            textvariable=self.am_pm_var,
            width=3,
            state="readonly"
        )
        am_pm.pack(side=tk.LEFT, padx=5)
        
        # Enable reminder checkbox
        self.enable_reminder_var = tk.BooleanVar(value=False)
        reminder_check = ttk.Checkbutton(
            reminder_frame, 
            text="Enable reminder", 
            variable=self.enable_reminder_var
        )
        reminder_check.grid(row=1, column=0, sticky="w", pady=5)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        # Start monitoring button
        self.start_btn = ttk.Button(
            button_frame,
            text="Start Sleep Monitoring",
            command=self.start_monitoring,
            width=25
        )
        self.start_btn.pack(side=tk.LEFT, padx=10)
        
        # View history button
        self.history_btn = ttk.Button(
            button_frame,
            text="View Sleep History",
            command=self.view_sleep_history,
            width=25
        )
        self.history_btn.pack(side=tk.LEFT, padx=10)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready to start monitoring")
        status_label = ttk.Label(
            main_frame, 
            textvariable=self.status_var,
            font=("Helvetica", 10, "italic")
        )
        status_label.pack(pady=10)
        
        # Add monitoring thread
        self.monitoring_thread = None
        
    def toggle_fan(self):
        """Toggle fan setting"""
        fan_on = self.fan_var.get()
        self.monitor.fan_is_on = fan_on
        self.calibrate_btn["state"] = tk.NORMAL if fan_on else tk.DISABLED
        self.status_var.set("Fan setting updated" + (" - Calibration needed" if fan_on else ""))
        
    def calibrate_fan(self):
        """Run fan noise calibration"""
        self.status_var.set("Calibrating fan noise... Please ensure your fan is on.")
        self.calibrate_btn["state"] = tk.DISABLED
        
        # Run calibration in a separate thread
        def calibrate_thread():
            success = self.monitor.calibrate_fan_noise()
            if success:
                self.status_var.set("Fan noise calibration complete!")
            else:
                self.status_var.set("Fan calibration failed. Please try again.")
                self.calibrate_btn["state"] = tk.NORMAL
                
        threading.Thread(target=calibrate_thread).start()
        
    def toggle_snore_detection(self):
        """Toggle snore detection"""
        self.monitor.snore_detection = self.snore_var.get()
        
    def update_sound_sensitivity(self, value):
        """Update sound sensitivity threshold"""
        self.monitor.sound_threshold = float(value)
        
    def update_movement_sensitivity(self, value):
        """Update movement sensitivity threshold"""
        self.monitor.movement_threshold = float(value)
        
    def start_monitoring(self):
        """Start sleep monitoring"""
        # Update settings from UI inputs
        self.monitor.fan_is_on = self.fan_var.get()
        self.monitor.snore_detection = self.snore_var.get()
        
        # Set bedtime reminder if enabled
        if self.enable_reminder_var.get():
            hour = int(self.hour_var.get())
            minute = int(self.minute_var.get())
            am_pm = self.am_pm_var.get()
            
            # Convert to 24-hour format
            if am_pm == "PM" and hour < 12:
                hour += 12
            elif am_pm == "AM" and hour == 12:
                hour = 0
                
            self.monitor.bedtime_reminder = f"{hour:02d}:{minute:02d}"
            
        # Disable UI controls
        self.start_btn["state"] = tk.DISABLED
        self.calibrate_btn["state"] = tk.DISABLED
        self.history_btn["state"] = tk.DISABLED
        
        self.status_var.set("Sleep monitoring started... Press Ctrl+C in terminal to stop.")
        
        # Start monitoring in a separate thread
        self.monitoring_thread = threading.Thread(target=self.run_monitoring)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
    def run_monitoring(self):
        """Run the monitoring process"""
        try:
            self.monitor.start_monitoring()
        except Exception as e:
            print(f"Error in monitoring: {e}")
        finally:
            # Re-enable UI controls
            self.root.after(0, self.reset_ui)
            
    def reset_ui(self):
        """Reset UI after monitoring completes"""
        self.start_btn["state"] = tk.NORMAL
        self.history_btn["state"] = tk.NORMAL
        if self.fan_var.get():
            self.calibrate_btn["state"] = tk.NORMAL
            
        self.status_var.set("Monitoring complete. Sleep report generated.")
        
        # Show message box with report summary
        if self.monitor.sleep_start_time:
            sleep_time = self.monitor.sleep_start_time.strftime('%H:%M:%S')
            messagebox.showinfo(
                "Sleep Report", 
                f"Sleep monitoring complete!\n\n"
                f"You fell asleep at: {sleep_time}\n"
                f"Report saved to: {self.monitor.folder_path}"
            )
        else:
            messagebox.showinfo(
                "Sleep Report", 
                f"Sleep monitoring complete!\n\n"
                f"Report saved to: {self.monitor.folder_path}"
            )
            
    def view_sleep_history(self):
        """Open a window to view sleep history"""
        history_window = tk.Toplevel(self.root)
        history_window.title("Sleep History")
        history_window.geometry("800x600")
        
        # Create main frame
        frame = ttk.Frame(history_window, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(frame, text="Sleep History", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Create treeview for history
        columns = ("Date", "Sleep Time", "Duration", "Quality", "Efficiency", "Awakenings", "Snoring")
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        
        # Configure columns
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True)
        
        # Populate with data
        for idx, record in enumerate(self.monitor.sleep_history):
            tree.insert("", "end", values=(
                record.get("date", ""),
                record.get("start_time", ""),
                f"{record.get('duration_hours', 0):.2f} hrs",
                f"{record.get('quality_score', 0):.1f}/100",
                f"{record.get('efficiency', 0):.1f}%",
                record.get("awakenings", 0),
                record.get("snoring_episodes", 0)
            ))
            
        # Add button to display trend graph
        def show_trends():
            if len(self.monitor.sleep_history) < 2:
                messagebox.showinfo("Not Enough Data", "At least 2 nights of sleep data are needed to show trends.")
                return
                
            # Generate trend graph in temporary location
            plt.figure(figsize=(10, 6))
            
            # Extract data for trends
            dates = [record.get('date', '') for record in self.monitor.sleep_history]
            quality_scores = [record.get('quality_score', 0) for record in self.monitor.sleep_history]
            durations = [record.get('duration_hours', 0) for record in self.monitor.sleep_history]
            
            # Plot quality trend
            plt.plot(dates[-14:], quality_scores[-14:], 'b-o', label='Quality Score')
            
            # Create second y-axis for duration
            ax2 = plt.gca().twinx()
            ax2.plot(dates[-14:], durations[-14:], 'r-^', label='Duration (hours)')
            
            plt.title('Sleep Quality and Duration Trends')
            plt.xlabel('Date')
            plt.ylabel('Sleep Quality Score (0-100)')
            ax2.set_ylabel('Sleep Duration (hours)')
            
            # Combine legends
            lines1, labels1 = plt.gca().get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save and open image
            temp_file = os.path.join(os.path.expanduser("~/Desktop"), "sleep_trends_temp.png")
            plt.savefig(temp_file)
            
            # Open the image in default viewer
            import subprocess
            import platform
            
            system = platform.system()
            if system == 'Darwin':  # macOS
                subprocess.call(['open', temp_file])
            elif system == 'Windows':
                os.startfile(temp_file)
            else:  # Linux
                subprocess.call(['xdg-open', temp_file])
        
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=10)
        
        trend_btn = ttk.Button(btn_frame, text="Show Trend Graph", command=show_trends)
        trend_btn.pack(side=tk.LEFT, padx=10)
        
        close_btn = ttk.Button(btn_frame, text="Close", command=history_window.destroy)
        close_btn.pack(side=tk.LEFT, padx=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = SleepMonitorApp(root)
    root.mainloop()