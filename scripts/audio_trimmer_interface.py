# File Location: D:\AI\Gits\hebrew-tutor-data-pipeline\scripts\audio_trimmer_interface.py

import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import logging

# Configure logging (consistent with pipeline)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('audio_trimmer_interface.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Embedded AudioTrimmer class (self-contained to fix import bug)
class AudioTrimmer:
    """Utility for trimming Tanakh audio files losslessly using FFmpeg."""
    
    def __init__(self, ffmpeg_path: str = "D:/audio/ffmpeg/ffmpeg.exe"):
        """
        Initialize with FFmpeg path.
        
        Args:
            ffmpeg_path: Path to FFmpeg executable (default based on project structure).
        """
        self.ffmpeg_path = Path(ffmpeg_path)
        if not self.ffmpeg_path.exists():
            raise FileNotFoundError(f"FFmpeg not found at {self.ffmpeg_path}")
        logger.info(f"Initialized AudioTrimmer with FFmpeg at {self.ffmpeg_path}")
    
    def trim_audio(
        self,
        input_path: str,
        output_path: str,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        copy_codec: bool = True
    ) -> bool:
        """
        Trim audio file from start_time to end_time (in seconds) without quality loss.
        
        Args:
            input_path: Path to input audio file (e.g., MP3).
            output_path: Path to save trimmed audio.
            start_time: Start time in seconds (default 0.0).
            end_time: End time in seconds (None for end of file).
            copy_codec: If True, copy streams without re-encoding (preserves quality).
        
        Returns:
            True if successful, False otherwise.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return False
        
        cmd = [str(self.ffmpeg_path), "-i", str(input_path), "-ss", str(start_time)]
        
        if end_time is not None:
            duration = end_time - start_time
            cmd.extend(["-t", str(duration)])
        
        if copy_codec:
            cmd.extend(["-acodec", "copy", "-vcodec", "copy"])
        
        cmd.append(str(output_path))
        
        try:
            logger.info(f"Executing FFmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.debug(f"FFmpeg output: {result.stdout}")
            if output_path.exists():
                logger.info(f"Trimmed audio saved: {output_path}")
                return True
            else:
                logger.error("Output file not created")
                return False
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return False
    
    def auto_trim_silence(
        self,
        input_path: str,
        output_path: str,
        silence_threshold: float = -40.0,
        silence_duration: float = 0.5
    ) -> bool:
        """
        Automatically trim leading/trailing silence.
        
        Args:
            input_path: Input audio path.
            output_path: Output path.
            silence_threshold: dB threshold for silence (default -40dB).
            silence_duration: Minimum silence duration in seconds.
        
        Returns:
            True if successful.
        """
        # FFmpeg filter for silence removal
        cmd = [
            str(self.ffmpeg_path), "-i", str(input_path),
            "-af", f"silenceremove=start_periods=1:start_duration={silence_duration}:start_threshold={silence_threshold}dB,"
                   f"stop_periods=1:stop_duration={silence_duration}:stop_threshold={silence_threshold}dB",
            "-acodec", "copy", "-vcodec", "copy",
            str(output_path)
        ]
        
        try:
            logger.info(f"Auto-trimming silence with command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.debug(f"FFmpeg output: {result.stdout}")
            if Path(output_path).exists():
                logger.info(f"Silence-trimmed audio saved: {output_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Auto-trim error: {str(e)}")
            return False

class AudioTrimmerGUI:
    """Tkinter-based GUI for selecting and trimming Tanakh audio files."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hebrew Tutor Audio Trimmer")
        self.root.geometry("500x300")
        
        # Default paths
        self.default_audio_path = r"D:\audio\tankh_audio_chp\01_Genesis\01-Gen_01.mp3"
        self.default_output_dir = r"D:\audio\tankh_audio_chp\01_Genesis"
        
        # Initialize trimmer
        self.trimmer = AudioTrimmer(ffmpeg_path=r"D:\audio\ffmpeg\ffmpeg.exe")
        
        # UI Elements
        self.create_widgets()
        
        self.root.mainloop()
    
    def create_widgets(self):
        """Create GUI widgets."""
        # Input Audio Path
        tk.Label(self.root, text="Input Audio Path:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.input_path_var = tk.StringVar(value=self.default_audio_path)
        self.input_entry = tk.Entry(self.root, textvariable=self.input_path_var, width=50)
        self.input_entry.grid(row=0, column=1, padx=10, pady=5)
        tk.Button(self.root, text="Browse", command=self.browse_input).grid(row=0, column=2, padx=10, pady=5)
        
        # Start Time
        tk.Label(self.root, text="Start Time (seconds):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.start_time_var = tk.DoubleVar(value=0.0)
        tk.Entry(self.root, textvariable=self.start_time_var, width=10).grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        # End Time (optional)
        tk.Label(self.root, text="End Time (seconds, optional):").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.end_time_var = tk.StringVar()  # Use StringVar for optional empty
        tk.Entry(self.root, textvariable=self.end_time_var, width=10).grid(row=2, column=1, padx=10, pady=5, sticky="w")
        
        # Auto Trim Silence Checkbox
        self.auto_silence_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.root, text="Auto Trim Silence", variable=self.auto_silence_var).grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        
        # Output Path
        tk.Label(self.root, text="Output Path:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.output_path_var = tk.StringVar(value=os.path.join(self.default_output_dir, "trimmed_gen_01.mp3"))
        self.output_entry = tk.Entry(self.root, textvariable=self.output_path_var, width=50)
        self.output_entry.grid(row=4, column=1, padx=10, pady=5)
        tk.Button(self.root, text="Browse", command=self.browse_output).grid(row=4, column=2, padx=10, pady=5)
        
        # Trim Button
        tk.Button(self.root, text="Trim Audio", command=self.perform_trim).grid(row=5, column=0, columnspan=3, pady=20)
    
    def browse_input(self):
        """Browse for input audio file."""
        file_path = filedialog.askopenfilename(
            initialdir=r"D:\audio\tankh_audio_chp",
            title="Select Audio File",
            filetypes=(("MP3 files", "*.mp3"), ("All files", "*.*"))
        )
        if file_path:
            self.input_path_var.set(file_path)
            # Auto-suggest output path
            output_dir = os.path.dirname(file_path)
            output_name = "trimmed_" + os.path.basename(file_path)
            self.output_path_var.set(os.path.join(output_dir, output_name))
    
    def browse_output(self):
        """Browse for output save location."""
        file_path = filedialog.asksaveasfilename(
            initialdir=self.default_output_dir,
            title="Save Trimmed Audio As",
            defaultextension=".mp3",
            filetypes=(("MP3 files", "*.mp3"), ("All files", "*.*"))
        )
        if file_path:
            self.output_path_var.set(file_path)
    
    def perform_trim(self):
        """Execute the trim operation."""
        input_path = self.input_path_var.get()
        output_path = self.output_path_var.get()
        start_time = self.start_time_var.get()
        end_time_str = self.end_time_var.get()
        end_time = float(end_time_str) if end_time_str else None
        auto_silence = self.auto_silence_var.get()
        
        if not input_path or not output_path:
            messagebox.showerror("Error", "Please select input and output paths.")
            return
        
        try:
            if auto_silence:
                success = self.trimmer.auto_trim_silence(input_path, output_path)
            else:
                success = self.trimmer.trim_audio(input_path, output_path, start_time, end_time)
            
            if success:
                messagebox.showinfo("Success", f"Audio trimmed and saved to {output_path}")
                logger.info(f"Trim successful: {input_path} -> {output_path}")
            else:
                messagebox.showerror("Error", "Trimming failed. Check logs for details.")
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error: {str(e)}")
            logger.error(f"Trim error: {str(e)}")

# Run the GUI
if __name__ == "__main__":
    AudioTrimmerGUI()