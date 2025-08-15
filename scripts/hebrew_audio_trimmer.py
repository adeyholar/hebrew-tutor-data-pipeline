#!/usr/bin/env python3
"""
Hebrew Audio Intro Trimmer Tool
Removes introductory portions from Hebrew Tanakh audio files
Designed for the Hebrew Tutor App audio processing pipeline
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Audio processing libraries
try:
    import librosa
    import soundfile as sf
    import numpy as np
except ImportError as e:
    print(f"Missing required audio libraries: {e}")
    print("Please install: pip install librosa soundfile numpy")
    sys.exit(1)

# Optional GUI support
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.widgets import Button, TextBox, Slider
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("GUI features disabled. Install matplotlib for visual editing: pip install matplotlib")

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('audio_trimmer.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class HebrewAudioTrimmer:
    """Audio trimming tool for Hebrew Tanakh audio files."""
    
    def __init__(self, audio_base_path: str = "D:/audio/tankh_audio_chp"):
        """Initialize the audio trimmer."""
        self.audio_base_path = Path(audio_base_path)
        self.backup_path = Path(audio_base_path + "_backup")
        self.config_file = Path("trim_config.json")
        
        # Audio settings
        self.sample_rate = 22050  # Standard sample rate for processing
        
        # Load or create configuration
        self.trim_config = self._load_config()
        
        logger.info("Hebrew Audio Trimmer initialized")
        logger.info(f"Audio path: {self.audio_base_path}")
        logger.info(f"Backup path: {self.backup_path}")
        
    def _load_config(self) -> Dict:
        """Load trimming configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration with {len(config)} entries")
                return config
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        
        return {}
    
    def _save_config(self):
        """Save trimming configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.trim_config, f, indent=2)
            logger.info("Configuration saved")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def create_backup(self, file_path: Path) -> bool:
        """Create backup of original file."""
        try:
            backup_dir = self.backup_path / file_path.parent.name
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_file = backup_dir / file_path.name
            
            if not backup_file.exists():
                shutil.copy2(file_path, backup_file)
                logger.info(f"Created backup: {backup_file}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    
    def analyze_audio(self, file_path: Path) -> Dict:
        """Analyze audio file for potential intro detection."""
        try:
            # Load audio
            audio, sr = librosa.load(str(file_path), sr=self.sample_rate)
            duration = len(audio) / sr
            
            # Calculate RMS energy in segments
            segment_length = int(sr * 2)  # 2-second segments
            segments = []
            
            for i in range(0, len(audio), segment_length):
                segment = audio[i:i + segment_length]
                if len(segment) > 0:
                    rms = np.sqrt(np.mean(segment**2))
                    segments.append({
                        'start_time': i / sr,
                        'end_time': min((i + segment_length) / sr, duration),
                        'rms_energy': float(rms)
                    })
            
            # Find potential intro end (significant energy increase)
            if len(segments) >= 3:
                energies = [seg['rms_energy'] for seg in segments]
                avg_energy = np.mean(energies[2:])  # Skip first 2 segments
                threshold = avg_energy * 0.3
                
                # Find first segment that exceeds threshold
                intro_end = 0
                for i, energy in enumerate(energies):
                    if energy > threshold and i > 0:
                        intro_end = segments[i]['start_time']
                        break
                
                # Suggest trim point (add small buffer)
                suggested_trim = max(0, intro_end - 0.5)
            else:
                suggested_trim = 0
            
            return {
                'duration': duration,
                'segments': segments,
                'suggested_trim_start': suggested_trim,
                'file_size_mb': file_path.stat().st_size / (1024 * 1024)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze audio {file_path}: {e}")
            return {'duration': 0, 'segments': [], 'suggested_trim_start': 0}
    
    def trim_audio(self, file_path: Path, trim_start: float, trim_end: Optional[float] = None) -> bool:
        """Trim audio file and save the result."""
        try:
            # Create backup first
            if not self.create_backup(file_path):
                logger.error("Failed to create backup, aborting trim")
                return False
            
            # Load audio
            audio, sr = librosa.load(str(file_path), sr=None)  # Preserve original sample rate
            original_duration = len(audio) / sr
            
            # Calculate trim indices
            start_idx = int(trim_start * sr)
            end_idx = int(trim_end * sr) if trim_end else len(audio)
            
            # Validate indices
            start_idx = max(0, start_idx)
            end_idx = min(len(audio), end_idx)
            
            if start_idx >= end_idx:
                logger.error(f"Invalid trim range: {trim_start} to {trim_end}")
                return False
            
            # Trim audio
            trimmed_audio = audio[start_idx:end_idx]
            new_duration = len(trimmed_audio) / sr
            
            # Save trimmed audio (overwrite original)
            sf.write(str(file_path), trimmed_audio, sr)
            
            # Update configuration
            config_key = str(file_path.relative_to(self.audio_base_path))
            self.trim_config[config_key] = {
                'trim_start': trim_start,
                'trim_end': trim_end,
                'original_duration': original_duration,
                'new_duration': new_duration,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            self._save_config()
            
            logger.info(f"Trimmed {file_path.name}: {original_duration:.1f}s -> {new_duration:.1f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to trim audio {file_path}: {e}")
            return False
    
    def trim_audio_by_length(self, file_path: Path, keep_duration: float, trim_from: str = "start") -> bool:
        """Trim audio to keep specified duration."""
        try:
            # Create backup first
            if not self.create_backup(file_path):
                logger.error("Failed to create backup, aborting trim")
                return False
            
            # Load audio
            audio, sr = librosa.load(str(file_path), sr=None)  # Preserve original sample rate
            original_duration = len(audio) / sr
            
            if keep_duration >= original_duration:
                logger.warning(f"Keep duration ({keep_duration}s) >= original duration ({original_duration:.1f}s)")
                return False
            
            # Calculate trim indices based on trim_from parameter
            if trim_from == "start":
                # Keep the end portion, trim from start
                trim_start = original_duration - keep_duration
                start_idx = int(trim_start * sr)
                end_idx = len(audio)
                trim_info = f"keeping last {keep_duration:.1f}s"
            elif trim_from == "end":
                # Keep the start portion, trim from end
                start_idx = 0
                end_idx = int(keep_duration * sr)
                trim_info = f"keeping first {keep_duration:.1f}s"
            else:
                logger.error(f"Invalid trim_from parameter: {trim_from}")
                return False
            
            # Validate indices
            start_idx = max(0, start_idx)
            end_idx = min(len(audio), end_idx)
            
            if start_idx >= end_idx:
                logger.error(f"Invalid trim indices")
                return False
            
            # Trim audio
            trimmed_audio = audio[start_idx:end_idx]
            new_duration = len(trimmed_audio) / sr
            
            # Save trimmed audio (overwrite original)
            sf.write(str(file_path), trimmed_audio, sr)
            
            # Update configuration
            config_key = str(file_path.relative_to(self.audio_base_path))
            self.trim_config[config_key] = {
                'trim_method': 'by_length',
                'keep_duration': keep_duration,
                'trim_from': trim_from,
                'original_duration': original_duration,
                'new_duration': new_duration,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            self._save_config()
            
            logger.info(f"Trimmed {file_path.name} ({trim_info}): {original_duration:.1f}s -> {new_duration:.1f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to trim audio {file_path}: {e}")
            return False
    
    def find_audio_files(self, pattern: str = "*") -> List[Path]:
        """Find audio files matching pattern."""
        audio_files = []
        
        # Handle absolute path
        if os.path.isabs(pattern):
            file_path = Path(pattern)
            if file_path.exists() and file_path.suffix.lower() == '.mp3':
                return [file_path]
            else:
                return []
        
        if pattern == "*":
            # Find all MP3 files
            for book_dir in self.audio_base_path.iterdir():
                if book_dir.is_dir() and not book_dir.name.startswith('.'):
                    for file_path in book_dir.glob("*.mp3"):
                        audio_files.append(file_path)
        else:
            # Find files matching specific pattern in all directories
            for book_dir in self.audio_base_path.iterdir():
                if book_dir.is_dir() and not book_dir.name.startswith('.'):
                    # Try direct pattern match
                    for file_path in book_dir.glob(f"*{pattern}*"):
                        if file_path.suffix.lower() == '.mp3':
                            audio_files.append(file_path)
        
        return sorted(audio_files)
    
    def interactive_trim(self, file_path: Path):
        """Interactive trimming with audio preview."""
        print(f"\n=== Interactive Trimming: {file_path.name} ===")
        
        # Analyze audio
        analysis = self.analyze_audio(file_path)
        duration = analysis['duration']
        suggested_trim = analysis['suggested_trim_start']
        
        print(f"File: {file_path}")
        print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"File size: {analysis['file_size_mb']:.1f} MB")
        print(f"Suggested trim start: {suggested_trim:.1f} seconds")
        
        # Check if already trimmed
        config_key = str(file_path.relative_to(self.audio_base_path))
        if config_key in self.trim_config:
            prev_trim = self.trim_config[config_key]
            if prev_trim.get('trim_method') == 'by_length':
                print(f"Previously trimmed by length: kept {prev_trim['keep_duration']:.1f}s from {prev_trim['trim_from']}")
            else:
                print(f"Previously trimmed: {prev_trim.get('trim_start', 0):.1f}s")
        
        while True:
            print("\nTrimming Options:")
            print("1. Use suggested trim point (remove intro)")
            print("2. Enter custom trim start/end times")
            print("3. Specify duration to keep")
            print("4. Preview audio segments")
            print("5. Skip this file")
            print("6. Quit")
            
            choice = input("Choose option (1-6): ").strip()
            
            if choice == '1':
                if self.trim_audio(file_path, suggested_trim):
                    print("✅ Audio trimmed successfully")
                break
            elif choice == '2':
                try:
                    print(f"\nEnter trim times (0 to {duration:.1f} seconds)")
                    print("Examples: '5' (start at 5s), '5.5' (start at 5.5s)")
                    custom_start_input = input(f"Enter trim start time: ").strip()
                    
                    # Handle various input formats
                    if '-' in custom_start_input:
                        print("❌ Please enter just the start time (single number)")
                        continue
                    
                    custom_start = float(custom_start_input)
                    
                    print("Enter trim end time (optional):")
                    print("Examples: '300' (end at 300s), leave empty for end of file")
                    custom_end_input = input(f"Enter trim end time (optional, max {duration:.1f}): ").strip()
                    
                    trim_end = None
                    if custom_end_input:
                        if '-' in custom_end_input:
                            print("❌ Please enter just the end time (single number)")
                            continue
                        trim_end = float(custom_end_input)
                    
                    if 0 <= custom_start < duration:
                        if trim_end is None or (custom_start < trim_end <= duration):
                            if self.trim_audio(file_path, custom_start, trim_end):
                                print("✅ Audio trimmed successfully")
                            break
                        else:
                            print("❌ Invalid end time (must be greater than start time)")
                    else:
                        print("❌ Invalid start time")
                except ValueError:
                    print("❌ Invalid number format. Please enter numbers only (e.g., 5.5)")
            elif choice == '3':
                try:
                    print(f"\nCurrent duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
                    print("Examples: '300' (keep 300 seconds = 5 minutes), '150' (keep 2.5 minutes)")
                    keep_input = input(f"Enter duration to keep (seconds, max {duration:.1f}): ").strip()
                    
                    # Handle various input formats
                    if '-' in keep_input:
                        print("❌ Please enter just the duration to keep (single number)")
                        continue
                    
                    keep_duration = float(keep_input)
                    
                    if keep_duration <= 0 or keep_duration >= duration:
                        print("❌ Invalid duration (must be between 0 and total duration)")
                        continue
                    
                    print("\nTrim from:")
                    print("1. Start (keep end portion)")
                    print("2. End (keep start portion)")
                    trim_choice = input("Choose (1-2): ").strip()
                    
                    trim_from = "start" if trim_choice == "1" else "end" if trim_choice == "2" else None
                    
                    if trim_from:
                        # Show what will be removed
                        if trim_from == "start":
                            removed_duration = duration - keep_duration
                            print(f"Will remove first {removed_duration:.1f}s, keep last {keep_duration:.1f}s")
                        else:
                            removed_duration = duration - keep_duration
                            print(f"Will remove last {removed_duration:.1f}s, keep first {keep_duration:.1f}s")
                        
                        confirm = input("Proceed? (y/N): ").strip().lower()
                        if confirm == 'y':
                            if self.trim_audio_by_length(file_path, keep_duration, trim_from):
                                print("✅ Audio trimmed successfully")
                            break
                    else:
                        print("❌ Invalid choice")
                except ValueError:
                    print("❌ Invalid number format")
            elif choice == '4':
                self.preview_segments(analysis)
            elif choice == '5':
                print("⏭️ Skipped")
                break
            elif choice == '6':
                return False  # Signal to quit
            else:
                print("❌ Invalid choice")
        
        return True
    
    def preview_segments(self, analysis: Dict):
        """Preview audio segments for intro detection."""
        segments = analysis['segments']
        if not segments:
            print("No segments to preview")
            return
        
        print("\nAudio Segments (2-second intervals):")
        print("Time Range    | Energy Level | Suggested Intro")
        print("-" * 45)
        
        max_energy = max(seg['rms_energy'] for seg in segments)
        for i, seg in enumerate(segments[:10]):  # Show first 10 segments
            energy_bar = "█" * int((seg['rms_energy'] / max_energy) * 20)
            intro_marker = " <-- SUGGESTED" if i * 2 <= analysis['suggested_trim_start'] <= (i + 1) * 2 else ""
            
            print(f"{seg['start_time']:4.1f}-{seg['end_time']:4.1f}s | {energy_bar:<20} | {intro_marker}")
    
    def batch_trim(self, pattern: str = "*", auto_apply: bool = False):
        """Batch trim multiple files."""
        files = self.find_audio_files(pattern)
        
        if not files:
            print(f"No audio files found matching pattern: {pattern}")
            return
        
        print(f"Found {len(files)} audio files")
        
        if auto_apply:
            print("Auto-applying suggested trim points...")
            success_count = 0
            
            for i, file_path in enumerate(files, 1):
                print(f"Processing {i}/{len(files)}: {file_path.name}")
                
                analysis = self.analyze_audio(file_path)
                suggested_trim = analysis['suggested_trim_start']
                
                if suggested_trim > 1.0:  # Only trim if significant intro detected
                    if self.trim_audio(file_path, suggested_trim):
                        success_count += 1
                        print(f"  ✅ Trimmed {suggested_trim:.1f}s")
                    else:
                        print(f"  ❌ Failed to trim")
                else:
                    print(f"  ⏭️ No significant intro detected")
            
            print(f"\nBatch complete: {success_count}/{len(files)} files trimmed")
        else:
            # Interactive mode
            for file_path in files:
                if not self.interactive_trim(file_path):
                    break  # User chose to quit
    
    def restore_backup(self, file_path: Path) -> bool:
        """Restore file from backup."""
        try:
            backup_file = self.backup_path / file_path.parent.name / file_path.name
            
            if backup_file.exists():
                shutil.copy2(backup_file, file_path)
                
                # Remove from config
                config_key = str(file_path.relative_to(self.audio_base_path))
                if config_key in self.trim_config:
                    del self.trim_config[config_key]
                    self._save_config()
                
                logger.info(f"Restored from backup: {file_path}")
                return True
            else:
                logger.error(f"Backup not found: {backup_file}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return False
    
    def show_statistics(self):
        """Show trimming statistics."""
        if not self.trim_config:
            print("No files have been trimmed yet")
            return
        
        print(f"\n=== Trimming Statistics ===")
        print(f"Total files trimmed: {len(self.trim_config)}")
        
        total_time_saved = 0
        avg_trim_start = 0
        
        for config in self.trim_config.values():
            time_saved = config.get('trim_start', 0)
            if config.get('trim_end'):
                time_saved += (config['original_duration'] - config['trim_end'])
            
            total_time_saved += time_saved
            avg_trim_start += config.get('trim_start', 0)
        
        avg_trim_start /= len(self.trim_config)
        
        print(f"Total time removed: {total_time_saved:.1f} seconds ({total_time_saved/60:.1f} minutes)")
        print(f"Average intro length: {avg_trim_start:.1f} seconds")
        
        # Show recent trims
        print(f"\nRecent trims:")
        sorted_configs = sorted(
            self.trim_config.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )
        
        for file_path, config in sorted_configs[:5]:
            print(f"  {Path(file_path).name}: -{config.get('trim_start', 0):.1f}s ({config['timestamp']})")

def main():
    """Main CLI interface."""
    trimmer = HebrewAudioTrimmer()
    
    if len(sys.argv) == 1:
        # Interactive mode
        while True:
            print("\n=== Hebrew Audio Intro Trimmer ===")
            print("1. Trim single file")
            print("2. Batch trim book")
            print("3. Auto-trim all files")
            print("4. Trim by duration (keep specified length)")
            print("5. Show statistics")
            print("6. Restore from backup")
            print("7. Exit")
            
            choice = input("Choose option (1-7): ").strip()
            
            if choice == '1':
                file_pattern = input("Enter file pattern (e.g., 'Gen_01' or full path): ").strip()
                files = trimmer.find_audio_files(file_pattern)
                
                if files:
                    for file_path in files:
                        if not trimmer.interactive_trim(file_path):
                            break
                else:
                    print("No files found")
                    
            elif choice == '2':
                book_pattern = input("Enter book pattern (e.g., 'Genesis' or 'Gen'): ").strip()
                trimmer.batch_trim(book_pattern)
                
            elif choice == '3':
                confirm = input("Auto-trim ALL files with suggested points? (y/N): ").strip().lower()
                if confirm == 'y':
                    trimmer.batch_trim("*", auto_apply=True)
                    
            elif choice == '4':
                # Duration-based trimming
                file_pattern = input("Enter file pattern (e.g., 'Gen_01' or full path): ").strip()
                files = trimmer.find_audio_files(file_pattern)
                
                if not files:
                    print("No files found")
                    continue
                
                # Show files and their durations
                print(f"\nFound {len(files)} file(s):")
                for i, file_path in enumerate(files, 1):
                    analysis = trimmer.analyze_audio(file_path)
                    duration = analysis['duration']
                    print(f"{i}. {file_path.name}: {duration:.1f}s ({duration/60:.1f}min)")
                
                try:
                    keep_duration = float(input(f"\nEnter duration to keep (seconds): "))
                    
                    print("\nTrim from:")
                    print("1. Start (keep end portion)")
                    print("2. End (keep start portion)")
                    trim_choice = input("Choose (1-2): ").strip()
                    
                    trim_from = "start" if trim_choice == "1" else "end" if trim_choice == "2" else None
                    
                    if trim_from:
                        confirm = input(f"Trim {len(files)} file(s) to keep {keep_duration:.1f}s from {'end' if trim_from == 'start' else 'start'}? (y/N): ").strip().lower()
                        if confirm == 'y':
                            success_count = 0
                            for file_path in files:
                                if trimmer.trim_audio_by_length(file_path, keep_duration, trim_from):
                                    success_count += 1
                                    print(f"✅ Trimmed: {file_path.name}")
                                else:
                                    print(f"❌ Failed: {file_path.name}")
                            
                            print(f"\nCompleted: {success_count}/{len(files)} files trimmed")
                    else:
                        print("❌ Invalid choice")
                except ValueError:
                    print("❌ Invalid number format")
                    
            elif choice == '5':
                trimmer.show_statistics()
                
            elif choice == '6':
                file_pattern = input("Enter file pattern to restore: ").strip()
                files = trimmer.find_audio_files(file_pattern)
                
                for file_path in files:
                    if trimmer.restore_backup(file_path):
                        print(f"✅ Restored: {file_path.name}")
                    else:
                        print(f"❌ Failed to restore: {file_path.name}")
                        
            elif choice == '7':
                print("Goodbye!")
                break
            else:
                print("Invalid choice")
    
    elif len(sys.argv) >= 2:
        command = sys.argv[1]
        
        if command == "analyze":
            # Analyze specific file
            if len(sys.argv) < 3:
                print("Usage: python audio_trimmer.py analyze <file_pattern>")
                return
            
            files = trimmer.find_audio_files(sys.argv[2])
            for file_path in files:
                analysis = trimmer.analyze_audio(file_path)
                print(f"\n{file_path.name}:")
                print(f"  Duration: {analysis['duration']:.1f}s")
                print(f"  Suggested trim: {analysis['suggested_trim_start']:.1f}s")
                trimmer.preview_segments(analysis)
        
        elif command == "trim":
            # Trim with specified parameters
            if len(sys.argv) < 4:
                print("Usage: python audio_trimmer.py trim <file_pattern> <trim_start> [trim_end]")
                return
            
            pattern = sys.argv[2]
            trim_start = float(sys.argv[3])
            trim_end = float(sys.argv[4]) if len(sys.argv) > 4 else None
            
            files = trimmer.find_audio_files(pattern)
            for file_path in files:
                if trimmer.trim_audio(file_path, trim_start, trim_end):
                    print(f"✅ Trimmed: {file_path.name}")
                else:
                    print(f"❌ Failed: {file_path.name}")
        
        elif command == "keep":
            # Trim to keep specified duration
            if len(sys.argv) < 5:
                print("Usage: python audio_trimmer.py keep <file_pattern> <keep_duration> <start|end>")
                return
            
            pattern = sys.argv[2]
            keep_duration = float(sys.argv[3])
            trim_from = sys.argv[4]
            
            if trim_from not in ['start', 'end']:
                print("trim_from must be 'start' or 'end'")
                return
            
            files = trimmer.find_audio_files(pattern)
            for file_path in files:
                if trimmer.trim_audio_by_length(file_path, keep_duration, trim_from):
                    print(f"✅ Trimmed: {file_path.name}")
                else:
                    print(f"❌ Failed: {file_path.name}")
        
        elif command == "batch":
            # Batch process
            pattern = sys.argv[2] if len(sys.argv) > 2 else "*"
            auto = len(sys.argv) > 3 and sys.argv[3] == "--auto"
            trimmer.batch_trim(pattern, auto_apply=auto)
        
        elif command == "stats":
            trimmer.show_statistics()
        
        else:
            print("Unknown command. Available commands: analyze, trim, keep, batch, stats")

if __name__ == "__main__":
    main()