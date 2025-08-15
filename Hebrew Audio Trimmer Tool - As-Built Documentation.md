# Hebrew Audio Trimmer Tool - As-Built Documentation

**Project**: Hebrew Tutor App - Audio Processing Pipeline  
**Component**: Audio Intro Trimmer Tool  
**Version**: 1.0  
**Date**: August 15, 2025  
**Author**: Ben Elohim  
**File**: `hebrew_audio_trimmer.py`

---

## 1. Executive Summary

The Hebrew Audio Trimmer Tool is a specialized audio processing utility designed to remove introductory portions and trim Hebrew Tanakh audio files for the Hebrew Tutor App. The tool provides both interactive and command-line interfaces for precise audio editing while maintaining data integrity through automatic backup systems.

### 1.1 Primary Functions
- Remove introductory segments from audio files
- Trim audio to specific durations
- Batch process multiple files
- Automatic backup and restore capabilities
- Intelligent intro detection using energy analysis

### 1.2 Integration Context
This tool is part of the larger Hebrew Tutor App audio-text alignment pipeline, working in conjunction with:
- `hebrew_alignment_production.py` - Main alignment processing
- Audio files stored in `D:\audio\tankh_audio_chp\` (929 MP3 files)
- Hebrew Bible JSON data for text synchronization

---

## 2. System Architecture

### 2.1 Core Components

```
HebrewAudioTrimmer
├── Configuration Management
│   ├── JSON-based config storage
│   └── Backup path management
├── Audio Processing Engine
│   ├── Librosa-based audio loading
│   ├── Energy analysis algorithms
│   └── Precision trimming operations
├── File Management System
│   ├── Pattern-based file discovery
│   ├── Automatic backup creation
│   └── Restore functionality
├── User Interfaces
│   ├── Interactive console interface
│   └── Command-line interface
└── Logging and Monitoring
    ├── UTF-8 encoded logging
    └── Operation tracking
```

### 2.2 Dependencies

#### Required Libraries
```python
import os, sys, json, time, shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import librosa          # Audio processing
import soundfile as sf  # Audio I/O
import numpy as np      # Numerical operations
```

#### Optional Libraries
```python
import matplotlib.pyplot as plt  # GUI features (disabled if unavailable)
from matplotlib.widgets import Button, TextBox, Slider
```

---

## 3. Detailed Component Analysis

### 3.1 HebrewAudioTrimmer Class

#### 3.1.1 Initialization
```python
def __init__(self, audio_base_path: str = "D:/audio/tankh_audio_chp")
```

**Purpose**: Initialize the trimmer with default paths and configuration

**Key Attributes**:
- `audio_base_path`: Source directory for Hebrew audio files
- `backup_path`: Automatic backup location (`{audio_base_path}_backup`)
- `config_file`: Operation tracking file (`trim_config.json`)
- `sample_rate`: Audio processing sample rate (22050 Hz)
- `trim_config`: Dictionary storing all trim operations

**Directory Structure Expected**:
```
D:/audio/tankh_audio_chp/
├── 01_Genesis/
│   ├── 01-Gen_01.mp3
│   ├── 01-Gen_02.mp3
│   └── ...
├── 02_Exodus/
│   ├── 02-Exo_01.mp3
│   └── ...
└── [39 total book directories]
```

#### 3.1.2 Configuration Management

##### `_load_config() -> Dict`
**Purpose**: Load existing trim configuration from JSON file

**Implementation**:
- Checks for `trim_config.json` existence
- Loads JSON data with error handling
- Returns empty dict if file doesn't exist or is corrupted
- Logs configuration load status

##### `_save_config()`
**Purpose**: Persist trim configuration to JSON file

**Data Structure**:
```json
{
  "01_Genesis/01-Gen_01.mp3": {
    "trim_method": "by_length|standard",
    "trim_start": 3.5,
    "trim_end": null,
    "keep_duration": 300.0,
    "trim_from": "start|end",
    "original_duration": 362.1,
    "new_duration": 300.0,
    "timestamp": "2025-08-15 19:04:14"
  }
}
```

### 3.2 Audio Processing Engine

#### 3.2.1 Audio Analysis
##### `analyze_audio(file_path: Path) -> Dict`
**Purpose**: Analyze audio file for intelligent intro detection

**Algorithm**:
1. Load audio using librosa with 22050 Hz sample rate
2. Divide into 2-second segments
3. Calculate RMS energy for each segment
4. Determine energy threshold (30% of average energy from segment 3+)
5. Find first segment exceeding threshold
6. Suggest trim point with 0.5s buffer

**Return Structure**:
```python
{
    'duration': float,              # Total file duration
    'segments': List[Dict],         # 2-second segment analysis
    'suggested_trim_start': float,  # AI-suggested trim point
    'file_size_mb': float          # File size in MB
}
```

**Segment Structure**:
```python
{
    'start_time': float,    # Segment start time
    'end_time': float,      # Segment end time  
    'rms_energy': float     # RMS energy level
}
```

#### 3.2.2 Core Trimming Operations

##### `trim_audio(file_path: Path, trim_start: float, trim_end: Optional[float]) -> bool`
**Purpose**: Standard audio trimming with start/end points

**Process Flow**:
1. **Backup Creation**: Automatic backup to `{backup_path}/{book_dir}/{filename}`
2. **Audio Loading**: Load with original sample rate preservation
3. **Index Calculation**: Convert time to sample indices
4. **Validation**: Ensure valid trim range
5. **Trimming**: Extract audio segment using numpy slicing
6. **Saving**: Overwrite original file with trimmed version
7. **Configuration Update**: Log operation in `trim_config.json`

**Error Handling**:
- Backup failure aborts operation
- Invalid trim ranges logged and rejected
- Audio loading errors caught and logged

##### `trim_audio_by_length(file_path: Path, keep_duration: float, trim_from: str) -> bool`
**Purpose**: Duration-based trimming (keep specific length)

**Trim Modes**:
- `trim_from="start"`: Keep end portion (remove intro)
- `trim_from="end"`: Keep start portion (remove ending)

**Implementation**:
```python
if trim_from == "start":
    # Keep the end portion, trim from start
    trim_start = original_duration - keep_duration
    start_idx = int(trim_start * sr)
    end_idx = len(audio)
elif trim_from == "end":
    # Keep the start portion, trim from end
    start_idx = 0
    end_idx = int(keep_duration * sr)
```

### 3.3 File Management System

#### 3.3.1 File Discovery
##### `find_audio_files(pattern: str) -> List[Path]`
**Purpose**: Locate audio files matching user patterns

**Pattern Support**:
- **Absolute paths**: `D:\audio\tankh_audio_chp\01_Genesis\01-Gen_01.mp3`
- **Wildcard**: `*` (all MP3 files)
- **Partial names**: `Gen_01` (finds `01-Gen_01.mp3`)
- **Book names**: `Genesis` (finds all Genesis files)

**Algorithm**:
1. Check if pattern is absolute path
2. If wildcard, scan all directories
3. For partial names, search all book directories with pattern matching
4. Return sorted list of Path objects

#### 3.3.2 Backup Management
##### `create_backup(file_path: Path) -> bool`
**Purpose**: Create automatic backup before any modification

**Backup Structure**:
```
D:/audio/tankh_audio_chp_backup/
├── 01_Genesis/
│   ├── 01-Gen_01.mp3  # Original backup
│   └── ...
└── ...
```

**Features**:
- Only creates backup if none exists (prevents overwriting)
- Preserves original file metadata
- Creates directory structure as needed
- Returns success/failure status

##### `restore_backup(file_path: Path) -> bool`
**Purpose**: Restore original file from backup

**Process**:
1. Locate corresponding backup file
2. Copy backup over current file
3. Remove entry from `trim_config.json`
4. Log restoration operation

### 3.4 User Interface Systems

#### 3.4.1 Interactive Console Interface
##### `interactive_trim(file_path: Path) -> bool`
**Purpose**: Provide guided trimming experience

**Menu Options**:
1. **Use suggested trim point**: Apply AI-detected intro removal
2. **Enter custom trim times**: Manual start/end specification
3. **Specify duration to keep**: Duration-based trimming
4. **Preview audio segments**: Display energy analysis
5. **Skip this file**: Continue to next file
6. **Quit**: Exit application

**Input Validation**:
- Detects range inputs (e.g., "0-10") and provides guidance
- Validates numerical formats
- Ensures trim ranges are within file duration
- Provides clear error messages with examples

**Visual Feedback**:
```
Time Range    | Energy Level | Suggested Intro
---------------------------------------------
 0.0- 2.0s | ████████             |  <-- SUGGESTED
 2.0- 4.0s | ████████████████     |
 4.0- 6.0s | ████████████         |
```

#### 3.4.2 Command Line Interface
**Available Commands**:

```bash
# Interactive mode (default)
python hebrew_audio_trimmer.py

# Analysis only
python hebrew_audio_trimmer.py analyze <pattern>

# Direct trimming
python hebrew_audio_trimmer.py trim <pattern> <start> [end]

# Duration-based trimming
python hebrew_audio_trimmer.py keep <pattern> <duration> <start|end>

# Batch processing
python hebrew_audio_trimmer.py batch <pattern> [--auto]

# Statistics
python hebrew_audio_trimmer.py stats
```

### 3.5 Batch Processing System

#### 3.5.1 `batch_trim(pattern: str, auto_apply: bool)`
**Purpose**: Process multiple files with minimal user intervention

**Manual Mode** (`auto_apply=False`):
- Calls `interactive_trim()` for each file
- User can quit at any point
- Full control over each file

**Auto Mode** (`auto_apply=True`):
- Applies AI-suggested trim points automatically
- Only trims if suggested trim > 1.0 seconds
- Provides progress feedback
- Reports success/failure statistics

### 3.6 Statistics and Monitoring

#### 3.6.1 `show_statistics()`
**Purpose**: Display comprehensive trimming analytics

**Metrics Calculated**:
- Total files trimmed
- Total time removed (seconds/minutes)
- Average intro length
- Recent trim operations (last 5)

**Sample Output**:
```
=== Trimming Statistics ===
Total files trimmed: 15
Total time removed: 52.3 seconds (0.9 minutes)
Average intro length: 3.5 seconds

Recent trims:
  01-Gen_01.mp3: -1.5s (2025-08-15 19:04:14)
  02-Exo_01.mp3: -2.1s (2025-08-15 18:55:23)
```

---

## 4. Configuration and Data Management

### 4.1 Configuration File Structure

#### `trim_config.json`
**Purpose**: Persistent storage of all trim operations

**Schema**:
```json
{
  "relative_file_path": {
    "trim_method": "standard|by_length",
    "trim_start": number,           // Start trim point (seconds)
    "trim_end": number|null,        // End trim point (optional)
    "keep_duration": number,        // Duration kept (by_length only)
    "trim_from": "start|end",       // Trim direction (by_length only)
    "original_duration": number,    // Original file length
    "new_duration": number,         // Post-trim length
    "timestamp": "YYYY-MM-DD HH:MM:SS"
  }
}
```

### 4.2 Logging Configuration

#### Log File: `audio_trimmer.log`
**Encoding**: UTF-8 (prevents Unicode errors)
**Format**: `%(asctime)s - %(levelname)s - %(message)s`

**Log Levels Used**:
- `INFO`: Successful operations, initialization
- `WARNING`: Non-fatal issues, backup concerns  
- `ERROR`: Failed operations, invalid inputs

**Sample Log Entries**:
```
2025-08-15 19:04:14,977 - INFO - Hebrew Audio Trimmer initialized
2025-08-15 19:04:14,977 - INFO - Audio path: D:\audio\tankh_audio_chp
2025-08-15 19:13:24,680 - INFO - Created backup: D:\...\01-Gen_01.mp3
2025-08-15 19:13:25,394 - INFO - Trimmed 01-Gen_01.mp3: 362.0s -> 360.5s
```

---

## 5. Error Handling and Recovery

### 5.1 Error Categories

#### 5.1.1 File System Errors
- **Missing files**: Graceful handling with user notification
- **Permission errors**: Logged with recommended actions
- **Disk space**: Caught during save operations

#### 5.1.2 Audio Processing Errors
- **Corrupt audio files**: Librosa error handling
- **Invalid formats**: Format validation before processing
- **Memory limitations**: Sample rate optimization

#### 5.1.3 User Input Errors
- **Invalid ranges**: Range detection and correction guidance
- **Format errors**: Type conversion with helpful examples
- **Out-of-bounds**: Validation against file duration

### 5.2 Recovery Mechanisms

#### 5.2.1 Automatic Backup System
- **Backup before modification**: Zero data loss guarantee
- **Backup verification**: File size and integrity checks
- **Restore functionality**: One-command restoration

#### 5.2.2 Configuration Recovery
- **Graceful config loading**: Continues with empty config if corrupted
- **Atomic saves**: Prevents partial configuration writes
- **Manual recovery**: Direct JSON file editing support

---

## 6. Performance Characteristics

### 6.1 Processing Performance

#### File Size Handling
- **Small files** (<1MB): Near-instantaneous processing
- **Medium files** (1-10MB): 1-3 seconds processing time
- **Large files** (>10MB): 5-10 seconds processing time

#### Memory Usage
- **Base memory**: ~50MB (libraries loaded)
- **Per file**: +~2x file size during processing
- **Optimization**: Sequential processing prevents memory accumulation

### 6.2 Scalability

#### Batch Processing
- **Small batches** (1-10 files): Interactive feedback
- **Medium batches** (10-50 files): Progress indicators
- **Large batches** (50+ files): Auto-mode recommended

#### File System Impact
- **Backup storage**: ~100% of original data size
- **Temporary space**: Minimal (in-memory processing)
- **I/O optimization**: Sequential read/write operations

---

## 7. Integration Points

### 7.1 Hebrew Tutor App Pipeline

#### Upstream Dependencies
- **Audio source**: Tanakh audio files in structured directories
- **File naming**: Specific format (`XX-YYY_ZZ.mp3`)

#### Downstream Impact
- **Alignment pipeline**: Trimmed files maintain timing synchronization
- **Duration changes**: Tracked in configuration for pipeline adjustment

### 7.2 File System Integration

#### Expected Directory Structure
```
D:/audio/tankh_audio_chp/          # Source files
D:/audio/tankh_audio_chp_backup/   # Automatic backups
trim_config.json                   # Operation tracking
audio_trimmer.log                  # Activity logging
```

#### File Naming Conventions
- **Book directories**: `XX_BookName` (e.g., `01_Genesis`)
- **Audio files**: `XX-YYY_ZZ.mp3` (e.g., `01-Gen_01.mp3`)
- **Backup preservation**: Maintains original structure

---

## 8. Security and Data Integrity

### 8.1 Data Protection

#### Backup Strategy
- **Automatic creation**: No user intervention required
- **Overwrite protection**: Existing backups preserved
- **Verification**: File existence and size validation

#### Operation Logging
- **Complete audit trail**: All operations logged with timestamps
- **Configuration tracking**: Reversible operation records
- **Error documentation**: Failed operations recorded

### 8.2 Input Validation

#### File Path Security
- **Path traversal protection**: Restricts operations to designated directories
- **Absolute path handling**: Secure handling of user-provided paths
- **Pattern injection prevention**: Safe glob pattern processing

#### Audio Processing Security
- **Format validation**: Ensures MP3 format before processing
- **Size limitations**: Prevents processing of extremely large files
- **Memory bounds**: Sample rate limits prevent memory exhaustion

---

## 9. Maintenance and Operations

### 9.1 Routine Maintenance

#### Log Management
- **Manual rotation**: Log files grow with usage
- **Size monitoring**: Monitor `audio_trimmer.log` size
- **Archive strategy**: Periodic backup of operation logs

#### Configuration Cleanup
- **Orphaned entries**: Remove config entries for deleted files
- **Validation**: Periodic config file integrity checks

### 9.2 Troubleshooting Guide

#### Common Issues

**Unicode Encoding Errors**:
- **Symptom**: Console encoding errors with arrow characters
- **Solution**: Log files use UTF-8, console uses ASCII arrows
- **Prevention**: Avoid Unicode characters in console output

**File Not Found**:
- **Symptom**: Pattern matching returns empty results
- **Diagnosis**: Check pattern format and file existence
- **Solution**: Use absolute paths or verify directory structure

**Backup Failures**:
- **Symptom**: Operations abort with backup errors
- **Diagnosis**: Check disk space and permissions
- **Solution**: Ensure write access to backup directory

**Audio Processing Errors**:
- **Symptom**: Librosa loading failures
- **Diagnosis**: Verify MP3 format and file integrity
- **Solution**: Use audio repair tools or re-encode files

---

## 10. Future Enhancement Opportunities

### 10.1 Planned Features

#### Advanced Audio Analysis
- **Silence detection**: More sophisticated intro identification
- **Voice activity detection**: Content-based trimming
- **Frequency analysis**: Music vs. speech differentiation

#### User Interface Improvements
- **GUI interface**: Visual waveform editing
- **Batch templates**: Saved trimming profiles
- **Preview playback**: Audio segment preview

#### Integration Enhancements
- **Pipeline integration**: Direct connection to alignment tool
- **Cloud storage**: Azure Blob Storage integration
- **Parallel processing**: Multi-threaded batch operations

### 10.2 Technical Debt

#### Code Organization
- **Class decomposition**: Split UI and processing concerns
- **Configuration management**: More robust config system
- **Error handling**: Centralized error management

#### Performance Optimization
- **Streaming processing**: Reduce memory usage for large files
- **Caching**: Intelligent audio analysis caching
- **Async operations**: Non-blocking file operations

---

## 11. Appendix

### 11.1 File Specifications

#### Audio File Format
- **Format**: MP3
- **Sample rates**: Variable (preserved during processing)
- **Channels**: Mono/Stereo (preserved)
- **Bitrates**: Variable (preserved)

#### Configuration File Format
- **Format**: JSON
- **Encoding**: UTF-8
- **Schema version**: 1.0
- **Backward compatibility**: Maintained

### 11.2 Command Reference

#### Interactive Mode Commands
```
1. Trim single file          - Process individual files
2. Batch trim book          - Process book collections  
3. Auto-trim all files      - Automated processing
4. Trim by duration         - Duration-based trimming
5. Show statistics          - Display analytics
6. Restore from backup      - Recovery operations
7. Exit                     - Clean shutdown
```

#### CLI Command Syntax
```bash
# Analysis mode
python hebrew_audio_trimmer.py analyze <pattern>

# Direct trimming
python hebrew_audio_trimmer.py trim <pattern> <start> [end]

# Duration trimming  
python hebrew_audio_trimmer.py keep <pattern> <duration> <start|end>

# Batch processing
python hebrew_audio_trimmer.py batch <pattern> [--auto]

# Statistics
python hebrew_audio_trimmer.py stats
```

### 11.3 Directory Structure Reference

```
Project Root/
├── hebrew_audio_trimmer.py     # Main application
├── trim_config.json           # Operation configuration  
├── audio_trimmer.log          # Activity log
├── D:/audio/tankh_audio_chp/  # Source audio files
│   ├── 01_Genesis/
│   │   ├── 01-Gen_01.mp3
│   │   └── ...
│   └── ...
└── D:/audio/tankh_audio_chp_backup/  # Automatic backups
    ├── 01_Genesis/
    │   ├── 01-Gen_01.mp3
    │   └── ...
    └── ...
```

---

**Document Control**
- **Created**: August 15, 2025
- **Last Modified**: August 15, 2025  
- **Version**: 1.0
- **Status**: As-Built
- **Next Review**: September 15, 2025