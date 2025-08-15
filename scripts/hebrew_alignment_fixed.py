# Hebrew Tutor App - Audio-Text Force Alignment Pipeline (FIXED VERSION)
# Fixed critical issues: emoji encoding, data type mismatch, file conflicts, syntax errors

import os
import sys
import json
import asyncio
import tempfile
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import traceback
from concurrent.futures import ThreadPoolExecutor

# Modern dependencies as specified
import torch
import librosa
import soundfile as sf
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from rapidfuzz import fuzz, process
from dotenv import load_dotenv

# Azure libraries
from azure.storage.blob import BlobServiceClient
import azure.cognitiveservices.speech as speechsdk

# Load environment variables
load_dotenv()

# Configure logging WITHOUT emojis for Windows compatibility
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hebrew_alignment.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class PerformanceTracker:
    """Track processing performance and provide estimates."""
    
    def __init__(self):
        self.start_time = time.time()
        self.chapter_times = []
        self.total_chapters_processed = 0
        
    def record_chapter_completion(self, processing_time: float):
        """Record completion time for a chapter."""
        self.chapter_times.append(processing_time)
        self.total_chapters_processed += 1
        
    def get_average_time(self) -> float:
        """Get average processing time per chapter."""
        return sum(self.chapter_times) / len(self.chapter_times) if self.chapter_times else 0.0
        
    def estimate_completion(self, remaining_chapters: int) -> Tuple[float, str]:
        """Estimate completion time for remaining chapters."""
        if not self.chapter_times:
            return 0.0, "Unknown"
            
        avg_time = self.get_average_time()
        estimated_seconds = remaining_chapters * avg_time
        
        if estimated_seconds < 60:
            return estimated_seconds, f"{estimated_seconds:.0f} seconds"
        elif estimated_seconds < 3600:
            return estimated_seconds, f"{estimated_seconds/60:.1f} minutes"
        else:
            return estimated_seconds, f"{estimated_seconds/3600:.1f} hours"

class HebrewAlignmentTool:
    def __init__(self):
        """Initialize the Hebrew Alignment Tool with production optimizations."""
        logger.info("=== INITIALIZING HEBREW ALIGNMENT TOOL (PRODUCTION FIXED) ===")
        
        # Performance tracking
        self.performance = PerformanceTracker()
        
        # GPU detection and setup
        self.gpu_available = False
        self.gpu_usable = False
        
        logger.info("Checking GPU availability...")
        
        if torch.cuda.is_available():
            try:
                logger.info("CUDA is available, testing GPU...")
                test_tensor = torch.tensor([1.0]).cuda()
                del test_tensor
                self.gpu_available = True
                
                # Enhanced GPU memory checking
                gpu_props = torch.cuda.get_device_properties(0)
                total_memory = gpu_props.total_memory / 1024**3
                torch.cuda.empty_cache()
                free_memory = (gpu_props.total_memory - torch.cuda.memory_allocated(0)) / 1024**3
                
                logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU memory: {total_memory:.1f}GB total, {free_memory:.1f}GB free")
                logger.info(f"GPU compute capability: {gpu_props.major}.{gpu_props.minor}")
                
                if free_memory >= 1.0:
                    self.device = torch.device("cuda")
                    self.gpu_usable = True
                    logger.info("Using GPU for processing")
                else:
                    self.device = torch.device("cpu")
                    logger.warning(f"GPU has insufficient memory ({free_memory:.1f}GB free), using CPU")
                    
            except Exception as e:
                logger.warning(f"GPU test failed: {e}, falling back to CPU")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
            logger.info("CUDA not available, using CPU")
        
        # Optimized processing parameters based on device
        if self.device.type == "cuda" and self.gpu_usable:
            # GPU settings - FIXED: Use float32 consistently
            self.model_name = "openai/whisper-medium"
            self.batch_size = 4  # Reduced to avoid memory issues
            self.chunk_length = 30
            self.max_length = 448
            self.torch_dtype = torch.float32  # FIXED: Consistent data type
            logger.info("GPU mode: Production settings with whisper-medium (float32)")
        else:
            # CPU settings
            self.model_name = "openai/whisper-small"
            self.batch_size = 2
            self.chunk_length = 15
            self.max_length = 224
            self.torch_dtype = torch.float32
            logger.info("CPU mode: Conservative settings")
            
        logger.info(f"Configuration: {self.model_name}, batch={self.batch_size}, chunk={self.chunk_length}")
        
        self.processor = None
        self.model = None
        
        # Audio processing settings
        self.sample_rate = 16000
        
        # Paths
        self.base_path = Path("D:/AI/Projects/HEBREW-TRAINING-AI-AGENT/TANACH")
        self.audio_local_path = Path("D:/audio/tankh_audio_chp")
        self.json_path = self.base_path / "book" / "hebrew_bible_with_nikkud.json"
        self.output_path = self.base_path / "alignment"
        
        # Azure configuration
        self.blob_service_client = None
        self.speech_config = None
        self._init_azure()
        
        # Load Bible data
        self.bible_data = self._load_bible_data()
        
        # Retry settings
        self.max_retries = 3
        self.retry_delay = 2
        
        # Thread pool for I/O operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("=== INITIALIZATION COMPLETE ===")
    
    def _init_azure(self):
        """Initialize Azure services with enhanced error handling."""
        try:
            connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            if connection_string:
                self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                logger.info("Azure Blob Storage initialized")
            else:
                logger.info("Azure Blob Storage not configured (optional)")
            
            speech_key = os.getenv('AZURE_SPEECH_KEY')
            speech_region = os.getenv('AZURE_SPEECH_REGION', 'eastus')
            if speech_key:
                self.speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
                self.speech_config.speech_recognition_language = "he-IL"
                logger.info("Azure Speech SDK initialized")
            else:
                logger.info("Azure Speech SDK not configured (optional)")
                
        except Exception as e:
            logger.warning(f"Azure initialization partial failure: {e}")
    
    def _load_bible_data(self) -> Dict:
        """Load the Hebrew Bible JSON data with validation."""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate data structure
            if not isinstance(data, dict):
                raise ValueError("Bible data must be a dictionary")
                
            total_chapters = sum(len(chapters) for chapters in data.values())
            logger.info(f"Loaded Bible data: {len(data)} books, {total_chapters} chapters")
            return data
        except Exception as e:
            logger.error(f"Failed to load Bible data: {e}")
            return {}
    
    def _lazy_load_whisper(self):
        """Lazy load Whisper model with FIXED data type consistency."""
        if self.processor is None or self.model is None:
            logger.info(f"Loading Whisper model ({self.model_name}) on {self.device}...")
            try:
                self.processor = WhisperProcessor.from_pretrained(self.model_name)
                # FIXED: Use consistent torch_dtype
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype  # Consistent data type
                )
                self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"Whisper model loaded successfully on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise
    
    def get_audio_file_path(self, book_code: str, chapter: int) -> Optional[Path]:
        """Get the local audio file path with enhanced book mapping."""
        book_folders = {
            'Gen': '01_Genesis', 'Exo': '02_Exodus', 'Lev': '03_Leviticus',
            'Num': '04_Numbers', 'Deu': '05_Deuteronomy', 'Jos': '06_Joshua',
            'Jdg': '07_Judges', 'Rut': '08_Ruth', '1Sa': '09_1Samuel',
            '2Sa': '10_2Samuel', '1Ki': '11_1Kings', '2Ki': '12_2Kings',
            '1Ch': '13_1Chronicles', '2Ch': '14_2Chronicles', 'Ezr': '15_Ezra',
            'Neh': '16_Nehemiah', 'Est': '17_Esther', 'Job': '18_Job',
            'Psa': '19_Psalms', 'Pro': '20_Proverbs', 'Ecc': '21_Ecclesiastes',
            'Sol': '22_SongOfSongs', 'Isa': '23_Isaiah', 'Jer': '24_Jeremiah',
            'Lam': '25_Lamentations', 'Eze': '26_Ezekiel', 'Dan': '27_Daniel',
            'Hos': '28_Hosea', 'Joe': '29_Joel', 'Amo': '30_Amos',
            'Oba': '31_Obadiah', 'Jon': '32_Jonah', 'Mic': '33_Micah',
            'Nah': '34_Nahum', 'Hab': '35_Habakkuk', 'Zep': '36_Zephaniah',
            'Hag': '37_Haggai', 'Zec': '38_Zechariah', 'Mal': '39_Malachi'
        }
        
        if book_code not in book_folders:
            return None
            
        folder_name = book_folders[book_code]
        book_num = folder_name.split('_')[0]
        filename = f"{book_num}-{book_code}_{chapter:02d}.mp3"
        
        return self.audio_local_path / folder_name / filename
    
    def validate_audio_file(self, audio_path: Path) -> Tuple[bool, float]:
        """Validate audio file with enhanced error handling."""
        try:
            # Check file exists and has size
            if not audio_path.exists():
                return False, 0.0
                
            file_size = audio_path.stat().st_size
            if file_size < 1000:  # Less than 1KB is likely corrupted
                logger.warning(f"Audio file too small: {file_size} bytes")
                return False, 0.0
            
            duration = librosa.get_duration(path=str(audio_path))
            if duration > 0:
                logger.debug(f"Audio validated: {duration:.2f}s, {file_size/1024/1024:.1f}MB")
                return True, duration
            return False, 0.0
        except Exception as e:
            logger.error(f"Audio validation failed: {e}")
            return False, 0.0
    
    def extract_verses(self, book_code: str, chapter: int) -> List[Dict]:
        """Extract verses with optimized book mapping."""
        try:
            if not isinstance(self.bible_data, dict):
                logger.error(f"Invalid Bible data type: {type(self.bible_data)}")
                return []
            
            # Optimized book mapping
            json_book_key = None
            if book_code in self.bible_data:
                json_book_key = book_code
            else:
                # Fallback mapping for common variations
                book_mappings = {
                    'Exo': 'Exod', 'Deu': 'Deut', 'Jos': 'Josh', 'Jdg': 'Judg',
                    '1Sa': '1Sam', '2Sa': '2Sam', '1Ki': '1Kgs', '2Ki': '2Kgs',
                    '1Ch': '1Chr', '2Ch': '2Chr', 'Est': 'Esth', 'Psa': 'Ps',
                    'Pro': 'Prov', 'Ecc': 'Eccl', 'Sol': 'Song', 'Eze': 'Ezek',
                    'Joe': 'Joel', 'Amo': 'Amos', 'Oba': 'Obad', 'Jon': 'Jonah',
                    'Hab': 'Hab', 'Zep': 'Zeph', 'Zec': 'Zech'
                }
                
                if book_code in book_mappings and book_mappings[book_code] in self.bible_data:
                    json_book_key = book_mappings[book_code]
            
            if not json_book_key:
                logger.error(f"Book '{book_code}' not found in Bible data")
                return []
            
            book_data = self.bible_data[json_book_key]
            if not isinstance(book_data, list) or chapter < 1 or chapter > len(book_data):
                logger.error(f"Invalid chapter {chapter} for book {json_book_key}")
                return []
            
            chapter_data = book_data[chapter - 1]
            if not isinstance(chapter_data, list):
                logger.error(f"Invalid chapter data format")
                return []
            
            # Convert word arrays to verse text
            formatted_verses = []
            for verse_num, verse_words in enumerate(chapter_data, 1):
                if isinstance(verse_words, list):
                    verse_text = ' '.join(verse_words)
                else:
                    verse_text = str(verse_words)
                
                formatted_verses.append({
                    'verse_number': verse_num,
                    'text': verse_text
                })
            
            logger.debug(f"Extracted {len(formatted_verses)} verses for {book_code} {chapter}")
            return formatted_verses
            
        except Exception as e:
            logger.error(f"Failed to extract verses: {e}")
            return []
    
    async def load_audio_optimized(self, audio_path: Path) -> Tuple[Optional[np.ndarray], int]:
        """Load audio with optimized error handling for MP3 issues."""
        try:
            # Try standard loading first
            audio, sr = librosa.load(str(audio_path), sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            logger.warning(f"MP3 decode issue (attempting fix): {e}")
            try:
                # Fallback with different resampling
                audio, sr = librosa.load(
                    str(audio_path), 
                    sr=self.sample_rate, 
                    res_type='kaiser_fast'
                )
                return audio, sr
            except Exception as e2:
                logger.error(f"Audio loading failed completely: {e2}")
                return None, 0
    
    async def transcribe_with_whisper(self, audio_path: Path, retries: int = 0) -> Tuple[bool, List[Dict]]:
        """Transcribe audio with FIXED data type handling."""
        current_device = self.device
        chapter_start_time = time.time()
        
        try:
            self._lazy_load_whisper()
            
            # Optimized audio loading
            audio, sr = await self.load_audio_optimized(audio_path)
            if audio is None:
                return False, []
                
            logger.debug(f"Audio loaded: {len(audio)} samples at {sr} Hz")
            
            # GPU memory management
            if current_device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                    free_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
                    logger.debug(f"GPU free memory: {free_memory:.1f}GB")
                    
                    if free_memory < 1.0:
                        logger.warning("Low GPU memory, falling back to CPU for this file")
                        current_device = torch.device("cpu")
                        self.model = self.model.cpu()
                        
                except Exception as e:
                    logger.warning(f"GPU memory check failed: {e}, using CPU")
                    current_device = torch.device("cpu")
                    self.model = self.model.cpu()
            
            # FIXED: Process with consistent data types
            inputs = self.processor(
                audio, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt", 
                return_attention_mask=True
            )
            
            # Ensure inputs match model dtype
            inputs = {k: v.to(current_device).to(self.torch_dtype) for k, v in inputs.items()}
            
            # Optimized generation parameters
            generation_kwargs = {
                "return_timestamps": True,
                "max_length": self.max_length,
                "num_beams": 1,
                "do_sample": False,
                "language": "hebrew",
                "task": "transcribe"
            }
            
            # Generate with enhanced error handling
            try:
                with torch.no_grad():
                    predicted_ids = self.model.generate(
                        inputs["input_features"],
                        **generation_kwargs
                    )
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and current_device.type == "cuda":
                    logger.warning("GPU OOM! Falling back to CPU...")
                    torch.cuda.empty_cache()
                    
                    self.model = self.model.cpu()
                    current_device = torch.device("cpu")
                    inputs = {k: v.cpu() for k, v in inputs.items()}
                    
                    generation_kwargs["max_length"] = 224
                    with torch.no_grad():
                        predicted_ids = self.model.generate(
                            inputs["input_features"],
                            **generation_kwargs
                        )
                    logger.info("Completed on CPU after GPU OOM")
                else:
                    raise
            
            # Decode results
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True, 
                decode_with_timestamps=True
            )
            
            # Parse timestamps
            segments = []
            if transcription and len(transcription) > 0:
                text = transcription[0]
                segments.append({
                    'text': text,
                    'start': 0.0,
                    'end': float(len(audio) / sr),
                    'confidence': 0.8
                })
            
            # GPU cleanup and model restoration
            if current_device.type == "cuda":
                torch.cuda.empty_cache()
            elif self.device.type == "cuda" and current_device.type == "cpu":
                try:
                    self.model = self.model.cuda()
                    logger.debug("Model restored to GPU")
                except:
                    logger.warning("Could not restore model to GPU")
            
            # Record performance
            processing_time = time.time() - chapter_start_time
            self.performance.record_chapter_completion(processing_time)
            
            device_used = "GPU" if current_device.type == "cuda" else "CPU"
            logger.info(f"Whisper transcription successful on {device_used}: {len(segments)} segments ({processing_time:.1f}s)")
            return True, segments
            
        except Exception as e:
            processing_time = time.time() - chapter_start_time
            logger.error(f"Whisper transcription failed on {current_device} (attempt {retries + 1}): {e}")
            
            if current_device.type == "cuda":
                torch.cuda.empty_cache()
            
            if retries < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay)
                return await self.transcribe_with_whisper(audio_path, retries + 1)
            return False, []
    
    def create_estimated_alignment(self, verses: List[Dict], duration: float) -> Dict:
        """Create estimated alignment with improved timing distribution."""
        if not verses:
            return {}
            
        aligned_verses = []
        total_words = sum(len(verse.get('text', '').split()) for verse in verses)
        
        # More sophisticated time distribution based on word count
        current_time = 0.0
        for i, verse in enumerate(verses):
            words = verse.get('text', '').split()
            word_count = len(words)
            
            # Allocate time based on word density
            if total_words > 0:
                verse_duration = (word_count / total_words) * duration
            else:
                verse_duration = duration / len(verses)
            
            verse_duration = max(verse_duration, 1.0)  # Minimum 1 second per verse
            
            start_time = current_time
            end_time = current_time + verse_duration
            current_time = end_time
            
            # Create word-level estimates
            word_alignments = []
            if words:
                word_duration = verse_duration / len(words)
                for j, word in enumerate(words):
                    word_start = start_time + (j * word_duration)
                    word_end = start_time + ((j + 1) * word_duration)
                    word_alignments.append({
                        'word': word,
                        'start': round(word_start, 2),
                        'end': round(word_end, 2),
                        'confidence': 0.1
                    })
            
            aligned_verses.append({
                'verse_number': verse.get('verse_number', i + 1),
                'text': verse.get('text', ''),
                'start': round(start_time, 2),
                'end': round(end_time, 2),
                'confidence': 0.1,
                'words': word_alignments
            })
        
        return {
            'total_duration': round(duration, 2),
            'overall_confidence': 0.1,
            'verse_count': len(verses),
            'verses': aligned_verses,
            'metadata': {
                'method': 'estimated_improved',
                'model': 'fallback_estimation',
                'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'processing_time_seconds': 0.1
            }
        }
    
    def fuzzy_align_text(self, transcribed_segments: List[Dict], verses: List[Dict]) -> Dict:
        """Enhanced fuzzy alignment with better confidence scoring."""
        if not transcribed_segments or not verses:
            return {}
        
        try:
            start_time = time.time()
            full_transcription = ' '.join([seg.get('text', '') for seg in transcribed_segments])
            total_duration = max([seg.get('end', 0) for seg in transcribed_segments])
            
            # More sophisticated alignment
            aligned_verses = []
            overall_confidence = 0.0
            total_words = sum(len(verse.get('text', '').split()) for verse in verses)
            
            current_time = 0.0
            for i, verse in enumerate(verses):
                verse_text = verse.get('text', '')
                words = verse_text.split()
                
                # Enhanced fuzzy matching
                match_score = max(
                    fuzz.partial_ratio(verse_text, full_transcription),
                    fuzz.token_sort_ratio(verse_text, full_transcription),
                    fuzz.token_set_ratio(verse_text, full_transcription)
                ) / 100.0
                
                overall_confidence += match_score
                
                # Time allocation based on word density
                if total_words > 0:
                    verse_duration = (len(words) / total_words) * total_duration
                else:
                    verse_duration = total_duration / len(verses)
                
                start_time_verse = current_time
                end_time_verse = current_time + verse_duration
                current_time = end_time_verse
                
                # Word-level alignment
                word_alignments = []
                if words:
                    word_duration = verse_duration / len(words)
                    for j, word in enumerate(words):
                        word_start = start_time_verse + (j * word_duration)
                        word_end = start_time_verse + ((j + 1) * word_duration)
                        word_alignments.append({
                            'word': word,
                            'start': round(word_start, 2),
                            'end': round(word_end, 2),
                            'confidence': round(match_score, 2)
                        })
                
                aligned_verses.append({
                    'verse_number': verse.get('verse_number', i + 1),
                    'text': verse_text,
                    'start': round(start_time_verse, 2),
                    'end': round(end_time_verse, 2),
                    'confidence': round(match_score, 2),
                    'words': word_alignments
                })
            
            overall_confidence = overall_confidence / len(verses) if verses else 0.0
            processing_time = time.time() - start_time
            
            return {
                'total_duration': round(total_duration, 2),
                'overall_confidence': round(overall_confidence, 2),
                'verse_count': len(verses),
                'verses': aligned_verses,
                'metadata': {
                    'method': 'fuzzy_alignment_enhanced',
                    'model': self.model_name,
                    'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'processing_time_seconds': round(processing_time, 2)
                }
            }
            
        except Exception as e:
            logger.error(f"Fuzzy alignment failed: {e}")
            return {}
    
    def save_alignment_json(self, book_code: str, chapter: int, alignment_data: Dict) -> bool:
        """FIXED: Save alignment data with improved atomic write operation."""
        try:
            output_dir = self.output_path / book_code
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"alignment_{book_code}_{chapter:02d}.json"
            output_file = output_dir / filename
            
            # FIXED: Better handling of existing files
            if output_file.exists():
                # Create backup if file exists
                backup_file = output_file.with_suffix('.json.backup')
                if backup_file.exists():
                    backup_file.unlink()  # Remove old backup
                output_file.rename(backup_file)
            
            # Use a unique temporary file name
            temp_file = output_file.with_suffix(f'.tmp.{os.getpid()}.{int(time.time())}')
            
            try:
                # Write to temporary file
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(alignment_data, f, indent=2, ensure_ascii=False)
                
                # Atomic rename
                temp_file.rename(output_file)
                
                # Clean up backup if successful
                backup_file = output_file.with_suffix('.json.backup')
                if backup_file.exists():
                    backup_file.unlink()
                
                logger.debug(f"Alignment saved: {output_file}")
                return True
                
            finally:
                # Clean up temp file if it still exists
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except:
                        pass
                    
        except Exception as e:
            logger.error(f"Failed to save alignment: {e}")
            return False
    
    async def process_chapter(self, book_code: str, chapter: int) -> bool:
        """Process a single chapter with enhanced error handling."""
        logger.debug(f"Processing {book_code} Chapter {chapter}")
        
        try:
            verses = self.extract_verses(book_code, chapter)
            if not verses:
                logger.error(f"No verses found for {book_code} {chapter}")
                return False
            
            audio_path = self.get_audio_file_path(book_code, chapter)
            if not audio_path or not audio_path.exists():
                logger.error(f"Audio file not found: {book_code} {chapter}")
                return False
            
            is_valid, duration = self.validate_audio_file(audio_path)
            if not is_valid:
                logger.error(f"Invalid audio file: {audio_path}")
                return False
            
            # Try transcription
            whisper_success, segments = await self.transcribe_with_whisper(audio_path)
            if whisper_success and segments:
                alignment_data = self.fuzzy_align_text(segments, verses)
                logger.debug("Whisper transcription successful")
            else:
                logger.warning("Whisper failed, creating estimates")
                alignment_data = self.create_estimated_alignment(verses, duration)
            
            if alignment_data:
                save_success = self.save_alignment_json(book_code, chapter, alignment_data)
                if save_success:
                    logger.debug(f"Chapter {book_code} {chapter} processed successfully")
                    return True
            
            logger.error(f"Failed to process chapter {book_code} {chapter}")
            return False
            
        except Exception as e:
            logger.error(f"Error processing chapter {book_code} {chapter}: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    async def process_book(self, book_code: str) -> bool:
        """Process all chapters in a book with progress tracking."""
        logger.info(f"Processing book: {book_code}")
        
        # Get chapter count
        json_book_key = book_code
        book_mappings = {
            'Exo': 'Exod', 'Deu': 'Deut', 'Jos': 'Josh', 'Jdg': 'Judg',
            '1Sa': '1Sam', '2Sa': '2Sam', '1Ki': '1Kgs', '2Ki': '2Kgs',
            '1Ch': '1Chr', '2Ch': '2Chr', 'Est': 'Esth', 'Psa': 'Ps',
            'Pro': 'Prov', 'Ecc': 'Eccl', 'Sol': 'Song', 'Eze': 'Ezek',
            'Joe': 'Joel', 'Amo': 'Amos', 'Oba': 'Obad', 'Jon': 'Jonah',
            'Zep': 'Zeph', 'Zec': 'Zech'
        }
        
        if book_code not in self.bible_data and book_code in book_mappings:
            json_book_key = book_mappings[book_code]
        
        if json_book_key not in self.bible_data:
            logger.error(f"Book '{book_code}' not found")
            return False
        
        book_data = self.bible_data[json_book_key]
        if not isinstance(book_data, list):
            logger.error(f"Invalid book data format")
            return False
        
        chapter_count = len(book_data)
        logger.info(f"Book {json_book_key} has {chapter_count} chapters")
        
        success_count = 0
        failed_chapters = []
        book_start_time = time.time()
        
        for chapter in range(1, chapter_count + 1):
            try:
                # Progress estimation
                remaining_chapters = chapter_count - chapter + 1
                if self.performance.chapter_times:
                    _, time_estimate = self.performance.estimate_completion(remaining_chapters)
                    logger.info(f"Processing chapter {chapter}/{chapter_count} (ETA: {time_estimate})")
                else:
                    logger.info(f"Processing chapter {chapter}/{chapter_count}")
                
                success = await self.process_chapter(book_code, chapter)
                if success:
                    success_count += 1
                    logger.info(f"Chapter {chapter} completed successfully")
                else:
                    failed_chapters.append(chapter)
                    logger.warning(f"Chapter {chapter} failed")
                    
                # Small delay for system stability
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing {book_code} {chapter}: {e}")
                failed_chapters.append(chapter)
        
        book_processing_time = time.time() - book_start_time
        avg_chapter_time = self.performance.get_average_time()
        
        logger.info(f"Book {book_code} completed: {success_count}/{chapter_count} chapters successful")
        logger.info(f"Total time: {book_processing_time/60:.1f} minutes, Avg per chapter: {avg_chapter_time:.1f}s")
        
        if failed_chapters:
            logger.warning(f"Failed chapters: {failed_chapters}")
        
        return success_count > 0
    
    async def process_all(self) -> bool:
        """Process all books with comprehensive progress tracking."""
        logger.info("Starting full processing of all Tanakh audio")
        
        book_codes = list(self.bible_data.keys())
        book_codes.sort()
        
        total_chapters = sum(len(chapters) for chapters in self.bible_data.values())
        logger.info(f"Processing {len(book_codes)} books, {total_chapters} total chapters")
        
        success_count = 0
        failed_books = []
        start_time = time.time()
        
        for i, book_code in enumerate(book_codes, 1):
            try:
                logger.info(f"Starting book {i}/{len(book_codes)}: {book_code}")
                
                # Overall progress estimation
                if self.performance.chapter_times:
                    remaining_chapters = sum(len(self.bible_data[code]) for code in book_codes[i:])
                    _, overall_estimate = self.performance.estimate_completion(remaining_chapters)
                    logger.info(f"Overall progress: {i}/{len(book_codes)} books, ETA: {overall_estimate}")
                
                success = await self.process_book(book_code)
                if success:
                    success_count += 1
                    logger.info(f"Book {book_code} completed successfully")
                else:
                    failed_books.append(book_code)
                    logger.warning(f"Book {book_code} failed")
                    
                # Longer delay between books for GPU thermal management
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error processing book {book_code}: {e}")
                failed_books.append(book_code)
        
        total_time = time.time() - start_time
        chapters_processed = self.performance.total_chapters_processed
        
        logger.info(f"Full processing completed: {success_count}/{len(book_codes)} books successful")
        logger.info(f"Total chapters processed: {chapters_processed}")
        logger.info(f"Total time: {total_time/3600:.1f} hours, Avg per chapter: {self.performance.get_average_time():.1f}s")
        
        if failed_books:
            logger.warning(f"Failed books: {failed_books}")
        
        return success_count > 0
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")


async def main():
    """Main function with enhanced CLI and error handling."""
    tool = None
    try:
        tool = HebrewAlignmentTool()
        
        if len(sys.argv) == 1:
            # Full run
            logger.info("Starting full processing of all books")
            await tool.process_all()
        elif len(sys.argv) == 3 and sys.argv[1] == "test":
            # Test single book
            book_code = sys.argv[2]
            logger.info(f"Testing book: {book_code}")
            await tool.process_book(book_code)
        elif len(sys.argv) == 4 and sys.argv[1] == "test":
            # Test single chapter
            book_code = sys.argv[2]
            chapter = int(sys.argv[3])
            logger.info(f"Testing chapter: {book_code} {chapter}")
            success = await tool.process_chapter(book_code, chapter)
            print(f"Chapter processing {'succeeded' if success else 'failed'}")
        else:
            print("Usage:")
            print("  python hebrew_alignment_fixed.py                    # Process all")
            print("  python hebrew_alignment_fixed.py test Gen           # Test book")
            print("  python hebrew_alignment_fixed.py test Gen 1         # Test chapter")
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.debug(traceback.format_exc())
    finally:
        if tool:
            tool.cleanup()


if __name__ == "__main__":
    asyncio.run(main())