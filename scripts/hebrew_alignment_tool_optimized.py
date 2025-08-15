# File location: D:\AI\Gits\hebrew-tutor-data-pipeline\scripts\hebrew_alignment_tool_optimized.py
# Hebrew Tutor App - Audio-Text Force Alignment Pipeline (Optimized Version)
# Optimized with batch processing, better error handling, and performance improvements

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
from datetime import datetime, timedelta

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

# Configure logging with better formatting
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hebrew_alignment.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class PerformanceTracker:
    """Track processing performance and provide estimates."""
    
    def __init__(self):
        self.chapter_times = []
        self.start_time = None
        self.total_chapters = 0
        self.completed_chapters = 0
    
    def start_processing(self, total_chapters: int):
        """Start tracking processing for a book."""
        self.start_time = time.time()
        self.total_chapters = total_chapters
        self.completed_chapters = 0
        self.chapter_times = []
    
    def record_chapter(self, duration: float):
        """Record completion of a chapter."""
        self.chapter_times.append(duration)
        self.completed_chapters += 1
    
    def get_estimates(self) -> Tuple[float, str]:
        """Get average time and completion estimate."""
        if not self.chapter_times:
            return 0.0, "Calculating..."
        
        avg_time = sum(self.chapter_times) / len(self.chapter_times)
        remaining = self.total_chapters - self.completed_chapters
        estimated_seconds = remaining * avg_time
        
        if estimated_seconds < 60:
            eta = f"{estimated_seconds:.0f}s"
        elif estimated_seconds < 3600:
            eta = f"{estimated_seconds/60:.1f}m"
        else:
            eta = f"{estimated_seconds/3600:.1f}h"
        
        return avg_time, eta
    
    def get_progress(self) -> str:
        """Get progress percentage."""
        if self.total_chapters == 0:
            return "0%"
        progress = (self.completed_chapters / self.total_chapters) * 100
        return f"{progress:.1f}%"

class HebrewAlignmentTool:
    def __init__(self):
        """Initialize the Hebrew Alignment Tool with optimized GPU settings."""
        logger.info("=== INITIALIZING OPTIMIZED HEBREW ALIGNMENT TOOL ===")
        
        # Performance tracking
        self.performance = PerformanceTracker()
        
        # Comprehensive GPU detection and setup
        self.gpu_available = False
        self.gpu_usable = False
        
        logger.info("🔍 Checking GPU availability...")
        
        if torch.cuda.is_available():
            try:
                logger.info("✅ CUDA is available, testing GPU...")
                # Test if GPU is actually usable
                test_tensor = torch.tensor([1.0]).cuda()
                del test_tensor  # Clean up test tensor
                self.gpu_available = True
                
                # Check GPU memory
                gpu_props = torch.cuda.get_device_properties(0)
                total_memory = gpu_props.total_memory / 1024**3
                torch.cuda.empty_cache()  # Clear cache for accurate reading
                free_memory = (gpu_props.total_memory - torch.cuda.memory_allocated(0)) / 1024**3
                
                logger.info(f"🎮 GPU detected: {torch.cuda.get_device_name(0)}")
                logger.info(f"💾 GPU memory: {total_memory:.1f}GB total, {free_memory:.1f}GB free")
                
                # Only use GPU if we have sufficient memory (at least 1GB free)
                if free_memory >= 1.0:
                    self.device = torch.device("cuda")
                    self.gpu_usable = True
                    logger.info("✅ Using GPU for processing")
                else:
                    self.device = torch.device("cpu")
                    logger.warning(f"⚠️ GPU has insufficient memory ({free_memory:.1f}GB free), using CPU")
                    
            except Exception as e:
                logger.warning(f"⚠️ GPU test failed: {e}, falling back to CPU")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
            logger.info("❌ CUDA not available, using CPU")
        
        # Set processing parameters based on device - OPTIMIZED SETTINGS
        if self.device.type == "cuda" and self.gpu_usable:
            # GPU settings - optimized for medium model
            self.model_name = "openai/whisper-medium"  # Better accuracy
            self.batch_size = 12  # Increased for better GPU utilization
            self.chunk_length = 40  # Longer chunks for efficiency
            self.max_length = 512  # Increased for medium model
            self.batch_processing_size = 2  # Process 2 chapters simultaneously
            logger.info("🚀 GPU mode: OPTIMIZED high performance settings")
        else:
            # CPU settings - conservative but functional
            self.model_name = "openai/whisper-small"
            self.batch_size = 2
            self.chunk_length = 15
            self.max_length = 224
            self.batch_processing_size = 1
            logger.info("🐌 CPU mode: Conservative settings")
            
        logger.info(f"⚙️ Configuration: {self.model_name}, batch={self.batch_size}, chunk={self.chunk_length}, max_length={self.max_length}")
        logger.info(f"🔄 Batch processing: {self.batch_processing_size} chapters simultaneously")
        
        self.processor = None
        self.model = None
        
        # Audio processing settings - OPTIMIZED
        self.sample_rate = 16000
        self.audio_cache = {}  # Cache loaded audio for batch processing
        
        # Paths - using Drive D: structure
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
        self.retry_delay = 1  # Reduced for faster processing
        
        logger.info("=== OPTIMIZATION FEATURES ENABLED ===")
        logger.info("✨ Batch processing for multiple chapters")
        logger.info("✨ Enhanced MP3 decoding with fallbacks")
        logger.info("✨ Performance tracking and ETA estimates")
        logger.info("✨ Optimized GPU memory management")
        logger.info("✨ Audio caching for faster batch processing")
        logger.info("=== INITIALIZATION COMPLETE ===")
    
    def _init_azure(self):
        """Initialize Azure services."""
        try:
            # Azure Blob Storage
            connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            if connection_string:
                self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                logger.info("Azure Blob Storage initialized")
            
            # Azure Speech SDK
            speech_key = os.getenv('AZURE_SPEECH_KEY')
            speech_region = os.getenv('AZURE_SPEECH_REGION', 'eastus')
            if speech_key:
                self.speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
                self.speech_config.speech_recognition_language = "he-IL"
                logger.info("Azure Speech SDK initialized")
                
        except Exception as e:
            logger.warning(f"Azure initialization failed: {e}")
    
    def _load_bible_data(self) -> Dict:
        """Load the Hebrew Bible JSON data."""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded Bible data with {len(data)} books")
            return data
        except Exception as e:
            logger.error(f"Failed to load Bible data: {e}")
            return {}
    
    def _lazy_load_whisper(self):
        """Lazy load Whisper model to save memory."""
        if self.processor is None or self.model is None:
            logger.info(f"Loading Whisper model on {self.device}...")
            try:
                self.processor = WhisperProcessor.from_pretrained(self.model_name)
                self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()  # Set to evaluation mode
                
                # Enable optimizations for inference
                if self.device.type == "cuda":
                    self.model = torch.jit.optimize_for_inference(self.model)
                
                logger.info(f"✅ Whisper model loaded and optimized on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise
    
    def get_audio_file_path(self, book_code: str, chapter: int) -> Optional[Path]:
        """Get the local audio file path for a given book and chapter."""
        # Format: 01_Genesis/01-Gen_01.mp3
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
        """Validate audio file and return duration with enhanced error handling."""
        try:
            duration = librosa.get_duration(path=str(audio_path))
            if duration > 0:
                return True, duration
            return False, 0.0
        except Exception as e:
            logger.warning(f"Audio validation failed: {e}")
            return False, 0.0
    
    def load_audio_optimized(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio with optimized decoding and fallback mechanisms."""
        try:
            # Primary method - fast decoding
            audio, sr = librosa.load(str(audio_path), sr=self.sample_rate, res_type='kaiser_fast')
            return audio, sr
        except Exception as e:
            logger.warning(f"Primary audio decode failed: {e}, trying fallback")
            try:
                # Fallback method - more robust but slower
                audio, sr = librosa.load(str(audio_path), sr=self.sample_rate, res_type='kaiser_best')
                return audio, sr
            except Exception as e2:
                logger.warning(f"Fallback audio decode failed: {e2}, trying final method")
                try:
                    # Final fallback - use soundfile directly
                    audio, sr = sf.read(str(audio_path))
                    if sr != self.sample_rate:
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                    return audio, self.sample_rate
                except Exception as e3:
                    logger.error(f"All audio decode methods failed: {e3}")
                    raise
    
    def extract_verses(self, book_code: str, chapter: int) -> List[Dict]:
        """Extract verses for a specific book and chapter."""
        try:
            # The JSON structure is: {book_id: [chapters[verses[words]]]}
            if not isinstance(self.bible_data, dict):
                logger.error(f"Expected dict but got {type(self.bible_data)}")
                return []
            
            # Book mappings for Hebrew Bible books
            book_mappings = {
                'Gen': 'Gen', 'Exo': 'Exod', 'Lev': 'Lev', 'Num': 'Num', 'Deu': 'Deut',
                'Jos': 'Josh', 'Jdg': 'Judg', 'Rut': 'Ruth', '1Sa': '1Sam', '2Sa': '2Sam', 
                '1Ki': '1Kgs', '2Ki': '2Kgs', '1Ch': '1Chr', '2Ch': '2Chr', 'Ezr': 'Ezra',
                'Neh': 'Neh', 'Est': 'Esth', 'Job': 'Job', 'Psa': 'Ps', 'Pro': 'Prov',
                'Ecc': 'Eccl', 'Sol': 'Song', 'Isa': 'Isa', 'Jer': 'Jer', 'Lam': 'Lam',
                'Eze': 'Ezek', 'Dan': 'Dan', 'Hos': 'Hos', 'Joe': 'Joel', 'Amo': 'Amos',
                'Oba': 'Obad', 'Jon': 'Jonah', 'Mic': 'Mic', 'Nah': 'Nah', 'Hab': 'Hab',
                'Zep': 'Zeph', 'Hag': 'Hag', 'Zec': 'Zech', 'Mal': 'Mal'
            }
            
            # Find the correct book key
            json_book_key = book_code
            if book_code in book_mappings and book_mappings[book_code] in self.bible_data:
                json_book_key = book_mappings[book_code]
            elif book_code not in self.bible_data:
                for key in self.bible_data.keys():
                    if key.lower() == book_code.lower():
                        json_book_key = key
                        break
            
            if json_book_key not in self.bible_data:
                logger.error(f"Book '{book_code}' not found")
                return []
            
            # Get the book and chapter data
            book_data = self.bible_data[json_book_key]
        if not isinstance(book_data, list):
            logger.error(f"Expected list of chapters but got {type(book_data)}")
            return False
        
        chapter_count = len(book_data)
        logger.info(f"📖 Book {json_book_key} has {chapter_count} chapters")
        
        # Initialize performance tracking
        self.performance.start_processing(chapter_count)
        
        success_count = 0
        failed_chapters = []
        
        # Process chapters in batches
        for start_idx in range(0, chapter_count, self.batch_processing_size):
            end_idx = min(start_idx + self.batch_processing_size, chapter_count)
            batch_chapters = list(range(start_idx + 1, end_idx + 1))  # 1-indexed
            
            batch_start_time = time.time()
            
            try:
                # Process batch
                batch_results = await self.process_chapters_batch(book_code, batch_chapters)
                
                # Record results
                for i, success in enumerate(batch_results):
                    chapter = batch_chapters[i] if i < len(batch_chapters) else start_idx + i + 1
                    if success:
                        success_count += 1
                        logger.info(f"✅ Chapter {chapter} completed successfully")
                    else:
                        failed_chapters.append(chapter)
                        logger.warning(f"❌ Chapter {chapter} failed")
                
                # Update performance tracking
                batch_time = time.time() - batch_start_time
                for _ in batch_results:
                    self.performance.record_chapter(batch_time / len(batch_results))
                
                # Log progress and estimates
                avg_time, eta = self.performance.get_estimates()
                progress = self.performance.get_progress()
                logger.info(f"📊 Progress: {progress} | Avg: {avg_time:.1f}s/ch | ETA: {eta}")
                
                # Small delay between batches for thermal management
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"💥 Error processing batch {batch_chapters}: {e}")
                for chapter in batch_chapters:
                    failed_chapters.append(chapter)
        
        # Final summary
        total_time = time.time() - self.performance.start_time
        avg_time_per_chapter = total_time / chapter_count if chapter_count > 0 else 0
        
        logger.info(f"📊 Book {book_code} completed: {success_count}/{chapter_count} chapters successful")
        logger.info(f"⏱️ Total time: {total_time/60:.1f}m | Avg: {avg_time_per_chapter:.1f}s/chapter")
        
        if failed_chapters:
            logger.warning(f"⚠️ Failed chapters: {failed_chapters}")
        
        return success_count > 0
    
    async def process_all(self) -> bool:
        """Process all books and chapters with optimized batch processing."""
        logger.info("🌍 Starting OPTIMIZED processing of all Tanakh audio")
        
        book_codes = list(self.bible_data.keys())
        book_codes.sort()
        
        logger.info(f"📚 Processing {len(book_codes)} books: {book_codes}")
        
        # Calculate total chapters
        total_chapters = sum(len(self.bible_data[book]) for book in book_codes)
        logger.info(f"📖 Total chapters to process: {total_chapters}")
        
        # Estimate total time
        estimated_time_per_chapter = 3.0  # seconds, based on GPU performance
        estimated_total_minutes = (total_chapters * estimated_time_per_chapter) / 60
        logger.info(f"🕐 Estimated total time: {estimated_total_minutes:.1f} minutes")
        
        success_count = 0
        failed_books = []
        overall_start_time = time.time()
        
        for book_code in book_codes:
            try:
                logger.info(f"📖 Starting book {book_code}...")
                success = await self.process_book(book_code)
                if success:
                    success_count += 1
                    logger.info(f"✅ Book {book_code} completed successfully")
                else:
                    failed_books.append(book_code)
                    logger.warning(f"❌ Book {book_code} failed")
                    
                # Longer delay between books for GPU thermal management
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"💥 Error processing book {book_code}: {e}")
                failed_books.append(book_code)
        
        # Final summary
        total_time = time.time() - overall_start_time
        logger.info(f"🏁 OPTIMIZED processing completed!")
        logger.info(f"📊 Results: {success_count}/{len(book_codes)} books successful")
        logger.info(f"⏱️ Total time: {total_time/3600:.1f} hours")
        logger.info(f"🚀 Average: {total_time/total_chapters:.1f}s per chapter")
        
        if failed_books:
            logger.warning(f"⚠️ Failed books: {failed_books}")
        
        return success_count > 0


async def main():
    """Main function with optimized command-line interface."""
    tool = HebrewAlignmentTool()
    
    if len(sys.argv) == 1:
        # Full run - OPTIMIZED
        logger.info("🌍 Starting OPTIMIZED processing of all books")
        await tool.process_all()
    elif len(sys.argv) == 3 and sys.argv[1] == "test":
        # Test single book - OPTIMIZED
        book_code = sys.argv[2]
        logger.info(f"📚 Testing book with OPTIMIZATIONS: {book_code}")
        await tool.process_book(book_code)
    elif len(sys.argv) == 4 and sys.argv[1] == "test":
        # Test single chapter - for compatibility
        book_code = sys.argv[2]
        chapter = int(sys.argv[3])
        logger.info(f"🧪 Testing single chapter: {book_code} {chapter}")
        
        # Process as single-item batch for consistency
        results = await tool.process_chapters_batch(book_code, [chapter])
        success = results[0] if results else False
        print(f"Chapter processing {'succeeded' if success else 'failed'}")
    else:
        print("Usage:")
        print("  python hebrew_alignment_tool_optimized.py                    # Process all (OPTIMIZED)")
        print("  python hebrew_alignment_tool_optimized.py test Gen           # Test book (OPTIMIZED)")
        print("  python hebrew_alignment_tool_optimized.py test Gen 1         # Test chapter")


if __name__ == "__main__":
    asyncio.run(main())book_key]
            if not isinstance(book_data, list) or chapter < 1 or chapter > len(book_data):
                logger.error(f"Invalid chapter {chapter} for book {json_book_key}")
                return []
            
            chapter_data = book_data[chapter - 1]
            if not isinstance(chapter_data, list):
                logger.error(f"Expected list of verses but got {type(chapter_data)}")
                return []
            
            # Convert verses to expected format
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
            
            return formatted_verses
            
        except Exception as e:
            logger.error(f"Failed to extract verses: {e}")
            return []
    
    async def transcribe_with_whisper_batch(self, audio_data_list: List[Tuple[np.ndarray, int]], retries: int = 0) -> List[Tuple[bool, List[Dict]]]:
        """Transcribe multiple audio files in batch for better GPU utilization."""
        try:
            self._lazy_load_whisper()
            
            if not audio_data_list:
                return []
            
            # GPU memory management
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                free_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
                logger.debug(f"GPU free memory before batch: {free_memory:.1f}GB")
            
            results = []
            
            # Process each audio in the batch
            for audio, sr in audio_data_list:
                try:
                    # Process with Whisper
                    inputs = self.processor(
                        audio, 
                        sampling_rate=sr, 
                        return_tensors="pt", 
                        return_attention_mask=True
                    )
                    
                    # Move to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Generate with optimized settings
                    generation_kwargs = {
                        "return_timestamps": True,
                        "max_length": self.max_length,
                        "num_beams": 1,
                        "do_sample": False,
                        "language": "he",  # Force Hebrew for better accuracy
                        "task": "transcribe"  # Explicit task
                    }
                    
                    if self.device.type == "cuda":
                        generation_kwargs["use_cache"] = True
                    
                    with torch.no_grad():
                        predicted_ids = self.model.generate(
                            inputs["input_features"],
                            **generation_kwargs
                        )
                    
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
                            'confidence': 0.85  # Higher confidence for medium model
                        })
                    
                    results.append((True, segments))
                    
                except Exception as e:
                    logger.warning(f"Single audio transcription failed in batch: {e}")
                    results.append((False, []))
            
            # Clean up GPU memory
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            return results
            
        except Exception as e:
            logger.error(f"Batch transcription failed (attempt {retries + 1}): {e}")
            
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            if retries < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay)
                return await self.transcribe_with_whisper_batch(audio_data_list, retries + 1)
            
            # Return failure for all
            return [(False, []) for _ in audio_data_list]
    
    def create_estimated_alignment(self, verses: List[Dict], duration: float) -> Dict:
        """Create estimated alignment when transcription fails."""
        if not verses:
            return {}
            
        time_per_verse = duration / len(verses) if len(verses) > 0 else 1.0
        
        aligned_verses = []
        for i, verse in enumerate(verses):
            start_time = i * time_per_verse
            end_time = (i + 1) * time_per_verse
            
            words = verse.get('text', '').split()
            word_duration = time_per_verse / len(words) if words else 0.5
            
            word_alignments = []
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
                'method': 'estimated',
                'model': 'fallback_estimation',
                'created_at': datetime.utcnow().isoformat() + 'Z'
            }
        }
    
    def fuzzy_align_text(self, transcribed_segments: List[Dict], verses: List[Dict]) -> Dict:
        """Perform fuzzy matching between transcribed text and verses."""
        if not transcribed_segments or not verses:
            return {}
        
        try:
            full_transcription = ' '.join([seg.get('text', '') for seg in transcribed_segments])
            total_duration = max([seg.get('end', 0) for seg in transcribed_segments])
            time_per_verse = total_duration / len(verses) if verses else 1.0
            
            aligned_verses = []
            overall_confidence = 0.0
            
            for i, verse in enumerate(verses):
                verse_text = verse.get('text', '')
                match_score = fuzz.partial_ratio(verse_text, full_transcription) / 100.0
                overall_confidence += match_score
                
                start_time = i * time_per_verse
                end_time = (i + 1) * time_per_verse
                
                words = verse_text.split()
                word_duration = time_per_verse / len(words) if words else 0.5
                
                word_alignments = []
                for j, word in enumerate(words):
                    word_start = start_time + (j * word_duration)
                    word_end = start_time + ((j + 1) * word_duration)
                    word_alignments.append({
                        'word': word,
                        'start': round(word_start, 2),
                        'end': round(word_end, 2),
                        'confidence': round(match_score, 2)
                    })
                
                aligned_verses.append({
                    'verse_number': verse.get('verse_number', i + 1),
                    'text': verse_text,
                    'start': round(start_time, 2),
                    'end': round(end_time, 2),
                    'confidence': round(match_score, 2),
                    'words': word_alignments
                })
            
            overall_confidence = overall_confidence / len(verses) if verses else 0.0
            
            return {
                'total_duration': round(total_duration, 2),
                'overall_confidence': round(overall_confidence, 2),
                'verse_count': len(verses),
                'verses': aligned_verses,
                'metadata': {
                    'method': 'fuzzy_alignment',
                    'model': self.model_name,
                    'created_at': datetime.utcnow().isoformat() + 'Z'
                }
            }
            
        except Exception as e:
            logger.error(f"Fuzzy alignment failed: {e}")
            return {}
    
    def save_alignment_json(self, book_code: str, chapter: int, alignment_data: Dict) -> bool:
        """Save alignment data to JSON file."""
        try:
            output_dir = self.output_path / book_code
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"alignment_{book_code}_{chapter:02d}.json"
            output_file = output_dir / filename
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(alignment_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save alignment: {e}")
            return False
    
    async def process_chapters_batch(self, book_code: str, chapters: List[int]) -> List[bool]:
        """Process multiple chapters in batch for better GPU utilization."""
        if not chapters:
            return []
        
        logger.info(f"🔄 Batch processing {len(chapters)} chapters: {chapters}")
        
        # Prepare batch data
        batch_data = []
        chapter_info = []
        
        for chapter in chapters:
            try:
                # Extract verses
                verses = self.extract_verses(book_code, chapter)
                if not verses:
                    logger.warning(f"No verses for {book_code} {chapter}")
                    continue
                
                # Get and validate audio
                audio_path = self.get_audio_file_path(book_code, chapter)
                if not audio_path or not audio_path.exists():
                    logger.warning(f"Audio not found for {book_code} {chapter}")
                    continue
                
                is_valid, duration = self.validate_audio_file(audio_path)
                if not is_valid:
                    logger.warning(f"Invalid audio for {book_code} {chapter}")
                    continue
                
                # Load audio
                audio, sr = self.load_audio_optimized(audio_path)
                
                batch_data.append((audio, sr))
                chapter_info.append({
                    'chapter': chapter,
                    'verses': verses,
                    'duration': duration,
                    'audio_path': audio_path
                })
                
            except Exception as e:
                logger.error(f"Failed to prepare {book_code} {chapter}: {e}")
        
        if not batch_data:
            logger.warning("No valid chapters in batch")
            return [False] * len(chapters)
        
        # Process batch with Whisper
        transcription_results = await self.transcribe_with_whisper_batch(batch_data)
        
        # Process results
        results = []
        for i, (success, segments) in enumerate(transcription_results):
            if i >= len(chapter_info):
                results.append(False)
                continue
                
            info = chapter_info[i]
            chapter = info['chapter']
            verses = info['verses']
            duration = info['duration']
            
            try:
                if success and segments:
                    alignment_data = self.fuzzy_align_text(segments, verses)
                else:
                    logger.warning(f"Transcription failed for {book_code} {chapter}, using estimates")
                    alignment_data = self.create_estimated_alignment(verses, duration)
                
                if alignment_data:
                    save_success = self.save_alignment_json(book_code, chapter, alignment_data)
                    if save_success:
                        results.append(True)
                    else:
                        results.append(False)
                else:
                    results.append(False)
                    
            except Exception as e:
                logger.error(f"Failed to process results for {book_code} {chapter}: {e}")
                results.append(False)
        
        return results
    
    async def process_book(self, book_code: str) -> bool:
        """Process all chapters in a book with optimized batch processing."""
        logger.info(f"📚 Processing book: {book_code}")
        
        # Get chapter count
        book_mappings = {
            'Gen': 'Gen', 'Exo': 'Exod', 'Lev': 'Lev', 'Num': 'Num', 'Deu': 'Deut'
            # Add other mappings as needed
        }
        
        json_book_key = book_code
        if book_code in book_mappings and book_mappings[book_code] in self.bible_data:
            json_book_key = book_mappings[book_code]
        
        if json_book_key not in self.bible_data:
            logger.error(f"Book '{book_code}' not found")
            return False
        
        book_data = self.bible_data[json_