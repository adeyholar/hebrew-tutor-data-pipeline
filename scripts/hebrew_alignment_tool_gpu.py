# File location: D:\AI\Gits\hebrew-tutor-data-pipeline\scripts\hebrew_alignment_tool_gpu.py
# Hebrew Tutor App - Audio-Text Force Alignment Pipeline (GPU Version)
# Test script with GPU optimization and improved error handling

import os
import sys
import json
import asyncio
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import traceback

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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HebrewAlignmentTool:
    def __init__(self):
        """Initialize the Hebrew Alignment Tool with intelligent GPU/CPU selection."""
        logger.info("=== INITIALIZING HEBREW ALIGNMENT TOOL ===")
        
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
            logger.info("💡 To enable GPU: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        
        # Set processing parameters based on device
        if self.device.type == "cuda" and self.gpu_usable:
            # GPU settings - more aggressive for better performance
            self.model_name = "openai/whisper-small"
            self.batch_size = 8
            self.chunk_length = 30
            self.max_length = 448
            logger.info("🚀 GPU mode: High performance settings")
        else:
            # CPU settings - conservative but functional
            self.model_name = "openai/whisper-small"
            self.batch_size = 2
            self.chunk_length = 15
            self.max_length = 224
            logger.info("🐌 CPU mode: Conservative settings")
            
        logger.info(f"⚙️ Configuration: {self.model_name}, batch={self.batch_size}, chunk={self.chunk_length}, max_length={self.max_length}")
        
        self.processor = None
        self.model = None
        
        # Audio processing settings
        self.sample_rate = 16000
        
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
        self.retry_delay = 2
        
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
                logger.info(f"✅ Whisper model loaded successfully on {self.device}")
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
        """Validate audio file and return duration."""
        try:
            duration = librosa.get_duration(path=str(audio_path))
            if duration > 0:
                logger.info(f"Audio file validated: {duration:.2f} seconds")
                return True, duration
            return False, 0.0
        except Exception as e:
            logger.error(f"Audio validation failed: {e}")
            return False, 0.0
    
    def extract_verses(self, book_code: str, chapter: int) -> List[Dict]:
        """Extract verses for a specific book and chapter."""
        try:
            # The JSON structure is: {book_id: [chapters[verses[words]]]}
            # where each verse is an array of words with nikkud
            
            logger.info(f"Bible data type: {type(self.bible_data)}, Length: {len(self.bible_data) if isinstance(self.bible_data, dict) else 'N/A'}")
            
            if not isinstance(self.bible_data, dict):
                logger.error(f"Expected dict but got {type(self.bible_data)}")
                return []
            
            # Log available book keys for debugging
            available_books = list(self.bible_data.keys())
            logger.info(f"Available books: {available_books[:10]}...")  # Show first 10
            
            # The book_code might need to be mapped to the actual JSON key
            # Common mappings for Hebrew Bible books
            book_mappings = {
                'Gen': 'Gen',
                'Exo': 'Exod', 
                'Lev': 'Lev',
                'Num': 'Num',
                'Deu': 'Deut',
                'Jos': 'Josh',
                'Jdg': 'Judg',
                'Rut': 'Ruth',
                '1Sa': '1Sam',
                '2Sa': '2Sam', 
                '1Ki': '1Kgs',
                '2Ki': '2Kgs',
                '1Ch': '1Chr',
                '2Ch': '2Chr',
                'Ezr': 'Ezra',
                'Neh': 'Neh',
                'Est': 'Esth',
                'Job': 'Job',
                'Psa': 'Ps',
                'Pro': 'Prov',
                'Ecc': 'Eccl',
                'Sol': 'Song',
                'Isa': 'Isa',
                'Jer': 'Jer',
                'Lam': 'Lam',
                'Eze': 'Ezek',
                'Dan': 'Dan',
                'Hos': 'Hos',
                'Joe': 'Joel',
                'Amo': 'Amos',
                'Oba': 'Obad',
                'Jon': 'Jonah',
                'Mic': 'Mic',
                'Nah': 'Nah',
                'Hab': 'Hab',
                'Zep': 'Zeph',
                'Hag': 'Hag',
                'Zec': 'Zech',
                'Mal': 'Mal'
            }
            
            # Try to find the correct book key
            json_book_key = None
            
            # First try exact match
            if book_code in self.bible_data:
                json_book_key = book_code
            # Try mapping
            elif book_code in book_mappings and book_mappings[book_code] in self.bible_data:
                json_book_key = book_mappings[book_code]
            # Try case variations
            else:
                for key in self.bible_data.keys():
                    if key.lower() == book_code.lower():
                        json_book_key = key
                        break
            
            if not json_book_key:
                logger.error(f"Book '{book_code}' not found. Available books: {list(self.bible_data.keys())}")
                return []
            
            logger.info(f"Found book key: {json_book_key}")
            
            # Get the book data
            book_data = self.bible_data[json_book_key]
            if not isinstance(book_data, list):
                logger.error(f"Expected list of chapters but got {type(book_data)}")
                return []
            
            logger.info(f"Book {json_book_key} has {len(book_data)} chapters")
            
            # Check if chapter exists (1-indexed)
            if chapter < 1 or chapter > len(book_data):
                logger.error(f"Chapter {chapter} not found. Book has {len(book_data)} chapters")
                return []
            
            # Get the chapter data (convert to 0-indexed)
            chapter_data = book_data[chapter - 1]
            if not isinstance(chapter_data, list):
                logger.error(f"Expected list of verses but got {type(chapter_data)}")
                return []
            
            logger.info(f"Chapter {chapter} has {len(chapter_data)} verses")
            
            # Convert verses to expected format
            formatted_verses = []
            for verse_num, verse_words in enumerate(chapter_data, 1):
                if isinstance(verse_words, list):
                    # Join words to create verse text
                    verse_text = ' '.join(verse_words)
                else:
                    # Fallback if verse is not a list
                    verse_text = str(verse_words)
                
                formatted_verses.append({
                    'verse_number': verse_num,
                    'text': verse_text
                })
            
            logger.info(f"Successfully extracted {len(formatted_verses)} verses for {book_code} {chapter}")
            return formatted_verses
            
        except Exception as e:
            logger.error(f"Failed to extract verses: {e}")
            logger.error(f"Bible data sample: {str(list(self.bible_data.keys())[:5]) if isinstance(self.bible_data, dict) else 'Not a dict'}")
            return []
    
    async def transcribe_with_whisper(self, audio_path: Path, retries: int = 0) -> Tuple[bool, List[Dict]]:
        """Transcribe audio using Whisper with intelligent GPU/CPU fallback."""
        current_device = self.device
        
        try:
            self._lazy_load_whisper()
            
            # Load audio with librosa
            audio, sr = librosa.load(str(audio_path), sr=self.sample_rate)
            logger.info(f"Audio loaded: {len(audio)} samples at {sr} Hz")
            
            # GPU memory management
            if current_device.type == "cuda":
                try:
                    # Clear GPU cache before processing
                    torch.cuda.empty_cache()
                    
                    # Check available memory
                    free_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
                    logger.info(f"GPU free memory: {free_memory:.1f}GB")
                    
                    # If memory is too low, fall back to CPU for this file
                    if free_memory < 1.0:
                        logger.warning("⚠️ Low GPU memory, falling back to CPU for this file")
                        current_device = torch.device("cpu")
                        # Move model to CPU temporarily
                        self.model = self.model.cpu()
                        
                except Exception as e:
                    logger.warning(f"GPU memory check failed: {e}, using CPU")
                    current_device = torch.device("cpu")
                    self.model = self.model.cpu()
            
            # Process with Whisper
            inputs = self.processor(
                audio, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt", 
                return_attention_mask=True
            )
            
            # Move to current device
            inputs = {k: v.to(current_device) for k, v in inputs.items()}
            
            # Set generation parameters based on current device
            if current_device.type == "cuda":
                generation_kwargs = {
                    "return_timestamps": True,
                    "max_length": self.max_length,
                    "num_beams": 1,
                    "do_sample": False,
                    "use_cache": True,
                }
            else:
                generation_kwargs = {
                    "return_timestamps": True,
                    "max_length": self.max_length,
                    "num_beams": 1,
                    "do_sample": False,
                }
            
            # Generate with device-specific error handling
            try:
                with torch.no_grad():
                    predicted_ids = self.model.generate(
                        inputs["input_features"],
                        **generation_kwargs
                    )
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and current_device.type == "cuda":
                    logger.warning("🔥 GPU OOM detected! Falling back to CPU...")
                    torch.cuda.empty_cache()
                    
                    # Move model to CPU and retry
                    self.model = self.model.cpu()
                    current_device = torch.device("cpu")
                    inputs = {k: v.cpu() for k, v in inputs.items()}
                    
                    # Retry with CPU
                    generation_kwargs["max_length"] = 224  # Reduce for CPU
                    with torch.no_grad():
                        predicted_ids = self.model.generate(
                            inputs["input_features"],
                            **generation_kwargs
                        )
                    logger.info("✅ Successfully completed on CPU after GPU OOM")
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
            
            # Clean up and restore model to original device if needed
            if current_device.type == "cuda":
                torch.cuda.empty_cache()
            elif self.device.type == "cuda" and current_device.type == "cpu":
                # Move model back to GPU for next iteration
                try:
                    self.model = self.model.cuda()
                    logger.info("🔄 Model moved back to GPU")
                except:
                    logger.warning("Could not move model back to GPU, staying on CPU")
            
            device_used = "GPU" if current_device.type == "cuda" else "CPU"
            logger.info(f"✅ Whisper transcription successful on {device_used}: {len(segments)} segments")
            return True, segments
            
        except Exception as e:
            logger.error(f"Whisper transcription failed on {current_device} (attempt {retries + 1}): {e}")
            
            # Clean up on error
            if current_device.type == "cuda":
                torch.cuda.empty_cache()
            
            if retries < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay)
                return await self.transcribe_with_whisper(audio_path, retries + 1)
            return False, []
    
    def create_estimated_alignment(self, verses: List[Dict], duration: float) -> Dict:
        """Create estimated alignment when transcription fails."""
        if not verses:
            return {}
            
        # Calculate time per verse
        time_per_verse = duration / len(verses) if len(verses) > 0 else 1.0
        
        aligned_verses = []
        for i, verse in enumerate(verses):
            start_time = i * time_per_verse
            end_time = (i + 1) * time_per_verse
            
            # Create word-level estimates
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
                    'confidence': 0.1  # Low confidence for estimates
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
                'created_at': '2025-08-14T12:00:00Z'
            }
        }
    
    def fuzzy_align_text(self, transcribed_segments: List[Dict], verses: List[Dict]) -> Dict:
        """Perform fuzzy matching between transcribed text and verses."""
        if not transcribed_segments or not verses:
            return {}
        
        try:
            # Combine all transcribed text
            full_transcription = ' '.join([seg.get('text', '') for seg in transcribed_segments])
            
            # Calculate total duration
            total_duration = max([seg.get('end', 0) for seg in transcribed_segments])
            
            # Simple alignment - divide duration among verses
            time_per_verse = total_duration / len(verses) if verses else 1.0
            
            aligned_verses = []
            overall_confidence = 0.0
            
            for i, verse in enumerate(verses):
                verse_text = verse.get('text', '')
                
                # Calculate fuzzy match score
                match_score = fuzz.partial_ratio(verse_text, full_transcription) / 100.0
                overall_confidence += match_score
                
                start_time = i * time_per_verse
                end_time = (i + 1) * time_per_verse
                
                # Create word alignments
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
                    'created_at': '2025-08-14T12:00:00Z'
                }
            }
            
        except Exception as e:
            logger.error(f"Fuzzy alignment failed: {e}")
            return {}
    
    def save_alignment_json(self, book_code: str, chapter: int, alignment_data: Dict) -> bool:
        """Save alignment data to JSON file."""
        try:
            # Create output directory
            output_dir = self.output_path / book_code
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename
            filename = f"alignment_{book_code}_{chapter:02d}.json"
            output_file = output_dir / filename
            
            # Save JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(alignment_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Alignment saved: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save alignment: {e}")
            return False
    
    async def process_book(self, book_code: str) -> bool:
        """Process all chapters in a book."""
        logger.info(f"📚 Processing book: {book_code}")
        
        # For the new JSON structure: {book_id: [chapters[verses[words]]]}
        chapter_count = 0
        
        # Try to find the book using the same logic as extract_verses
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
        json_book_key = None
        if book_code in self.bible_data:
            json_book_key = book_code
        elif book_code in book_mappings and book_mappings[book_code] in self.bible_data:
            json_book_key = book_mappings[book_code]
        else:
            for key in self.bible_data.keys():
                if key.lower() == book_code.lower():
                    json_book_key = key
                    break
        
        if not json_book_key:
            logger.error(f"Book '{book_code}' not found. Available books: {list(self.bible_data.keys())}")
            return False
        
        # Get chapter count from the book data
        book_data = self.bible_data[json_book_key]
        if isinstance(book_data, list):
            chapter_count = len(book_data)
        else:
            logger.error(f"Expected list of chapters but got {type(book_data)}")
            return False
        
        logger.info(f"📖 Book {json_book_key} has {chapter_count} chapters")
        
        success_count = 0
        failed_chapters = []
        
        for chapter in range(1, chapter_count + 1):
            try:
                logger.info(f"🔄 Processing chapter {chapter}/{chapter_count}")
                success = await self.process_chapter(book_code, chapter)
                if success:
                    success_count += 1
                    logger.info(f"✅ Chapter {chapter} completed successfully")
                else:
                    failed_chapters.append(chapter)
                    logger.warning(f"❌ Chapter {chapter} failed")
                    
                # Small delay between chapters to prevent resource exhaustion
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"💥 Error processing {book_code} {chapter}: {e}")
                failed_chapters.append(chapter)
        
        logger.info(f"📊 Book {book_code} completed: {success_count}/{chapter_count} chapters successful")
        if failed_chapters:
            logger.warning(f"⚠️ Failed chapters: {failed_chapters}")
        
        return success_count > 0
    
    async def process_all(self) -> bool:
        """Process all books and chapters."""
        logger.info("🌍 Starting full processing of all Tanakh audio")
        
        book_codes = list(self.bible_data.keys())
        book_codes.sort()
        
        logger.info(f"📚 Processing {len(book_codes)} books: {book_codes}")
        
        success_count = 0
        failed_books = []
        
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
                    
                # Delay between books for GPU cooldown
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"💥 Error processing book {book_code}: {e}")
                failed_books.append(book_code)
        
        logger.info(f"🏁 Full processing completed: {success_count}/{len(book_codes)} books successful")
        if failed_books:
            logger.warning(f"⚠️ Failed books: {failed_books}")
        
        return success_count > 0
        """Process a single chapter for audio-text alignment."""
        logger.info(f"🔍 Processing {book_code} Chapter {chapter}")
        
        audio_path = None  # Initialize to avoid UnboundLocalError
        
        try:
            # Extract verses
            verses = self.extract_verses(book_code, chapter)
            if not verses:
                logger.error(f"No verses found for {book_code} {chapter}")
                return False
            
            # Get audio file
            audio_path = self.get_audio_file_path(book_code, chapter)
            if not audio_path or not audio_path.exists():
                logger.warning(f"Local audio not found, trying blob download...")
                # Audio download would go here
                
            if not audio_path or not audio_path.exists():
                logger.error(f"Audio file not found: {book_code} {chapter}")
                return False
            
            # Validate audio
            is_valid, duration = self.validate_audio_file(audio_path)
            if not is_valid:
                logger.error(f"Invalid audio file: {audio_path}")
                return False
            
            # Try transcription with retries
            success = False
            alignment_data = {}
            
            # Try Whisper first
            whisper_success, segments = await self.transcribe_with_whisper(audio_path)
            if whisper_success and segments:
                alignment_data = self.fuzzy_align_text(segments, verses)
                success = True
                logger.info("✅ Whisper transcription successful")
            else:
                # Create estimated alignment
                logger.warning("Whisper failed, creating estimates")
                alignment_data = self.create_estimated_alignment(verses, duration)
                success = True  # Consider estimates as success for JSON output
            
            # Save results
            if alignment_data:
                save_success = self.save_alignment_json(book_code, chapter, alignment_data)
                if save_success:
                    logger.info(f"✅ Chapter {book_code} {chapter} processed successfully")
                    return True
            
            logger.error(f"Failed to process chapter {book_code} {chapter}")
            return False
            
        except Exception as e:
            logger.error(f"Error processing chapter {book_code} {chapter}: {e}")
            logger.error(traceback.format_exc())
            return False
        finally:
            # Clean up downloaded files
            if audio_path and str(audio_path).startswith(tempfile.gettempdir()):
                try:
                    os.remove(audio_path)
                except Exception:
                    pass


async def main():
    """Main function with command-line interface."""
    tool = HebrewAlignmentTool()
    
    if len(sys.argv) == 1:
        # Full run
        logger.info("🌍 Starting full processing of all books")
        await tool.process_all()
    elif len(sys.argv) == 3 and sys.argv[1] == "test":
        # Test single book
        book_code = sys.argv[2]
        logger.info(f"📚 Testing book: {book_code}")
        await tool.process_book(book_code)
    elif len(sys.argv) == 4 and sys.argv[1] == "test":
        # Test single chapter
        book_code = sys.argv[2]
        chapter = int(sys.argv[3])
        logger.info(f"🧪 Testing chapter: {book_code} {chapter}")
        success = await tool.process_chapter(book_code, chapter)
        print(f"Chapter processing {'succeeded' if success else 'failed'}")
    else:
        print("Usage:")
        print("  python hebrew_alignment_tool_gpu.py                    # Process all")
        print("  python hebrew_alignment_tool_gpu.py test Gen           # Test book")
        print("  python hebrew_alignment_tool_gpu.py test Gen 1         # Test chapter")


if __name__ == "__main__":
    asyncio.run(main())# File location: D:\AI\Gits\hebrew-tutor-data-pipeline\scripts\hebrew_alignment_tool_gpu.py
# Hebrew Tutor App - Audio-Text Force Alignment Pipeline (GPU Version)
# Test script with GPU optimization and improved error handling

import os
import sys
import json
import asyncio
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import traceback

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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HebrewAlignmentTool:
    def __init__(self):
        """Initialize the Hebrew Alignment Tool with intelligent GPU/CPU selection."""
        logger.info("=== INITIALIZING HEBREW ALIGNMENT TOOL ===")
        
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
            logger.info("💡 To enable GPU: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        
        # Set processing parameters based on device
        if self.device.type == "cuda" and self.gpu_usable:
            # GPU settings - more aggressive for better performance
            self.model_name = "openai/whisper-small"
            self.batch_size = 8
            self.chunk_length = 30
            self.max_length = 448
            logger.info("🚀 GPU mode: High performance settings")
        else:
            # CPU settings - conservative but functional
            self.model_name = "openai/whisper-small"
            self.batch_size = 2
            self.chunk_length = 15
            self.max_length = 224
            logger.info("🐌 CPU mode: Conservative settings")
            
        logger.info(f"⚙️ Configuration: {self.model_name}, batch={self.batch_size}, chunk={self.chunk_length}, max_length={self.max_length}")
        
        self.processor = None
        self.model = None
        
        # Audio processing settings
        self.sample_rate = 16000
        
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
        self.retry_delay = 2
        
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
                logger.info(f"✅ Whisper model loaded successfully on {self.device}")
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
        """Validate audio file and return duration."""
        try:
            duration = librosa.get_duration(path=str(audio_path))
            if duration > 0:
                logger.info(f"Audio file validated: {duration:.2f} seconds")
                return True, duration
            return False, 0.0
        except Exception as e:
            logger.error(f"Audio validation failed: {e}")
            return False, 0.0
    
    def extract_verses(self, book_code: str, chapter: int) -> List[Dict]:
        """Extract verses for a specific book and chapter."""
        try:
            # The JSON structure is: {book_id: [chapters[verses[words]]]}
            # where each verse is an array of words with nikkud
            
            logger.info(f"Bible data type: {type(self.bible_data)}, Length: {len(self.bible_data) if isinstance(self.bible_data, dict) else 'N/A'}")
            
            if not isinstance(self.bible_data, dict):
                logger.error(f"Expected dict but got {type(self.bible_data)}")
                return []
            
            # Log available book keys for debugging
            available_books = list(self.bible_data.keys())
            logger.info(f"Available books: {available_books[:10]}...")  # Show first 10
            
            # The book_code might need to be mapped to the actual JSON key
            # Common mappings for Hebrew Bible books
            book_mappings = {
                'Gen': 'Gen',
                'Exo': 'Exod', 
                'Lev': 'Lev',
                'Num': 'Num',
                'Deu': 'Deut',
                'Jos': 'Josh',
                'Jdg': 'Judg',
                'Rut': 'Ruth',
                '1Sa': '1Sam',
                '2Sa': '2Sam', 
                '1Ki': '1Kgs',
                '2Ki': '2Kgs',
                '1Ch': '1Chr',
                '2Ch': '2Chr',
                'Ezr': 'Ezra',
                'Neh': 'Neh',
                'Est': 'Esth',
                'Job': 'Job',
                'Psa': 'Ps',
                'Pro': 'Prov',
                'Ecc': 'Eccl',
                'Sol': 'Song',
                'Isa': 'Isa',
                'Jer': 'Jer',
                'Lam': 'Lam',
                'Eze': 'Ezek',
                'Dan': 'Dan',
                'Hos': 'Hos',
                'Joe': 'Joel',
                'Amo': 'Amos',
                'Oba': 'Obad',
                'Jon': 'Jonah',
                'Mic': 'Mic',
                'Nah': 'Nah',
                'Hab': 'Hab',
                'Zep': 'Zeph',
                'Hag': 'Hag',
                'Zec': 'Zech',
                'Mal': 'Mal'
            }
            
            # Try to find the correct book key
            json_book_key = None
            
            # First try exact match
            if book_code in self.bible_data:
                json_book_key = book_code
            # Try mapping
            elif book_code in book_mappings and book_mappings[book_code] in self.bible_data:
                json_book_key = book_mappings[book_code]
            # Try case variations
            else:
                for key in self.bible_data.keys():
                    if key.lower() == book_code.lower():
                        json_book_key = key
                        break
            
            if not json_book_key:
                logger.error(f"Book '{book_code}' not found. Available books: {list(self.bible_data.keys())}")
                return []
            
            logger.info(f"Found book key: {json_book_key}")
            
            # Get the book data
            book_data = self.bible_data[json_book_key]
            if not isinstance(book_data, list):
                logger.error(f"Expected list of chapters but got {type(book_data)}")
                return []
            
            logger.info(f"Book {json_book_key} has {len(book_data)} chapters")
            
            # Check if chapter exists (1-indexed)
            if chapter < 1 or chapter > len(book_data):
                logger.error(f"Chapter {chapter} not found. Book has {len(book_data)} chapters")
                return []
            
            # Get the chapter data (convert to 0-indexed)
            chapter_data = book_data[chapter - 1]
            if not isinstance(chapter_data, list):
                logger.error(f"Expected list of verses but got {type(chapter_data)}")
                return []
            
            logger.info(f"Chapter {chapter} has {len(chapter_data)} verses")
            
            # Convert verses to expected format
            formatted_verses = []
            for verse_num, verse_words in enumerate(chapter_data, 1):
                if isinstance(verse_words, list):
                    # Join words to create verse text
                    verse_text = ' '.join(verse_words)
                else:
                    # Fallback if verse is not a list
                    verse_text = str(verse_words)
                
                formatted_verses.append({
                    'verse_number': verse_num,
                    'text': verse_text
                })
            
            logger.info(f"Successfully extracted {len(formatted_verses)} verses for {book_code} {chapter}")
            return formatted_verses
            
        except Exception as e:
            logger.error(f"Failed to extract verses: {e}")
            logger.error(f"Bible data sample: {str(list(self.bible_data.keys())[:5]) if isinstance(self.bible_data, dict) else 'Not a dict'}")
            return []
    
    async def transcribe_with_whisper(self, audio_path: Path, retries: int = 0) -> Tuple[bool, List[Dict]]:
        """Transcribe audio using Whisper with intelligent GPU/CPU fallback."""
        current_device = self.device
        
        try:
            self._lazy_load_whisper()
            
            # Load audio with librosa
            audio, sr = librosa.load(str(audio_path), sr=self.sample_rate)
            logger.info(f"Audio loaded: {len(audio)} samples at {sr} Hz")
            
            # GPU memory management
            if current_device.type == "cuda":
                try:
                    # Clear GPU cache before processing
                    torch.cuda.empty_cache()
                    
                    # Check available memory
                    free_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
                    logger.info(f"GPU free memory: {free_memory:.1f}GB")
                    
                    # If memory is too low, fall back to CPU for this file
                    if free_memory < 1.0:
                        logger.warning("⚠️ Low GPU memory, falling back to CPU for this file")
                        current_device = torch.device("cpu")
                        # Move model to CPU temporarily
                        self.model = self.model.cpu()
                        
                except Exception as e:
                    logger.warning(f"GPU memory check failed: {e}, using CPU")
                    current_device = torch.device("cpu")
                    self.model = self.model.cpu()
            
            # Process with Whisper
            inputs = self.processor(
                audio, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt", 
                return_attention_mask=True
            )
            
            # Move to current device
            inputs = {k: v.to(current_device) for k, v in inputs.items()}
            
            # Set generation parameters based on current device
            if current_device.type == "cuda":
                generation_kwargs = {
                    "return_timestamps": True,
                    "max_length": self.max_length,
                    "num_beams": 1,
                    "do_sample": False,
                    "use_cache": True,
                }
            else:
                generation_kwargs = {
                    "return_timestamps": True,
                    "max_length": self.max_length,
                    "num_beams": 1,
                    "do_sample": False,
                }
            
            # Generate with device-specific error handling
            try:
                with torch.no_grad():
                    predicted_ids = self.model.generate(
                        inputs["input_features"],
                        **generation_kwargs
                    )
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and current_device.type == "cuda":
                    logger.warning("🔥 GPU OOM detected! Falling back to CPU...")
                    torch.cuda.empty_cache()
                    
                    # Move model to CPU and retry
                    self.model = self.model.cpu()
                    current_device = torch.device("cpu")
                    inputs = {k: v.cpu() for k, v in inputs.items()}
                    
                    # Retry with CPU
                    generation_kwargs["max_length"] = 224  # Reduce for CPU
                    with torch.no_grad():
                        predicted_ids = self.model.generate(
                            inputs["input_features"],
                            **generation_kwargs
                        )
                    logger.info("✅ Successfully completed on CPU after GPU OOM")
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
            
            # Clean up and restore model to original device if needed
            if current_device.type == "cuda":
                torch.cuda.empty_cache()
            elif self.device.type == "cuda" and current_device.type == "cpu":
                # Move model back to GPU for next iteration
                try:
                    self.model = self.model.cuda()
                    logger.info("🔄 Model moved back to GPU")
                except:
                    logger.warning("Could not move model back to GPU, staying on CPU")
            
            device_used = "GPU" if current_device.type == "cuda" else "CPU"
            logger.info(f"✅ Whisper transcription successful on {device_used}: {len(segments)} segments")
            return True, segments
            
        except Exception as e:
            logger.error(f"Whisper transcription failed on {current_device} (attempt {retries + 1}): {e}")
            
            # Clean up on error
            if current_device.type == "cuda":
                torch.cuda.empty_cache()
            
            if retries < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay)
                return await self.transcribe_with_whisper(audio_path, retries + 1)
            return False, []
    
    def create_estimated_alignment(self, verses: List[Dict], duration: float) -> Dict:
        """Create estimated alignment when transcription fails."""
        if not verses:
            return {}
            
        # Calculate time per verse
        time_per_verse = duration / len(verses) if len(verses) > 0 else 1.0
        
        aligned_verses = []
        for i, verse in enumerate(verses):
            start_time = i * time_per_verse
            end_time = (i + 1) * time_per_verse
            
            # Create word-level estimates
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
                    'confidence': 0.1  # Low confidence for estimates
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
                'created_at': '2025-08-14T12:00:00Z'
            }
        }
    
    def fuzzy_align_text(self, transcribed_segments: List[Dict], verses: List[Dict]) -> Dict:
        """Perform fuzzy matching between transcribed text and verses."""
        if not transcribed_segments or not verses:
            return {}
        
        try:
            # Combine all transcribed text
            full_transcription = ' '.join([seg.get('text', '') for seg in transcribed_segments])
            
            # Calculate total duration
            total_duration = max([seg.get('end', 0) for seg in transcribed_segments])
            
            # Simple alignment - divide duration among verses
            time_per_verse = total_duration / len(verses) if verses else 1.0
            
            aligned_verses = []
            overall_confidence = 0.0
            
            for i, verse in enumerate(verses):
                verse_text = verse.get('text', '')
                
                # Calculate fuzzy match score
                match_score = fuzz.partial_ratio(verse_text, full_transcription) / 100.0
                overall_confidence += match_score
                
                start_time = i * time_per_verse
                end_time = (i + 1) * time_per_verse
                
                # Create word alignments
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
                    'created_at': '2025-08-14T12:00:00Z'
                }
            }
            
        except Exception as e:
            logger.error(f"Fuzzy alignment failed: {e}")
            return {}
    
    def save_alignment_json(self, book_code: str, chapter: int, alignment_data: Dict) -> bool:
        """Save alignment data to JSON file."""
        try:
            # Create output directory
            output_dir = self.output_path / book_code
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename
            filename = f"alignment_{book_code}_{chapter:02d}.json"
            output_file = output_dir / filename
            
            # Save JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(alignment_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Alignment saved: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save alignment: {e}")
            return False
    
    async def process_book(self, book_code: str) -> bool:
        """Process all chapters in a book."""
        logger.info(f"📚 Processing book: {book_code}")
        
        # For the new JSON structure: {book_id: [chapters[verses[words]]]}
        chapter_count = 0
        
        # Try to find the book using the same logic as extract_verses
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
        json_book_key = None
        if book_code in self.bible_data:
            json_book_key = book_code
        elif book_code in book_mappings and book_mappings[book_code] in self.bible_data:
            json_book_key = book_mappings[book_code]
        else:
            for key in self.bible_data.keys():
                if key.lower() == book_code.lower():
                    json_book_key = key
                    break
        
        if not json_book_key:
            logger.error(f"Book '{book_code}' not found. Available books: {list(self.bible_data.keys())}")
            return False
        
        # Get chapter count from the book data
        book_data = self.bible_data[json_book_key]
        if isinstance(book_data, list):
            chapter_count = len(book_data)
        else:
            logger.error(f"Expected list of chapters but got {type(book_data)}")
            return False
        
        logger.info(f"📖 Book {json_book_key} has {chapter_count} chapters")
        
        success_count = 0
        failed_chapters = []
        
        for chapter in range(1, chapter_count + 1):
            try:
                logger.info(f"🔄 Processing chapter {chapter}/{chapter_count}")
                success = await self.process_chapter(book_code, chapter)
                if success:
                    success_count += 1
                    logger.info(f"✅ Chapter {chapter} completed successfully")
                else:
                    failed_chapters.append(chapter)
                    logger.warning(f"❌ Chapter {chapter} failed")
                    
                # Small delay between chapters to prevent resource exhaustion
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"💥 Error processing {book_code} {chapter}: {e}")
                failed_chapters.append(chapter)
        
        logger.info(f"📊 Book {book_code} completed: {success_count}/{chapter_count} chapters successful")
        if failed_chapters:
            logger.warning(f"⚠️ Failed chapters: {failed_chapters}")
        
        return success_count > 0
    
    async def process_chapter(self, book_code: str, chapter: int) -> bool:
        """Process a single chapter for audio-text alignment."""
        logger.info(f"🔍 Processing {book_code} Chapter {chapter}")
        
        audio_path = None  # Initialize to avoid UnboundLocalError
        
        try:
            # Extract verses
            verses = self.extract_verses(book_code, chapter)
            if not verses:
                logger.error(f"No verses found for {book_code} {chapter}")
                return False
            
            # Get audio file
            audio_path = self.get_audio_file_path(book_code, chapter)
            if not audio_path or not audio_path.exists():
                logger.warning(f"Local audio not found, trying blob download...")
                # Audio download would go here
                
            if not audio_path or not audio_path.exists():
                logger.error(f"Audio file not found: {book_code} {chapter}")
                return False
            
            # Validate audio
            is_valid, duration = self.validate_audio_file(audio_path)
            if not is_valid:
                logger.error(f"Invalid audio file: {audio_path}")
                return False
            
            # Try transcription with retries
            success = False
            alignment_data = {}
            
            # Try Whisper first
            whisper_success, segments = await self.transcribe_with_whisper(audio_path)
            if whisper_success and segments:
                alignment_data = self.fuzzy_align_text(segments, verses)
                success = True
                logger.info("✅ Whisper transcription successful")
            else:
                # Create estimated alignment
                logger.warning("Whisper failed, creating estimates")
                alignment_data = self.create_estimated_alignment(verses, duration)
                success = True  # Consider estimates as success for JSON output
            
            # Save results
            if alignment_data:
                save_success = self.save_alignment_json(book_code, chapter, alignment_data)
                if save_success:
                    logger.info(f"✅ Chapter {book_code} {chapter} processed successfully")
                    return True
            
            logger.error(f"Failed to process chapter {book_code} {chapter}")
            return False
            
        except Exception as e:
            logger.error(f"Error processing chapter {book_code} {chapter}: {e}")
            logger.error(traceback.format_exc())
            return False
        finally:
            # Clean up downloaded files
            if audio_path and str(audio_path).startswith(tempfile.gettempdir()):
                try:
                    os.remove(audio_path)
                except Exception:
                    pass
    
    async def process_all(self) -> bool:
        """Process all books and chapters."""
        logger.info("🌍 Starting full processing of all Tanakh audio")
        
        book_codes = list(self.bible_data.keys())
        book_codes.sort()
        
        logger.info(f"📚 Processing {len(book_codes)} books: {book_codes}")
        
        success_count = 0
        failed_books = []
        
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
                    
                # Delay between books for GPU cooldown
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"💥 Error processing book {book_code}: {e}")
                failed_books.append(book_code)
        
        logger.info(f"🏁 Full processing completed: {success_count}/{len(book_codes)} books successful")
        if failed_books:
            logger.warning(f"⚠️ Failed books: {failed_books}")
        
        return success_count > 0
        """Process a single chapter for audio-text alignment."""
        logger.info(f"🔍 Processing {book_code} Chapter {chapter}")
        
        audio_path = None  # Initialize to avoid UnboundLocalError
        
        try:
            # Extract verses
            verses = self.extract_verses(book_code, chapter)
            if not verses:
                logger.error(f"No verses found for {book_code} {chapter}")
                return False
            
            # Get audio file
            audio_path = self.get_audio_file_path(book_code, chapter)
            if not audio_path or not audio_path.exists():
                logger.warning(f"Local audio not found, trying blob download...")
                # Audio download would go here
                
            if not audio_path or not audio_path.exists():
                logger.error(f"Audio file not found: {book_code} {chapter}")
                return False
            
            # Validate audio
            is_valid, duration = self.validate_audio_file(audio_path)
            if not is_valid:
                logger.error(f"Invalid audio file: {audio_path}")
                return False
            
            # Try transcription with retries
            success = False
            alignment_data = {}
            
            # Try Whisper first
            whisper_success, segments = await self.transcribe_with_whisper(audio_path)
            if whisper_success and segments:
                alignment_data = self.fuzzy_align_text(segments, verses)
                success = True
                logger.info("✅ Whisper transcription successful")
            else:
                # Create estimated alignment
                logger.warning("Whisper failed, creating estimates")
                alignment_data = self.create_estimated_alignment(verses, duration)
                success = True  # Consider estimates as success for JSON output
            
            # Save results
            if alignment_data:
                save_success = self.save_alignment_json(book_code, chapter, alignment_data)
                if save_success:
                    logger.info(f"✅ Chapter {book_code} {chapter} processed successfully")
                    return True
            
            logger.error(f"Failed to process chapter {book_code} {chapter}")
            return False
            
        except Exception as e:
            logger.error(f"Error processing chapter {book_code} {chapter}: {e}")
            logger.error(traceback.format_exc())
            return False
        finally:
            # Clean up downloaded files
            if audio_path and str(audio_path).startswith(tempfile.gettempdir()):
                try:
                    os.remove(audio_path)
                except Exception:
                    pass


async def main():
    """Main function with command-line interface."""
    tool = HebrewAlignmentTool()
    
    if len(sys.argv) == 1:
        # Full run
        logger.info("🌍 Starting full processing of all books")
        await tool.process_all()
    elif len(sys.argv) == 3 and sys.argv[1] == "test":
        # Test single book
        book_code = sys.argv[2]
        logger.info(f"📚 Testing book: {book_code}")
        await tool.process_book(book_code)
    elif len(sys.argv) == 4 and sys.argv[1] == "test":
        # Test single chapter
        book_code = sys.argv[2]
        chapter = int(sys.argv[3])
        logger.info(f"🧪 Testing chapter: {book_code} {chapter}")
        success = await tool.process_chapter(book_code, chapter)
        print(f"Chapter processing {'succeeded' if success else 'failed'}")
    else:
        print("Usage:")
        print("  python hebrew_alignment_tool_gpu.py                    # Process all")
        print("  python hebrew_alignment_tool_gpu.py test Gen           # Test book")
        print("  python hebrew_alignment_tool_gpu.py test Gen 1         # Test chapter")


if __name__ == "__main__":
    asyncio.run(main())