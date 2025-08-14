#!/usr/bin/env python3
"""
Hebrew Bible Audio-Text Alignment Tool - FIXED VERSION
Addresses Azure Speech SDK errors and book mapping issues
"""

import os
import json
import time
import logging
import asyncio
import tempfile
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import re
from rapidfuzz import fuzz
from dotenv import load_dotenv
import requests
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech import SpeechConfig, AudioConfig
from azure.storage.blob import BlobServiceClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('D:/AI/Gits/hebrew-tutor-data-pipeline/hebrew_alignment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# COMPLETE UPDATED BOOK MAPPING - covers all possible book codes
COMPLETE_BOOK_MAPPING = {
    # Original mappings
    'Gen': {'folder': '01_Genesis', 'abbr': 'Gen'},
    'Exo': {'folder': '02_Exodus', 'abbr': 'Exo'},
    'Lev': {'folder': '03_Leviticus', 'abbr': 'Lev'},
    'Num': {'folder': '04_Numbers', 'abbr': 'Num'},
    'Deu': {'folder': '05_Deuteronomy', 'abbr': 'Deu'},
    'Jos': {'folder': '06_Joshua', 'abbr': 'Jos'},
    'Jdg': {'folder': '07_Judges', 'abbr': 'Judg'},
    '1Sa': {'folder': '08_1Samuel', 'abbr': '1Sam'},
    '2Sa': {'folder': '09_2Samuel', 'abbr': '2Sam'},
    '1Ki': {'folder': '10_1Kings', 'abbr': '1King'},
    '2Ki': {'folder': '11_2Kings', 'abbr': '2King'},
    'Isa': {'folder': '12_Isaiah', 'abbr': 'Isa'},
    'Jer': {'folder': '13_Jeremiah', 'abbr': 'Jer'},
    'Ezk': {'folder': '14_Ezekiel', 'abbr': 'Ezek'},
    'Hos': {'folder': '15_Hosea', 'abbr': 'Hos'},
    'Jol': {'folder': '16_Joel', 'abbr': 'Joel'},
    'Amo': {'folder': '17_Amos', 'abbr': 'Amos'},
    'Oba': {'folder': '18_Obadiah', 'abbr': 'Obad'},
    'Jon': {'folder': '19_Jonah', 'abbr': 'Jonah'},
    'Mic': {'folder': '20_Micah', 'abbr': 'Mic'},
    'Nam': {'folder': '21_Nahum', 'abbr': 'Nah'},
    'Hab': {'folder': '22_Habakkuk', 'abbr': 'Hab'},
    'Zep': {'folder': '23_Zephaniah', 'abbr': 'Zeph'},
    'Hag': {'folder': '24_Haggai', 'abbr': 'Hag'},
    'Zec': {'folder': '25_Zechariah', 'abbr': 'Zech'},
    'Mal': {'folder': '26_Malachi', 'abbr': 'Mal'},
    'Psa': {'folder': '27_Psalms', 'abbr': 'Psa'},
    'Job': {'folder': '28_Job', 'abbr': 'Job'},
    'Pro': {'folder': '29_Proverbs', 'abbr': 'Prov'},
    'Rut': {'folder': '30_Ruth', 'abbr': 'Ruth'},
    'Sng': {'folder': '31_SongofSongs', 'abbr': 'Song'},
    'Ecc': {'folder': '32_Ecclesiastes', 'abbr': 'Eccl'},
    'Lam': {'folder': '33_Lamentations', 'abbr': 'Lam'},
    'Est': {'folder': '34_Esther', 'abbr': 'Esth'},
    'Dan': {'folder': '35_Daniel', 'abbr': 'Dan'},
    'Ezr': {'folder': '36_Ezra', 'abbr': 'Ezra'},
    'Neh': {'folder': '37_Nehemiah', 'abbr': 'Nehem'},
    '1Chr': {'folder': '38_1Chronicles', 'abbr': '1Chron'},
    '2Chr': {'folder': '39_2Chronicles', 'abbr': '2Chron'},
    
    # Additional mappings for missing codes found in logs
    'Obad': {'folder': '18_Obadiah', 'abbr': 'Obad'},
    'Prov': {'folder': '29_Proverbs', 'abbr': 'Prov'},
    'Ps': {'folder': '27_Psalms', 'abbr': 'Psa'},
    'Ruth': {'folder': '30_Ruth', 'abbr': 'Ruth'},
    'Song': {'folder': '31_SongofSongs', 'abbr': 'Song'},
    'Zech': {'folder': '25_Zechariah', 'abbr': 'Zech'},
    'Zeph': {'folder': '23_Zephaniah', 'abbr': 'Zeph'},
    
    # Alternative spellings/codes
    'Psalms': {'folder': '27_Psalms', 'abbr': 'Psa'},
    'Proverbs': {'folder': '29_Proverbs', 'abbr': 'Prov'},
    'Joel': {'folder': '16_Joel', 'abbr': 'Joel'},
    'Obadiah': {'folder': '18_Obadiah', 'abbr': 'Obad'},
    'Zephaniah': {'folder': '23_Zephaniah', 'abbr': 'Zeph'},
    'Zechariah': {'folder': '25_Zechariah', 'abbr': 'Zech'},
}

class HebrewBibleAligner:
    def __init__(self):
        self.subscription_key = os.getenv('AZURE_SPEECH_KEY')
        self.region = os.getenv('AZURE_SPEECH_REGION', 'eastus')
        self.storage_connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        self.use_batch = os.getenv('USE_BATCH', 'False').lower() == 'true'
        
        if not self.subscription_key:
            raise ValueError("AZURE_SPEECH_KEY environment variable not set")

        # Initialize speech config with error handling
        try:
            self.speech_config = SpeechConfig(
                subscription=self.subscription_key,
                region=self.region
            )
            self.speech_config.speech_recognition_language = "he-IL"
            self.speech_config.request_word_level_timestamps()
            self.speech_config.output_format = speechsdk.OutputFormat.Detailed
            
            # Additional configuration to prevent header errors
            self.speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_RecoMode, "INTERACTIVE")
            self.speech_config.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "5000")
            
        except Exception as e:
            logger.error(f"Failed to initialize Speech Config: {e}")
            raise

        # Initialize blob storage if batch mode is enabled
        self.blob_service_client = None
        self.container_client = None
        if self.use_batch and self.storage_connection_string:
            try:
                self.blob_service_client = BlobServiceClient.from_connection_string(self.storage_connection_string)
                self.container_client = self.blob_service_client.get_container_client("hebrew-audio")
                try:
                    self.container_client.create_container()
                except Exception:
                    pass  # Container might already exist
            except Exception as e:
                logger.warning(f"Failed to initialize blob storage: {e}. Falling back to local transcription.")
                self.use_batch = False

        # Initialize paths
        self.json_path = Path(os.getenv('HEBREW_BIBLE_JSON_PATH', 'D:/AI/Projects/HEBREW-TRAINING-AI-AGENT/TANACH/book/hebrew_bible_with_nikkud.json'))
        self.audio_base_dir = Path(os.getenv('AUDIO_BASE_DIR', 'D:/audio/tankh_audio_chp'))
        self.output_base_dir = Path(os.getenv('OUTPUT_BASE_DIR', 'D:/AI/Projects/HEBREW-TRAINING-AI-AGENT/TANACH/alignment'))

        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        # Load and validate Bible data
        self.bible_data = self._load_bible_data()
        self._validate_book_mappings()
        
        logger.info(f"Initialized Hebrew Bible Aligner for {len(self.bible_data)} books")

    def _load_bible_data(self) -> Dict[str, List[List[List[str]]]]:
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'Aramaic' in data:
                del data['Aramaic']
            logger.info(f"Loaded Bible data with {len(data)} books")
            logger.info(f"Available book codes: {list(data.keys())}")
            return data
        except FileNotFoundError:
            logger.error(f"Bible data file not found: {self.json_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in Bible data file: {e}")
            raise

    def _validate_book_mappings(self):
        """Check which books from JSON don't have mappings and report missing audio files"""
        missing_mappings = []
        missing_audio = []
        
        for book_code in self.bible_data.keys():
            if book_code not in COMPLETE_BOOK_MAPPING:
                missing_mappings.append(book_code)
            else:
                # Check if audio files exist for this book
                try:
                    chapters = self.bible_data[book_code]
                    for chapter_num in range(1, min(3, len(chapters) + 1)):  # Check first 2 chapters
                        audio_file = self._get_audio_filename(book_code, chapter_num)
                        if not audio_file.exists():
                            missing_audio.append(f"{book_code} Ch{chapter_num}: {audio_file}")
                            break  # Don't check all chapters if first is missing
                except Exception as e:
                    logger.debug(f"Error checking audio for {book_code}: {e}")
        
        if missing_mappings:
            logger.warning(f"Missing book mappings: {missing_mappings}")
        
        if missing_audio:
            logger.warning(f"Sample missing audio files: {missing_audio[:10]}...")  # Show first 10

    def _get_audio_filename(self, book_code: str, chapter: int) -> Path:
        if book_code not in COMPLETE_BOOK_MAPPING:
            raise ValueError(f"No mapping for book code {book_code}")
        
        mapping = COMPLETE_BOOK_MAPPING[book_code]
        folder_name = mapping['folder']
        abbr = mapping['abbr']
        num = folder_name.split('_')[0]
        audio_filename = f"{num}-{abbr}_{chapter:02d}.mp3"
        return self.audio_base_dir / folder_name / audio_filename

    def _extract_chapter_text(self, book_code: str, chapter_num: int) -> Tuple[str, List[Dict]]:
        try:
            chapters = self.bible_data[book_code]
            if chapter_num > len(chapters):
                raise IndexError(f"Chapter {chapter_num} not found in {book_code}")

            chapter_data = chapters[chapter_num - 1]
            verse_metadata = []
            full_text_parts = []

            for verse_num, verse_words in enumerate(chapter_data, 1):
                verse_text = ' '.join(verse_words)
                verse_metadata.append({
                    'verse_num': verse_num,
                    'text': verse_text,
                    'words': verse_words,
                    'word_count': len(verse_words)
                })
                full_text_parts.append(verse_text)

            full_chapter_text = ' '.join(full_text_parts)
            logger.info(f"Extracted {len(verse_metadata)} verses for {book_code} Ch{chapter_num}")
            return full_chapter_text, verse_metadata
        except KeyError:
            logger.error(f"Book code '{book_code}' not found in Bible data")
            raise
        except IndexError as e:
            logger.error(f"Chapter extraction error: {e}")
            raise

    def _validate_audio_file(self, audio_file: Path) -> bool:
        """Validate audio file before processing"""
        if not audio_file.exists():
            return False
        
        # Check file size (should be > 1KB for valid MP3)
        if audio_file.stat().st_size < 1024:
            logger.warning(f"Audio file too small: {audio_file}")
            return False
        
        # Check file extension
        if audio_file.suffix.lower() not in ['.mp3', '.wav', '.m4a']:
            logger.warning(f"Unsupported audio format: {audio_file}")
            return False
        
        return True

    async def _transcribe_with_timestamps(self, audio_file: Path, reference_text: str) -> Optional[Dict]:
        """Enhanced transcription with better error handling"""
        try:
            # Validate audio file first
            if not self._validate_audio_file(audio_file):
                logger.error(f"Audio file validation failed: {audio_file}")
                return None

            # Create audio config with absolute path
            audio_config = AudioConfig(filename=str(audio_file.absolute()))
            
            # Create recognizer with retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    speech_recognizer = speechsdk.SpeechRecognizer(
                        speech_config=self.speech_config,
                        audio_config=audio_config
                    )
                    break
                except Exception as e:
                    logger.warning(f"Recognizer creation attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(1)
            
            transcription_results = []
            recognition_done = asyncio.Event()
            
            def handle_final_result(evt: speechsdk.SpeechRecognitionEventArgs):
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    try:
                        result_json = json.loads(evt.result.json)
                        transcription_results.append(result_json)
                        logger.debug(f"Recognized: {evt.result.text[:100]}...")
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse recognition result JSON")
                elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                    logger.warning(f"No speech recognized: {evt.result.no_match_details}")

            def handle_canceled(evt: speechsdk.SpeechRecognitionCanceledEventArgs):
                logger.error(f"Recognition canceled: {evt.cancellation_details}")
                recognition_done.set()

            def handle_session_stopped(evt):
                logger.debug("Recognition session stopped")
                recognition_done.set()

            # Connect event handlers
            speech_recognizer.recognized.connect(handle_final_result)
            speech_recognizer.canceled.connect(handle_canceled)
            speech_recognizer.session_stopped.connect(handle_session_stopped)

            logger.info(f"Starting transcription for {audio_file.name}")
            speech_recognizer.start_continuous_recognition()

            # Wait for completion with timeout
            try:
                await asyncio.wait_for(recognition_done.wait(), timeout=300)  # 5 minute timeout
            except asyncio.TimeoutError:
                logger.warning(f"Transcription timeout for {audio_file.name}")
            finally:
                speech_recognizer.stop_continuous_recognition()

            if not transcription_results:
                logger.warning(f"No transcription results for {audio_file.name}")
                return None

            return self._process_transcription_results(transcription_results, reference_text)

        except Exception as e:
            logger.error(f"Transcription error for {audio_file}: {e}")
            return None

    async def _transcribe_with_batch(self, audio_file: Path, reference_text: str) -> Optional[Dict]:
        """Batch transcription with fallback to local"""
        if not self.blob_service_client or not self.use_batch:
            logger.info("Using local transcription")
            return await self._transcribe_with_timestamps(audio_file, reference_text)

        # Try batch transcription
        try:
            # Upload to blob storage
            blob_name = f"audio/{audio_file.name}_{int(time.time())}"
            blob_client = self.container_client.get_blob_client(blob_name)
            
            with open(audio_file, 'rb') as f:
                blob_client.upload_blob(f, overwrite=True)
            
            audio_url = blob_client.url
            logger.info(f"Uploaded audio to blob: {blob_name}")

            # Create batch transcription job
            endpoint = f"https://{self.region}.stt.speech.microsoft.com/speech/recognition/transcribe/batch/v3.1/transcriptions"
            headers = {
                'Ocp-Apim-Subscription-Key': self.subscription_key,
                'Content-Type': 'application/json'
            }
            body = {
                "contentUrls": [audio_url],
                "locale": "he-IL",
                "displayName": f"Hebrew Alignment - {audio_file.name}",
                "properties": {
                    "wordLevelTimestamps": True,
                    "punctuationMode": "DictatedAndAutomatic"
                }
            }

            response = requests.post(endpoint, headers=headers, json=body)
            if response.status_code != 201:
                logger.error(f"Batch job creation failed: {response.text}")
                raise Exception("Batch transcription failed")

            job_location = response.headers['Location']
            job_id = job_location.split('/')[-1]
            logger.info(f"Created batch job: {job_id}")

            # Poll for completion
            status_endpoint = f"{endpoint}/{job_id}"
            max_attempts = 60
            
            for attempt in range(max_attempts):
                status_response = requests.get(status_endpoint, headers=headers)
                status_data = status_response.json()
                status = status_data['status']
                
                if status == "Succeeded":
                    logger.info(f"Batch transcription completed: {job_id}")
                    
                    # Get results
                    files_endpoint = f"{status_endpoint}/files"
                    files_response = requests.get(files_endpoint, headers=headers)
                    files = files_response.json()['values']
                    
                    transcript_file = next((f for f in files if f['kind'] == 'Transcription'), None)
                    if not transcript_file:
                        raise Exception("No transcription file found")
                    
                    transcript_url = transcript_file['links']['contentUrl']
                    transcript_response = requests.get(transcript_url)
                    transcript_data = transcript_response.json()
                    
                    return self._process_transcription_results(transcript_data, reference_text)
                    
                elif status == "Failed":
                    error_msg = status_data.get('properties', {}).get('error', {}).get('message', 'Unknown error')
                    logger.error(f"Batch transcription failed: {error_msg}")
                    raise Exception(f"Batch transcription failed: {error_msg}")
                
                await asyncio.sleep(5)
            
            raise Exception("Batch transcription timeout")
            
        except Exception as e:
            logger.warning(f"Batch transcription failed: {e}. Falling back to local.")
            return await self._transcribe_with_timestamps(audio_file, reference_text)

    def _process_transcription_results(self, results: Any, reference_text: str) -> Dict:
        """Process transcription results with enhanced error handling"""
        processed_words = []
        total_confidence = 0
        word_count = 0
        total_duration = 0

        try:
            # Handle batch API response format
            if isinstance(results, dict) and 'recognizedPhrases' in results:
                for phrase in results['recognizedPhrases']:
                    if not phrase.get('nbest'):
                        continue
                        
                    best_result = phrase['nbest'][0]
                    phrase_confidence = best_result.get('confidence', 0.0)
                    
                    for word in best_result.get('words', []):
                        start_time = word.get('offsetInTicks', 0) / 10000000.0
                        duration = word.get('durationInTicks', 0) / 10000000.0
                        end_time = start_time + duration
                        
                        processed_word = {
                            'text': word.get('word', ''),
                            'start': start_time,
                            'end': end_time,
                            'confidence': phrase_confidence
                        }
                        processed_words.append(processed_word)
                        total_confidence += phrase_confidence
                        word_count += 1
                        total_duration = max(total_duration, end_time)

            # Handle local SDK response format
            elif isinstance(results, list):
                for result in results:
                    if not result.get('NBest'):
                        continue
                        
                    best_result = result['NBest'][0]
                    result_confidence = best_result.get('Confidence', 0.0)
                    
                    for word_info in best_result.get('Words', []):
                        start_time = word_info.get('Offset', 0) / 10000000.0
                        duration = word_info.get('Duration', 0) / 10000000.0
                        end_time = start_time + duration
                        
                        processed_word = {
                            'text': word_info.get('Word', ''),
                            'start': start_time,
                            'end': end_time,
                            'confidence': result_confidence
                        }
                        processed_words.append(processed_word)
                        total_confidence += result_confidence
                        word_count += 1
                        total_duration = max(total_duration, end_time)

            average_confidence = total_confidence / word_count if word_count > 0 else 0.0

            return {
                'words': processed_words,
                'confidence': average_confidence,
                'word_count': word_count,
                'total_duration': total_duration
            }
            
        except Exception as e:
            logger.error(f"Error processing transcription results: {e}")
            return {
                'words': [],
                'confidence': 0.0,
                'word_count': 0,
                'total_duration': 0.0
            }

    def _align_words_to_reference(self, transcribed_words: List[Dict], verse_metadata: List[Dict]) -> List[Dict]:
        """Enhanced word alignment with better fuzzy matching"""
        aligned_verses = []
        transcription_index = 0

        for verse_meta in verse_metadata:
            verse_words = verse_meta['words']
            aligned_words = []
            verse_start_time = None
            verse_end_time = None

            for ref_word in verse_words:
                best_match = None
                best_similarity = 0.0
                best_index = transcription_index

                # Look for best match in next few words
                search_window = min(len(transcribed_words), transcription_index + 5)
                
                for i in range(transcription_index, search_window):
                    if i >= len(transcribed_words):
                        break
                        
                    transcribed_word = transcribed_words[i]
                    normalized_ref = self._normalize_hebrew_text(ref_word)
                    normalized_trans = self._normalize_hebrew_text(transcribed_word['text'])

                    # Try multiple similarity metrics
                    ratio_sim = fuzz.ratio(normalized_ref, normalized_trans) / 100.0
                    partial_sim = fuzz.partial_ratio(normalized_ref, normalized_trans) / 100.0
                    similarity = max(ratio_sim, partial_sim)

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = transcribed_word
                        best_index = i

                # Use match if similarity is good enough
                if best_similarity >= 0.6 and best_match:
                    aligned_word = {
                        'text': ref_word,
                        'start': best_match['start'],
                        'end': best_match['end'],
                        'confidence': best_match['confidence'] * best_similarity
                    }
                    transcription_index = best_index + 1
                else:
                    # Estimate timing for unmatched words
                    estimated_duration = 0.4
                    estimated_start = verse_end_time if verse_end_time else 0.0
                    aligned_word = {
                        'text': ref_word,
                        'start': estimated_start,
                        'end': estimated_start + estimated_duration,
                        'confidence': 0.1
                    }

                aligned_words.append(aligned_word)
                if verse_start_time is None:
                    verse_start_time = aligned_word['start']
                verse_end_time = aligned_word['end']

            aligned_verse = {
                'verse_num': verse_meta['verse_num'],
                'text': verse_meta['text'],
                'start': verse_start_time or 0.0,
                'end': verse_end_time or 0.0,
                'words': aligned_words,
                'word_count': len(aligned_words)
            }
            aligned_verses.append(aligned_verse)

        return aligned_verses

    def _normalize_hebrew_text(self, text: str) -> str:
        """Enhanced Hebrew text normalization"""
        if not text:
            return ""
            
        # Remove nikkud (diacritics)
        nikkud_pattern = r'[\u0591-\u05C7]'
        normalized = re.sub(nikkud_pattern, '', text)
        
        # Keep only Hebrew letters and spaces
        normalized = re.sub(r'[^\u05D0-\u05EA\s]', '', normalized)
        
        # Normalize whitespace
        normalized = ' '.join(normalized.split()).strip()
        
        return normalized

    def _validate_output(self, output_data: Dict, audio_file: Path) -> bool:
        """Validate alignment output"""
        try:
            # Basic validation
            if not output_data.get('verses'):
                logger.warning("No verses in output data")
                return False
            
            if output_data.get('overall_confidence', 0) < 0.1:
                logger.warning("Very low confidence alignment")
                return False
            
            # Optional: Check against audio duration if librosa is available
            try:
                import librosa
                audio_duration = librosa.get_duration(path=str(audio_file))
                if output_data['total_duration'] > audio_duration * 1.2:
                    logger.warning(f"Alignment duration {output_data['total_duration']} significantly exceeds audio duration {audio_duration}")
                    return False
            except ImportError:
                logger.debug("librosa not available for audio validation")
            except Exception as e:
                logger.debug(f"Audio validation error: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error for {audio_file}: {e}")
            return False

    async def process_chapter(self, book_code: str, chapter_num: int) -> bool:
        """Process a single chapter with enhanced error handling"""
        try:
            logger.info(f"Processing {book_code} Chapter {chapter_num}")
            
            # Check if book mapping exists
            if book_code not in COMPLETE_BOOK_MAPPING:
                logger.error(f"No mapping for book code {book_code}")
                return False
            
            # Get audio file path
            audio_file = self._get_audio_filename(book_code, chapter_num)
            if not audio_file.exists():
                logger.error(f"Audio file not found: {audio_file}")
                return False

            # Extract chapter text
            try:
                chapter_text, verse_metadata = self._extract_chapter_text(book_code, chapter_num)
            except (KeyError, IndexError) as e:
                logger.error(f"Failed to extract chapter text for {book_code} Ch{chapter_num}: {e}")
                return False

            # Transcribe audio
            transcription_result = await self._transcribe_with_batch(audio_file, chapter_text)
            if not transcription_result:
                logger.error(f"Transcription failed for {book_code} Ch{chapter_num}")
                return False

            if transcription_result['confidence'] < 0.2:
                logger.warning(f"Low confidence transcription ({transcription_result['confidence']:.2f}) for {book_code} Ch{chapter_num}")

            # Align words
            aligned_verses = self._align_words_to_reference(transcription_result['words'], verse_metadata)

            # Create output data
            output_data = {
                'book': book_code,
                'chapter': chapter_num,
                'total_duration': transcription_result['total_duration'],
                'overall_confidence': transcription_result['confidence'],
                'verse_count': len(aligned_verses),
                'verses': aligned_verses,
                'metadata': {
                    'audio_file': str(audio_file),
                    'processing_timestamp': time.time(),
                    'transcribed_word_count': transcription_result['word_count'],
                    'alignment_method': 'batch' if self.use_batch else 'local'
                }
            }

            # Validate output
            if not self._validate_output(output_data, audio_file):
                logger.warning(f"Validation failed for {book_code} Ch{chapter_num}, but proceeding anyway")

            # Save output
            book_output_dir = self.output_base_dir / book_code
            book_output_dir.mkdir(exist_ok=True)
            output_filename = f"alignment_{book_code}_{chapter_num:02d}.json"
            output_path = book_output_dir / output_filename

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Successfully processed {book_code} Ch{chapter_num} -> {output_path}")
            logger.info(f"  Confidence: {transcription_result['confidence']:.2f}, Duration: {transcription_result['total_duration']:.1f}s")
            return True

        except Exception as e:
            logger.error(f"Error processing {book_code} Ch{chapter_num}: {e}")
            return False

    async def process_book(self, book_code: str, max_concurrent: int = 2) -> Dict[str, int]:
        """Process all chapters in a single book"""
        if book_code not in self.bible_data:
            logger.error(f"Book {book_code} not found in Bible data")
            return {'total_chapters': 0, 'successful': 0, 'failed': 1}

        chapters = self.bible_data[book_code]
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(chapter_num: int):
            async with semaphore:
                return await self.process_chapter(book_code, chapter_num)

        tasks = []
        for chapter_num in range(1, len(chapters) + 1):
            task = process_with_semaphore(chapter_num)
            tasks.append(task)

        logger.info(f"Processing {len(tasks)} chapters for book {book_code}")
        
        successful = 0
        failed = 0
        
        for i, task in enumerate(tasks, 1):
            try:
                result = await task
                if result:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Task failed for {book_code} Ch{i}: {e}")
                failed += 1

        summary = {
            'total_chapters': len(tasks),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(tasks) if tasks else 0
        }
        
        logger.info(f"Book {book_code} summary: {summary}")
        return summary

    async def process_all_chapters(self, max_concurrent: int = 3) -> Dict[str, int]:
        """Process all chapters with reduced concurrency to prevent Azure SDK errors"""
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = []
        total_chapters = 0

        async def process_with_semaphore(book_code: str, chapter_num: int):
            async with semaphore:
                # Add small delay between requests to prevent rate limiting
                await asyncio.sleep(0.1)
                return await self.process_chapter(book_code, chapter_num)

        # Only process books that have mappings
        valid_books = [book for book in self.bible_data.keys() if book in COMPLETE_BOOK_MAPPING]
        
        for book_code in valid_books:
            chapters = self.bible_data[book_code]
            for chapter_num in range(1, len(chapters) + 1):
                task = process_with_semaphore(book_code, chapter_num)
                tasks.append((book_code, chapter_num, task))
                total_chapters += 1

        logger.info(f"Starting processing of {total_chapters} chapters from {len(valid_books)} books with max {max_concurrent} concurrent")
        
        successful = 0
        failed = 0

        for book_code, chapter_num, task in tasks:
            try:
                result = await task
                if result:
                    successful += 1
                else:
                    failed += 1
                
                processed = successful + failed
                if processed % 10 == 0:
                    logger.info(f"Progress: {processed}/{total_chapters} chapters processed ({successful} successful, {failed} failed)")
                    
            except Exception as e:
                logger.error(f"Task failed for {book_code} Ch{chapter_num}: {e}")
                failed += 1

        summary = {
            'total_chapters': total_chapters,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total_chapters if total_chapters > 0 else 0
        }
        
        logger.info(f"Processing complete: {summary}")
        return summary

    def test_single_chapter(self, book_code: str = "Gen", chapter_num: int = 1) -> bool:
        """Test processing a single chapter for debugging"""
        async def test():
            return await self.process_chapter(book_code, chapter_num)
        
        return asyncio.run(test())

    def list_available_books(self) -> List[str]:
        """List books that have both data and mapping"""
        available = []
        for book_code in self.bible_data.keys():
            if book_code in COMPLETE_BOOK_MAPPING:
                available.append(book_code)
        return available

    def diagnose_setup(self) -> Dict[str, Any]:
        """Diagnose setup issues"""
        diagnosis = {
            'azure_key_set': bool(self.subscription_key),
            'azure_region': self.region,
            'bible_data_loaded': bool(self.bible_data),
            'total_books_in_data': len(self.bible_data),
            'books_with_mappings': len([b for b in self.bible_data.keys() if b in COMPLETE_BOOK_MAPPING]),
            'sample_audio_files_exist': {},
            'missing_mappings': [b for b in self.bible_data.keys() if b not in COMPLETE_BOOK_MAPPING]
        }
        
        # Check sample audio files
        sample_books = list(self.bible_data.keys())[:5]  # Check first 5 books
        for book in sample_books:
            if book in COMPLETE_BOOK_MAPPING:
                try:
                    audio_file = self._get_audio_filename(book, 1)
                    diagnosis['sample_audio_files_exist'][book] = audio_file.exists()
                except Exception as e:
                    diagnosis['sample_audio_files_exist'][book] = f"Error: {e}"
        
        return diagnosis


async def main():
    """Main execution with better error handling and testing options"""
    try:
        aligner = HebrewBibleAligner()
        
        # Run diagnosis first
        diagnosis = aligner.diagnose_setup()
        logger.info(f"Setup diagnosis: {json.dumps(diagnosis, indent=2, default=str)}")
        
        # Test with a single chapter first
        logger.info("Testing with Genesis Chapter 1...")
        test_result = aligner.test_single_chapter("Gen", 1)
        
        if test_result:
            logger.info("Test successful! Proceeding with full processing...")
            # Uncomment the line below to process all chapters
            # summary = await aligner.process_all_chapters(max_concurrent=2)
            summary = {"test_only": True, "test_result": test_result}
        else:
            logger.error("Test failed. Please check the logs for issues.")
            summary = {"test_only": True, "test_result": False}
        
        print(f"Processing summary: {summary}")

    except Exception as e:
        logger.error(f"Main execution error: {e}")
        raise


if __name__ == "__main__":
    # For testing individual components
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "diagnose":
            aligner = HebrewBibleAligner()
            diagnosis = aligner.diagnose_setup()
            print(json.dumps(diagnosis, indent=2, default=str))
            
        elif command == "test" and len(sys.argv) >= 4:
            book = sys.argv[2]
            chapter = int(sys.argv[3])
            aligner = HebrewBibleAligner()
            result = aligner.test_single_chapter(book, chapter)
            print(f"Test result for {book} Ch{chapter}: {result}")
            
        elif command == "list":
            aligner = HebrewBibleAligner()
            books = aligner.list_available_books()
            print(f"Available books: {books}")
            
        else:
            print("Usage:")
            print("  python script.py diagnose")
            print("  python script.py test <book> <chapter>")
            print("  python script.py list")
    else:
        asyncio.run(main())