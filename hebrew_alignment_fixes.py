#!/usr/bin/env python3
"""
Critical fixes for Hebrew Bible Audio-Text Alignment Tool
Addresses batch transcription issues and missing book mappings
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
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech import SpeechConfig, AudioConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('D:/AI/Gits/hebrew-tutor-data-pipeline/hebrew_alignment_fixed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# COMPLETE Book mapping - includes all Tanakh books
COMPLETE_BOOK_MAPPING = {
    # Torah (5 books)
    'Gen': {'folder': '01_Genesis', 'abbr': 'Gen'},
    'Exo': {'folder': '02_Exodus', 'abbr': 'Exo'},
    'Lev': {'folder': '03_Leviticus', 'abbr': 'Lev'},
    'Num': {'folder': '04_Numbers', 'abbr': 'Num'},
    'Deu': {'folder': '05_Deuteronomy', 'abbr': 'Deu'},
    
    # Nevi'im - Former Prophets (6 books)
    'Jos': {'folder': '06_Joshua', 'abbr': 'Jos'},
    'Jdg': {'folder': '07_Judges', 'abbr': 'Judg'},
    'Judg': {'folder': '07_Judges', 'abbr': 'Judg'},  # Alternative
    '1Sa': {'folder': '08_1Samuel', 'abbr': '1Sam'},
    '2Sa': {'folder': '09_2Samuel', 'abbr': '2Sam'},
    '1Ki': {'folder': '10_1Kings', 'abbr': '1King'},
    '2Ki': {'folder': '11_2Kings', 'abbr': '2King'},
    
    # Nevi'im - Latter Prophets (15 books)
    'Isa': {'folder': '12_Isaiah', 'abbr': 'Isa'},
    'Jer': {'folder': '13_Jeremiah', 'abbr': 'Jer'},
    'Ezk': {'folder': '14_Ezekiel', 'abbr': 'Ezek'},
    'Ezek': {'folder': '14_Ezekiel', 'abbr': 'Ezek'},  # Alternative
    'Hos': {'folder': '15_Hosea', 'abbr': 'Hos'},
    'Jol': {'folder': '16_Joel', 'abbr': 'Joel'},
    'Joel': {'folder': '16_Joel', 'abbr': 'Joel'},  # Alternative
    'Amo': {'folder': '17_Amos', 'abbr': 'Amos'},
    'Amos': {'folder': '17_Amos', 'abbr': 'Amos'},  # Alternative
    'Oba': {'folder': '18_Obadiah', 'abbr': 'Obad'},
    'Obad': {'folder': '18_Obadiah', 'abbr': 'Obad'},  # Alternative
    'Jon': {'folder': '19_Jonah', 'abbr': 'Jonah'},
    'Jonah': {'folder': '19_Jonah', 'abbr': 'Jonah'},  # Alternative
    'Mic': {'folder': '20_Micah', 'abbr': 'Mic'},
    'Micah': {'folder': '20_Micah', 'abbr': 'Mic'},  # Alternative
    'Nam': {'folder': '21_Nahum', 'abbr': 'Nah'},
    'Nah': {'folder': '21_Nahum', 'abbr': 'Nah'},  # Alternative
    'Hab': {'folder': '22_Habakkuk', 'abbr': 'Hab'},
    'Habakkuk': {'folder': '22_Habakkuk', 'abbr': 'Hab'},  # Alternative
    'Zep': {'folder': '23_Zephaniah', 'abbr': 'Zeph'},
    'Zeph': {'folder': '23_Zephaniah', 'abbr': 'Zeph'},  # Alternative
    'Hag': {'folder': '24_Haggai', 'abbr': 'Hag'},
    'Haggai': {'folder': '24_Haggai', 'abbr': 'Hag'},  # Alternative
    'Zec': {'folder': '25_Zechariah', 'abbr': 'Zech'},
    'Zech': {'folder': '25_Zechariah', 'abbr': 'Zech'},  # Alternative
    'Zechariah': {'folder': '25_Zechariah', 'abbr': 'Zech'},  # Alternative
    'Mal': {'folder': '26_Malachi', 'abbr': 'Mal'},
    'Malachi': {'folder': '26_Malachi', 'abbr': 'Mal'},  # Alternative
    
    # Ketuvim - Writings (11 books)
    'Psa': {'folder': '27_Psalms', 'abbr': 'Psa'},
    'Ps': {'folder': '27_Psalms', 'abbr': 'Psa'},  # Alternative
    'Psalms': {'folder': '27_Psalms', 'abbr': 'Psa'},  # Alternative
    'Job': {'folder': '28_Job', 'abbr': 'Job'},
    'Pro': {'folder': '29_Proverbs', 'abbr': 'Prov'},
    'Prov': {'folder': '29_Proverbs', 'abbr': 'Prov'},  # Alternative
    'Proverbs': {'folder': '29_Proverbs', 'abbr': 'Prov'},  # Alternative
    'Rut': {'folder': '30_Ruth', 'abbr': 'Ruth'},
    'Ruth': {'folder': '30_Ruth', 'abbr': 'Ruth'},  # Alternative
    'Sng': {'folder': '31_SongofSongs', 'abbr': 'Song'},
    'Song': {'folder': '31_SongofSongs', 'abbr': 'Song'},  # Alternative
    'SongofSongs': {'folder': '31_SongofSongs', 'abbr': 'Song'},  # Alternative
    'Ecc': {'folder': '32_Ecclesiastes', 'abbr': 'Eccl'},
    'Eccl': {'folder': '32_Ecclesiastes', 'abbr': 'Eccl'},  # Alternative
    'Ecclesiastes': {'folder': '32_Ecclesiastes', 'abbr': 'Eccl'},  # Alternative
    'Lam': {'folder': '33_Lamentations', 'abbr': 'Lam'},
    'Lamentations': {'folder': '33_Lamentations', 'abbr': 'Lam'},  # Alternative
    'Est': {'folder': '34_Esther', 'abbr': 'Esth'},
    'Esth': {'folder': '34_Esther', 'abbr': 'Esth'},  # Alternative
    'Esther': {'folder': '34_Esther', 'abbr': 'Esth'},  # Alternative
    'Dan': {'folder': '35_Daniel', 'abbr': 'Dan'},
    'Daniel': {'folder': '35_Daniel', 'abbr': 'Dan'},  # Alternative
    'Ezr': {'folder': '36_Ezra', 'abbr': 'Ezra'},
    'Ezra': {'folder': '36_Ezra', 'abbr': 'Ezra'},  # Alternative
    'Neh': {'folder': '37_Nehemiah', 'abbr': 'Nehem'},
    'Nehem': {'folder': '37_Nehemiah', 'abbr': 'Nehem'},  # Alternative
    'Nehemiah': {'folder': '37_Nehemiah', 'abbr': 'Nehem'},  # Alternative
    '1Chr': {'folder': '38_1Chronicles', 'abbr': '1Chron'},
    '1Chron': {'folder': '38_1Chronicles', 'abbr': '1Chron'},  # Alternative
    '2Chr': {'folder': '39_2Chronicles', 'abbr': '2Chron'},
    '2Chron': {'folder': '39_2Chronicles', 'abbr': '2Chron'}  # Alternative
}

class FixedHebrewBibleAligner:
    """Fixed version of Hebrew Bible audio-text alignment using Azure Speech SDK"""

    def __init__(self):
        """Initialize the aligner with Azure credentials and paths"""
        # Azure credentials from environment variables
        self.subscription_key = os.getenv('AZURE_SPEECH_KEY')
        self.region = os.getenv('AZURE_SPEECH_REGION', 'eastus')
        
        if not self.subscription_key:
            raise ValueError("AZURE_SPEECH_KEY environment variable not set")

        # Initialize Azure Speech Config for LOCAL transcription only
        self.speech_config = SpeechConfig(
            subscription=self.subscription_key,
            region=self.region
        )
        self.speech_config.speech_recognition_language = "he-IL"
        self.speech_config.request_word_level_timestamps()
        self.speech_config.output_format = speechsdk.OutputFormat.Detailed

        # File paths
        self.json_path = Path('D:/AI/Projects/HEBREW TRAINING AI AGENT/TANACH/book/hebrew_bible_with_nikkud.json')
        self.audio_base_dir = Path('D:/audio/tankh_audio_chp')
        self.output_base_dir = Path('D:/AI/Projects/HEBREW TRAINING AI AGENT/TANACH/alignment')

        # Ensure output directory exists
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        # Load Hebrew Bible text data
        self.bible_data = self._load_bible_data()

        logger.info(f"Initialized Fixed Hebrew Bible Aligner for {len(self.bible_data)} books")

    def _load_bible_data(self) -> Dict[str, List[List[List[str]]]]:
        """Load the Hebrew Bible JSON data"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded Bible data with {len(data)} books")
            
            # Log available book codes for debugging
            logger.info(f"Available book codes: {list(data.keys())}")
            return data
        except FileNotFoundError:
            logger.error(f"Bible data file not found: {self.json_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in Bible data file: {e}")
            raise

    def _get_audio_filename(self, book_code: str, chapter: int) -> Path:
        """Generate audio filename for given book and chapter"""
        if book_code not in COMPLETE_BOOK_MAPPING:
            logger.error(f"No mapping for book code {book_code}")
            logger.info(f"Available mappings: {list(COMPLETE_BOOK_MAPPING.keys())}")
            raise ValueError(f"No mapping for book code {book_code}")
        
        mapping = COMPLETE_BOOK_MAPPING[book_code]
        folder_name = mapping['folder']
        abbr = mapping['abbr']
        num = folder_name.split('_')[0]  # Extract number like '01'
        audio_filename = f"{num}-{abbr}_{chapter:02d}.mp3"
        return self.audio_base_dir / folder_name / audio_filename

    def _extract_chapter_text(self, book_code: str, chapter_num: int) -> Tuple[str, List[Dict]]:
        """Extract text for a specific chapter and return full text + verse structure"""
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

    async def _transcribe_with_local_sdk(self, audio_file: Path, reference_text: str) -> Optional[Dict]:
        """
        Use LOCAL Azure Speech SDK transcription (avoiding batch transcription issues)
        """
        try:
            logger.info(f"Starting LOCAL transcription for {audio_file.name}")
            
            # Check if audio file exists
            if not audio_file.exists():
                logger.error(f"Audio file does not exist: {audio_file}")
                return None
            
            # Configure audio input
            audio_config = AudioConfig(filename=str(audio_file))
            
            # Create speech recognizer
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            # Storage for results
            all_results = []
            recognition_done = asyncio.Event()
            
            def handle_final_result(evt: speechsdk.SpeechRecognitionEventArgs):
                """Handle final recognition results"""
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    try:
                        result_json = json.loads(evt.result.json)
                        all_results.append(result_json)
                        logger.debug(f"Recognized segment: {evt.result.text[:50]}...")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse recognition JSON: {e}")
                elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                    logger.warning(f"No speech recognized in segment")
            
            def handle_canceled(evt: speechsdk.SpeechRecognitionCanceledEventArgs):
                """Handle cancellation"""
                logger.error(f"Recognition canceled: {evt.cancellation_details.reason}")
                if evt.cancellation_details.error_details:
                    logger.error(f"Error details: {evt.cancellation_details.error_details}")
                recognition_done.set()
            
            def handle_stopped(evt):
                """Handle recognition stopped"""
                logger.info("Recognition stopped")
                recognition_done.set()
            
            # Connect event handlers
            speech_recognizer.recognized.connect(handle_final_result)
            speech_recognizer.canceled.connect(handle_canceled)
            speech_recognizer.session_stopped.connect(handle_stopped)
            
            # Start continuous recognition
            speech_recognizer.start_continuous_recognition()
            
            # Wait for completion with timeout
            try:
                await asyncio.wait_for(recognition_done.wait(), timeout=300.0)  # 5 minute timeout
            except asyncio.TimeoutError:
                logger.warning(f"Recognition timeout for {audio_file.name}")
            
            # Stop recognition
            speech_recognizer.stop_continuous_recognition()
            
            if not all_results:
                logger.warning(f"No transcription results for {audio_file.name}")
                return None
            
            # Process and combine results
            return self._process_transcription_results(all_results, reference_text)
            
        except Exception as e:
            logger.error(f"Local transcription error for {audio_file}: {e}")
            return None

    def _process_transcription_results(self, results: List[Dict], reference_text: str) -> Dict:
        """Process Azure transcription results to extract word-level timestamps"""
        processed_words = []
        total_confidence = 0
        word_count = 0
        total_duration = 0

        for result in results:
            if 'NBest' in result and result['NBest']:
                best_result = result['NBest'][0]
                if 'Words' in best_result:
                    for word_info in best_result['Words']:
                        start_time = word_info.get('Offset', 0) / 10000000.0
                        end_time = (word_info.get('Offset', 0) + word_info.get('Duration', 0)) / 10000000.0
                        processed_word = {
                            'text': word_info.get('Word', ''),
                            'start': start_time,
                            'end': end_time,
                            'confidence': word_info.get('Confidence', 0.0)
                        }
                        processed_words.append(processed_word)
                        total_confidence += processed_word['confidence']
                        word_count += 1
                        total_duration = max(total_duration, end_time)

        average_confidence = total_confidence / word_count if word_count > 0 else 0.0

        return {
            'words': processed_words,
            'confidence': average_confidence,
            'word_count': word_count,
            'total_duration': total_duration
        }

    def _align_words_to_reference(self, transcribed_words: List[Dict], verse_metadata: List[Dict]) -> List[Dict]:
        """Align transcribed words with reference text using fuzzy matching"""
        aligned_verses = []
        transcription_index = 0

        for verse_meta in verse_metadata:
            verse_words = verse_meta['words']
            aligned_words = []
            verse_start_time = None
            verse_end_time = None

            for ref_word in verse_words:
                if transcription_index < len(transcribed_words):
                    transcribed_word = transcribed_words[transcription_index]
                    normalized_ref = self._normalize_hebrew_text(ref_word)
                    normalized_trans = self._normalize_hebrew_text(transcribed_word['text'])

                    similarity = fuzz.ratio(normalized_ref, normalized_trans) / 100.0
                    if similarity >= 0.6:  # Lowered threshold for better matching
                        aligned_word = {
                            'text': ref_word,
                            'start': transcribed_word['start'],
                            'end': transcribed_word['end'],
                            'confidence': transcribed_word['confidence'] * similarity
                        }
                        aligned_words.append(aligned_word)
                        if verse_start_time is None:
                            verse_start_time = transcribed_word['start']
                        verse_end_time = transcribed_word['end']
                        transcription_index += 1
                    else:
                        # Estimate timing for unmatched words
                        estimated_duration = 0.5
                        estimated_start = verse_end_time if verse_end_time else 0.0
                        aligned_word = {
                            'text': ref_word,
                            'start': estimated_start,
                            'end': estimated_start + estimated_duration,
                            'confidence': 0.1
                        }
                        aligned_words.append(aligned_word)
                        verse_end_time = estimated_start + estimated_duration
                else:
                    # No more transcribed words - estimate
                    estimated_duration = 0.5
                    estimated_start = verse_end_time if verse_end_time else 0.0
                    aligned_word = {
                        'text': ref_word,
                        'start': estimated_start,
                        'end': estimated_start + estimated_duration,
                        'confidence': 0.1
                    }
                    aligned_words.append(aligned_word)
                    verse_end_time = estimated_start + estimated_duration

            aligned_verse = {
                'verse_num': verse_meta['verse_num'],
                'text': verse_meta['text'],
                'start': verse_start_time if verse_start_time else 0.0,
                'end': verse_end_time if verse_end_time else 0.0,
                'words': aligned_words,
                'word_count': len(aligned_words)
            }
            aligned_verses.append(aligned_verse)

        return aligned_verses

    def _normalize_hebrew_text(self, text: str) -> str:
        """Normalize Hebrew text for comparison"""
        # Remove nikkud (Hebrew diacritics)
        nikkud_pattern = r'[\u0591-\u05C7]'
        normalized = re.sub(nikkud_pattern, '', text)
        # Remove punctuation and extra spaces
        normalized = re.sub(r'[^\u05D0-\u05EA\s]', '', normalized)
        return ' '.join(normalized.split()).strip()

    async def process_chapter(self, book_code: str, chapter_num: int) -> bool:
        """Process a single chapter with improved error handling"""
        try:
            logger.info(f"Processing {book_code} Chapter {chapter_num}")
            
            # Check if book code exists in mapping
            if book_code not in COMPLETE_BOOK_MAPPING:
                logger.error(f"Book code '{book_code}' not found in mapping")
                return False
            
            # Get audio file path
            try:
                audio_file = self._get_audio_filename(book_code, chapter_num)
            except ValueError as e:
                logger.error(f"Could not get audio filename: {e}")
                return False
            
            if not audio_file.exists():
                logger.error(f"Audio file not found: {audio_file}")
                return False

            # Extract chapter text
            try:
                chapter_text, verse_metadata = self._extract_chapter_text(book_code, chapter_num)
            except (KeyError, IndexError) as e:
                logger.error(f"Could not extract chapter text: {e}")
                return False

            # Transcribe with LOCAL SDK (avoiding batch transcription issues)
            transcription_result = await self._transcribe_with_local_sdk(audio_file, chapter_text)

            if not transcription_result:
                logger.error(f"Transcription failed for {book_code} Ch{chapter_num}")
                return False

            if transcription_result['confidence'] < 0.2:
                logger.warning(f"Very low confidence transcription: {transcription_result['confidence']}")

            # Align words to reference text
            aligned_verses = self._align_words_to_reference(transcription_result['words'], verse_metadata)

            # Create output structure
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
                    'processing_method': 'local_sdk'
                }
            }

            # Create book-specific output directory
            book_output_dir = self.output_base_dir / book_code
            book_output_dir.mkdir(exist_ok=True)
            
            # Save alignment result
            output_filename = f"alignment_{book_code}_{chapter_num:02d}.json"
            output_path = book_output_dir / output_filename

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ Successfully processed {book_code} Ch{chapter_num} -> {output_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Error processing {book_code} Ch{chapter_num}: {e}")
            return False

    async def process_single_book(self, book_code: str, max_chapters: int = None) -> Dict[str, int]:
        """Process a single book for testing"""
        if book_code not in self.bible_data:
            logger.error(f"Book '{book_code}' not found in Bible data")
            return {'successful': 0, 'failed': 1, 'total_chapters': 0}

        chapters = self.bible_data[book_code]
        total_chapters = len(chapters)
        if max_chapters:
            total_chapters = min(total_chapters, max_chapters)
        
        successful = 0
        failed = 0

        logger.info(f"Processing book {book_code} - {total_chapters} chapters")

        for chapter_num in range(1, total_chapters + 1):
            success = await self.process_chapter(book_code, chapter_num)
            if success:
                successful += 1
            else:
                failed += 1
            
            # Progress update
            logger.info(f"Book {book_code} progress: {successful + failed}/{total_chapters}")

        summary = {
            'book': book_code,
            'total_chapters': total_chapters,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total_chapters if total_chapters > 0 else 0
        }

        logger.info(f"Book {book_code} complete: {summary}")
        return summary

# Test function
async def test_single_chapter():
    """Test with a single chapter to verify fixes"""
    try:
        logger.info("🔧 Testing fixed Hebrew Bible Aligner")
        aligner = FixedHebrewBibleAligner()
        
        # Test with Ruth (short book) - using correct book code
        test_book = 'Rut'  # Use the book code from your JSON
        if test_book not in aligner.bible_data:
            # Try alternative codes
            for alt_code in ['Ruth', 'Ru']:
                if alt_code in aligner.bible_data:
                    test_book = alt_code
                    break
            else:
                logger.error("Could not find Ruth in bible data")
                return
        
        success = await aligner.process_chapter(test_book, 1)
        if success:
            logger.info("✅ Single chapter test PASSED")
        else:
            logger.error("❌ Single chapter test FAILED")
            
        # Test a small book
        summary = await aligner.process_single_book(test_book, max_chapters=2)
        logger.info(f"📊 Test summary: {summary}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

# Main execution
async def main_fixed():
    """Main execution with fixes"""
    try:
        # Test single chapter first
        await test_single_chapter()
        
        # If test passes, you can uncomment this to process more
        # aligner = FixedHebrewBibleAligner()
        # summary = await aligner.process_single_book('Gen', max_chapters=5)
        # print(f"Processing summary: {summary}")
        
    except Exception as e:
        logger.error(f"Main execution error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main_fixed())
