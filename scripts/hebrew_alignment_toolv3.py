# File: D:\AI\Gits\hebrew-tutor-data-pipeline\hebrew_alignment_tool.py
#!/usr/bin/env python3
"""
Hebrew Bible Audio-Text Alignment Tool
Uses Azure Cognitive Services Speech SDK for forced alignment of Hebrew audio with text
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
from azure.storage.blob import BlobServiceClient
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech import SpeechConfig, AudioConfig
from azure.cognitiveservices.speech.transcription import (
    ConversationTranscriber,
    ConversationTranscriptionEventArgs
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('D:/AI/Gits/hebrew-tutor-data-pipeline/hebrew_alignment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Book to folder mapping
BOOK_TO_FOLDER = {
    'Gen': '01_Genesis',
    'Exo': '02_Exodus',
    'Lev': '03_Leviticus',
    'Num': '04_Numbers',
    'Deu': '05_Deuteronomy',
    'Jos': '06_Joshua',
    'Jdg': '07_Judges',
    '1Sa': '08_1Samuel',
    '2Sa': '09_2Samuel',
    '1Ki': '10_1Kings',
    '2Ki': '11_2Kings',
    'Isa': '12_Isaiah',
    'Jer': '13_Jeremiah',
    'Ezk': '14_Ezekiel',
    'Hos': '15_Hosea',
    'Jol': '16_Joel',
    'Amo': '17_Amos',
    'Oba': '18_Obadiah',
    'Jon': '19_Jonah',
    'Mic': '20_Micah',
    'Nam': '21_Nahum',
    'Hab': '22_Habakkuk',
    'Zep': '23_Zephaniah',
    'Hag': '24_Haggai',
    'Zec': '25_Zechariah',
    'Mal': '26_Malachi',
    'Psa': '27_Psalms',
    'Job': '28_Job',
    'Pro': '29_Proverbs',
    'Rut': '30_Ruth',
    'Sng': '31_SongofSongs',
    'Ecc': '32_Ecclesiastes',
    'Lam': '33_Lamentations',
    'Est': '34_Esther',
    'Dan': '35_Daniel',
    'Ezr': '36_Ezra',
    'Neh': '37_Nehemiah',
    '1Chr': '38_1Chronicles',
    '2Chr': '39_2Chronicles',
    'Aramaic': '40_Aramaic'
}

class HebrewBibleAligner:
    """Main class for Hebrew Bible audio-text alignment using Azure Speech SDK"""

    def __init__(self):
        """Initialize the aligner with Azure credentials and paths"""
        # Azure credentials from environment variables
        self.subscription_key = os.getenv('AZURE_SPEECH_KEY')
        self.region = os.getenv('AZURE_SPEECH_REGION', 'eastus')
        self.storage_connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        
        if not self.subscription_key:
            raise ValueError("AZURE_SPEECH_KEY environment variable not set")
        if not self.storage_connection_string:
            logger.warning("AZURE_STORAGE_CONNECTION_STRING not set; falling back to local processing")

        # Initialize Azure Speech Config
        self.speech_config = SpeechConfig(
            subscription=self.subscription_key,
            region=self.region
        )
        self.speech_config.speech_recognition_language = "he-IL"
        self.speech_config.request_word_level_timestamps()
        self.speech_config.output_format = speechsdk.OutputFormat.Detailed

        # Initialize Blob Service Client (if available)
        self.blob_service_client = None
        if self.storage_connection_string:
            self.blob_service_client = BlobServiceClient.from_connection_string(self.storage_connection_string)
            self.container_client = self.blob_service_client.get_container_client("hebrew-audio")

        # File paths
        self.json_path = Path('D:/AI/Projects/HEBREW TRAINING AI AGENT/TANACH/book/hebrew_bible_with_nikkud.json')
        self.audio_base_dir = Path('D:/audio/tankh_audio_chp')
        self.output_base_dir = Path('D:/AI/Projects/HEBREW TRAINING AI AGENT/TANACH/alignment')

        # Ensure output directory exists
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        # Load Hebrew Bible text data
        self.bible_data = self._load_bible_data()

        logger.info(f"Initialized Hebrew Bible Aligner for {len(self.bible_data)} books")

    def _load_bible_data(self) -> Dict[str, List[List[List[str]]]]:
        """Load the Hebrew Bible JSON data"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded Bible data with {len(data)} books")
            return data
        except FileNotFoundError:
            logger.error(f"Bible data file not found: {self.json_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in Bible data file: {e}")
            raise

    def _get_audio_filename(self, book_code: str, chapter: int) -> Path:
        """Generate audio filename for given book and chapter"""
        folder_name = BOOK_TO_FOLDER.get(book_code, '')
        if not folder_name:
            raise ValueError(f"No folder mapping for book code {book_code}")
        # Assume naming convention: {book_code}_{chapter:02d}.mp3 (e.g., Gen_01.mp3)
        audio_filename = f"{book_code}_{chapter:02d}.mp3"
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

    def _create_temp_text_file(self, text: str) -> str:
        """Create temporary text file for alignment"""
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False) as f:
            f.write(text)
            return f.name

    async def _transcribe_with_batch(self, audio_file: Path, reference_text: str) -> Optional[Dict]:
        """Use Azure Batch Transcription for large-scale processing"""
        if not self.blob_service_client:
            logger.warning("No Blob Storage configured; falling back to local transcription")
            return await self._transcribe_with_timestamps(audio_file, reference_text)

        try:
            # Upload audio to Blob Storage
            blob_name = f"audio/{audio_file.name}"
            blob_client = self.container_client.get_blob_client(blob_name)
            with open(audio_file, 'rb') as f:
                blob_client.upload_blob(f, overwrite=True)
            audio_url = blob_client.url

            # Configure Batch Transcription
            transcription_definition = speechsdk.transcription.ConversationTranscriptionDefinition(
                audio_url=audio_url,
                language="he-IL",
                word_level_timestamps_enabled=True,
                output_format=speechsdk.OutputFormat.Detailed
            )

            # Submit transcription job
            transcriber = speechsdk.transcription.ConversationTranscriber(self.speech_config)
            transcription = transcriber.create_transcription(transcription_definition)

            # Poll for completion
            max_attempts = 60
            attempt = 0
            while attempt < max_attempts:
                status = transcriber.get_transcription_status(transcription)
                if status.status == speechsdk.TranscriptionStatus.Completed:
                    break
                elif status.status == speechsdk.TranscriptionStatus.Failed:
                    logger.error(f"Batch transcription failed for {audio_file.name}: {status.error_details}")
                    return None
                await asyncio.sleep(5)
                attempt += 1

            if attempt >= max_attempts:
                logger.error(f"Batch transcription timeout for {audio_file.name}")
                return None

            # Retrieve results
            results = transcriber.get_transcription_results(transcription)
            return self._process_transcription_results(results, reference_text)

        except Exception as e:
            logger.error(f"Batch transcription error for {audio_file}: {e}")
            return None

    async def _transcribe_with_timestamps(self, audio_file: Path, reference_text: str) -> Optional[Dict]:
        """Transcribe audio file and get word-level timestamps (local fallback)"""
        try:
            audio_config = AudioConfig(filename=str(audio_file))
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            transcription_results = []

            def handle_final_result(evt: speechsdk.SpeechRecognitionEventArgs):
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    result_json = json.loads(evt.result.json)
                    transcription_results.append(result_json)
                    logger.debug(f"Recognized: {evt.result.text}")
                elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                    logger.warning(f"No speech recognized: {evt.result.no_match_details}")

            def handle_canceled(evt: speechsdk.SpeechRecognitionCanceledEventArgs):
                logger.error(f"Recognition canceled: {evt.cancellation_details}")

            speech_recognizer.recognized.connect(handle_final_result)
            speech_recognizer.canceled.connect(handle_canceled)

            logger.info(f"Starting transcription for {audio_file.name}")
            speech_recognizer.start_continuous_recognition()

            max_wait_time = 300
            wait_time = 0
            while len(transcription_results) == 0 and wait_time < max_wait_time:
                await asyncio.sleep(1)
                wait_time += 1

            speech_recognizer.stop_continuous_recognition()

            if not transcription_results:
                logger.warning(f"No transcription results for {audio_file.name}")
                return None

            return self._process_transcription_results(transcription_results, reference_text)

        except Exception as e:
            logger.error(f"Transcription error for {audio_file}: {e}")
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
                    if similarity >= 0.7:
                        aligned_word = {
                            'text': ref_word,
                            'start': transcribed_word['start'],
                            'end': transcribed_word['end'],
                            'confidence': transcribed_word['confidence']
                        }
                        aligned_words.append(aligned_word)
                        if verse_start_time is None:
                            verse_start_time = transcribed_word['start']
                        verse_end_time = transcribed_word['end']
                        transcription_index += 1
                    else:
                        estimated_duration = 0.5
                        estimated_start = verse_end_time or 0.0
                        aligned_word = {
                            'text': ref_word,
                            'start': estimated_start,
                            'end': estimated_start + estimated_duration,
                            'confidence': 0.1
                        }
                        aligned_words.append(aligned_word)
                        verse_end_time = estimated_start + estimated_duration
                else:
                    estimated_duration = 0.5
                    estimated_start = verse_end_time or 0.0
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
                'start': verse_start_time or 0.0,
                'end': verse_end_time or 0.0,
                'words': aligned_words,
                'word_count': len(aligned_words)
            }
            aligned_verses.append(aligned_verse)

        return aligned_verses

    def _normalize_hebrew_text(self, text: str) -> str:
        """Normalize Hebrew text for comparison"""
        nikkud_pattern = r'[\u0591-\u05C7]'
        normalized = re.sub(nikkud_pattern, '', text)
        normalized = re.sub(r'[^\u05D0-\u05EA\s]', '', normalized)
        return ' '.join(normalized.split()).strip()

    def _validate_output(self, output_data: Dict, audio_file: Path) -> bool:
        """Validate alignment output against audio duration"""
        try:
            import librosa
            audio_duration = librosa.get_duration(filename=str(audio_file))
            if output_data['total_duration'] > audio_duration * 1.1:
                logger.warning(f"Alignment duration {output_data['total_duration']} exceeds audio duration {audio_duration}")
                return False
            return True
        except Exception as e:
            logger.error(f"Validation error for {audio_file}: {e}")
            return False

    async def process_chapter(self, book_code: str, chapter_num: int) -> bool:
        """Process a single chapter"""
        try:
            logger.info(f"Processing {book_code} Chapter {chapter_num}")
            audio_file = self._get_audio_filename(book_code, chapter_num)
            if not audio_file.exists():
                logger.error(f"Audio file not found: {audio_file}")
                return False

            chapter_text, verse_metadata = self._extract_chapter_text(book_code, chapter_num)
            transcription_result = await self._transcribe_with_batch(audio_file, chapter_text)

            if not transcription_result:
                logger.error(f"Transcription failed for {book_code} Ch{chapter_num}")
                return False

            if transcription_result['confidence'] < 0.3:
                logger.warning(f"Low confidence transcription: {transcription_result['confidence']}")

            aligned_verses = self._align_words_to_reference(transcription_result['words'], verse_metadata)

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
                    'transcribed_word_count': transcription_result['word_count']
                }
            }

            if not self._validate_output(output_data, audio_file):
                logger.warning(f"Validation failed for {book_code} Ch{chapter_num}")
                return False

            book_output_dir = self.output_base_dir / book_code
            book_output_dir.mkdir(exist_ok=True)
            output_filename = f"alignment_{book_code}_{chapter_num:02d}.json"
            output_path = book_output_dir / output_filename

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Successfully processed {book_code} Ch{chapter_num} -> {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error processing {book_code} Ch{chapter_num}: {e}")
            return False

    async def process_all_chapters(self, max_concurrent: int = 5) -> Dict[str, int]:
        """Process all chapters with concurrency control"""
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = []
        total_chapters = 0

        async def process_with_semaphore(book_code: str, chapter_num: int):
            async with semaphore:
                return await self.process_chapter(book_code, chapter_num)

        for book_code, chapters in self.bible_data.items():
            for chapter_num in range(1, len(chapters) + 1):
                task = process_with_semaphore(book_code, chapter_num)
                tasks.append((book_code, chapter_num, task))
                total_chapters += 1

        logger.info(f"Starting processing of {total_chapters} chapters with max {max_concurrent} concurrent")
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
                    logger.info(f"Progress: {processed}/{total_chapters} chapters processed")
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

    async def process_book(self, book_code: str) -> bool:
        """Process all chapters in a specific book"""
        if book_code not in self.bible_data:
            logger.error(f"Book '{book_code}' not found in Bible data")
            return False

        chapters = self.bible_data[book_code]
        success_count = 0

        for chapter_num in range(1, len(chapters) + 1):
            success = await self.process_chapter(book_code, chapter_num)
            if success:
                success_count += 1

        logger.info(f"Book {book_code}: {success_count}/{len(chapters)} chapters processed successfully")
        return success_count == len(chapters)

async def main():
    """Main execution function"""
    try:
        aligner = HebrewBibleAligner()
        # Process all chapters
        summary = await aligner.process_all_chapters(max_concurrent=5)
        print(f"Processing summary: {summary}")

    except Exception as e:
        logger.error(f"Main execution error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())