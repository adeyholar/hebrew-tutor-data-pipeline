# File location: D:\AI\Gits\hebrew-tutor-data-pipeline\scripts\hebrew_alignment_tool.py
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
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import re
from difflib import SequenceMatcher
from dotenv import load_dotenv

# Azure Speech SDK imports - install with: pip install azure-cognitiveservices-speech
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech import SpeechConfig, AudioConfig, ResultReason, CancellationReason

# Load .env file
load_dotenv(Path('D:/AI/Gits/hebrew-tutor-data-pipeline/.env'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hebrew_alignment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Mapping of book codes to audio file names
BOOK_CODE_TO_FILE_MAP = {
    'Gen': '01_genesis.mp3',
    'Exod': '02_exodus.mp3',
    'Lev': '03_leviticus.mp3',
    'Num': '04_numbers.mp3',
    'Deut': '05_deuteronomy.mp3',
    'Josh': '06_joshua.mp3',
    'Judg': '07_judges.mp3',
    'Ruth': '08_ruth.mp3',
    '1Sam': '09_1-samuel.mp3',
    '2Sam': '10_2-samuel.mp3',
    '1Kgs': '11_1-kings.mp3',
    '2Kgs': '12_2-kings.mp3',
    '1Chr': '13_1-chronicles.mp3',
    '2Chr': '14_2-chronicles.mp3',
    'Ezra': '15_ezra.mp3',
    'Neh': '16_nehemiah.mp3',
    'Esth': '17_esther.mp3',
    'Job': '18_job.mp3',
    'Ps': '19_psalms.mp3',
    'Prov': '20_proverbs.mp3',
    'Eccl': '21_ecclesiastes.mp3',
    'Song': '22_songofsongs.mp3',
    'Isa': '23_isaiah.mp3',
    'Jer': '24_jeremiah.mp3',
    'Lam': '25_lamentations.mp3',
    'Ezek': '26_ezekiel.mp3',
    'Dan': '27_daniel.mp3',
    'Hos': '28_hosea.mp3',
    'Joel': '29_joel.mp3',
    'Amos': '30_amos.mp3',
    'Obad': '31_obadiah.mp3',
    'Jonah': '32_jonah.mp3',
    'Mic': '33_micah.mp3',
    'Nah': '34_nahum.mp3',
    'Hab': '35_habakkuk.mp3',
    'Zeph': '36_zephaniah.mp3',
    'Hag': '37_haggai.mp3',
    'Zech': '38_zechariah.mp3',
    'Mal': '39_malachi.mp3'
}

class HebrewBibleAligner:
    """Main class for Hebrew Bible audio-text alignment using Azure Speech SDK"""
    
    def __init__(self):
        """Initialize the aligner with Azure credentials and paths"""
        self.subscription_key = os.getenv('AZURE_SPEECH_KEY')
        self.region = os.getenv('AZURE_SPEECH_REGION', 'eastus')
        
        if not self.subscription_key:
            raise ValueError("AZURE_SPEECH_KEY environment variable not set")
        
        self.speech_config = SpeechConfig(
            subscription=self.subscription_key, 
            region=self.region
        )
        self.speech_config.speech_recognition_language = "he-IL"
        self.speech_config.request_word_level_timestamps()
        self.speech_config.output_format = speechsdk.OutputFormat.Detailed
        
        self.json_path = Path('D:/AI/Projects/HEBREW TRAINING AI AGENT/TANACH/book/hebrew_bible_with_nikkud.json')
        self.audio_dir = Path('D:/audio/tanakh_audio')
        self.output_dir = Path('D:/AI/Projects/HEBREW TRAINING AI AGENT/TANACH/alignment')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
    
    def _get_audio_filename(self, book_code: str) -> Path:
        """Get audio filename for given book"""
        filename = BOOK_CODE_TO_FILE_MAP.get(book_code)
        if not filename:
            raise ValueError(f"No audio file mapping for book code: {book_code}")
        return self.audio_dir / filename
    
    def _extract_chapter_text(self, book_code: str, chapter_num: int) -> Tuple[str, List[Dict]]:
        """
        Extract text for a specific chapter and return full text + verse structure
        """
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
    
    async def _transcribe_with_timestamps(self, audio_file: Path, reference_text: str) -> Optional[Dict]:
        """
        Transcribe audio file and get word-level timestamps using Azure Speech SDK
        """
        try:
            audio_config = AudioConfig(filename=str(audio_file))
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            logger.info(f"Starting transcription for {audio_file.name}")
            done = False
            transcription_results = []
            
            def stop_cb(evt: speechsdk.SessionEventArgs):
                nonlocal done
                speech_recognizer.stop_continuous_recognition()
                done = True
            
            def handle_result(evt: speechsdk.SpeechRecognitionEventArgs):
                if evt.result.reason == ResultReason.RecognizedSpeech:
                    result_json = json.loads(evt.result.json)
                    transcription_results.append(result_json)
                    logger.debug(f"Recognized: {evt.result.text}")
                elif evt.result.reason == ResultReason.NoMatch:
                    logger.warning(f"No speech recognized")
            
            def handle_cancel(evt: speechsdk.SpeechRecognitionCanceledEventArgs):
                if evt.reason == CancellationReason.Error:
                    logger.error(f"Transcription error: {evt.error_details}")
                
            speech_recognizer.recognized.connect(handle_result)
            speech_recognizer.canceled.connect(handle_cancel)
            speech_recognizer.session_stopped.connect(stop_cb)
            
            speech_recognizer.start_continuous_recognition()
            
            while not done:
                await asyncio.sleep(0.5)
            
            if not transcription_results:
                logger.warning(f"No transcription results for {audio_file.name}")
                return None
            
            combined_result = self._process_transcription_results(
                transcription_results, reference_text
            )
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Transcription error for {audio_file}: {e}")
            return None
    
    def _process_transcription_results(self, results: List[Dict], reference_text: str) -> Dict:
        """
        Process Azure transcription results to extract word-level timestamps
        """
        processed_words = []
        total_confidence = 0
        word_count = 0
        
        for result in results:
            if 'NBest' in result and result['NBest']:
                best_result = result['NBest'][0]
                
                if 'Words' in best_result:
                    for word_info in best_result['Words']:
                        processed_word = {
                            'text': word_info.get('Word', ''),
                            'start': word_info.get('Offset', 0) / 10000000.0,
                            'end': (word_info.get('Offset', 0) + word_info.get('Duration', 0)) / 10000000.0,
                            'confidence': word_info.get('Confidence', 0.0)
                        }
                        processed_words.append(processed_word)
                        total_confidence += processed_word['confidence']
                        word_count += 1
        
        average_confidence = total_confidence / word_count if word_count > 0 else 0.0
        
        return {
            'words': processed_words,
            'confidence': average_confidence,
            'word_count': word_count,
            'total_duration': processed_words[-1]['end'] if processed_words else 0.0
        }
    
    def _align_words_to_reference(self, transcribed_words: List[Dict], 
                                 verse_metadata: List[Dict]) -> List[Dict]:
        """
        Align transcribed words with reference text using fuzzy matching
        """
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
                    
                    if self._words_similar(normalized_ref, normalized_trans):
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
        """Normalize Hebrew text for comparison (remove nikkud, etc.)"""
        nikkud_pattern = r'[\u0591-\u05C7]'
        normalized = re.sub(nikkud_pattern, '', text)
        normalized = re.sub(r'[^\u05D0-\u05EA\s]', '', normalized)
        normalized = ' '.join(normalized.split())
        return normalized.strip()
    
    def _words_similar(self, word1: str, word2: str, threshold: float = 0.7) -> bool:
        """Check if two Hebrew words are similar enough for alignment"""
        normalized_ref = self._normalize_hebrew_text(word1)
        normalized_trans = self._normalize_hebrew_text(word2)
        similarity = SequenceMatcher(None, normalized_ref, normalized_trans).ratio()
        return similarity >= threshold
    
    async def process_chapter(self, book_code: str, chapter_num: int) -> bool:
        """
        Process a single chapter: transcribe audio and align with text
        """
        try:
            logger.info(f"Processing {book_code} Chapter {chapter_num}")
            audio_file = self._get_audio_filename(book_code)
            if not audio_file.exists():
                logger.error(f"Audio file not found: {audio_file}")
                return False
            
            chapter_text, verse_metadata = self._extract_chapter_text(book_code, chapter_num)
            transcription_result = await self._transcribe_with_timestamps(audio_file, chapter_text)
            
            if not transcription_result:
                logger.error(f"Transcription failed for {book_code} Ch{chapter_num}")
                return False
            
            if transcription_result['confidence'] < 0.3:
                logger.warning(f"Low confidence transcription: {transcription_result['confidence']}")
                total_duration = transcription_result['total_duration']
                verse_count = len(verse_metadata)
                verse_duration = total_duration / verse_count if verse_count > 0 else 0.5
                aligned_verses = []
                for i, verse_meta in enumerate(verse_metadata):
                    start_time = i * verse_duration
                    aligned_verse = {
                        'verse_num': verse_meta['verse_num'],
                        'text': verse_meta['text'],
                        'start': start_time,
                        'end': start_time + verse_duration,
                        'words': [{'text': w, 'start': start_time, 'end': start_time + verse_duration/len(verse_meta['words']), 'confidence': 0.1} for w in verse_meta['words']],
                        'word_count': len(verse_meta['words'])
                    }
                    aligned_verses.append(aligned_verse)
            else:
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
            
            output_filename = f"alignment_{book_code}_{chapter_num:02d}.json"
            output_path = self.output_dir / output_filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Successfully processed {book_code} Ch{chapter_num} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {book_code} Ch{chapter_num}: {e}")
            return False
    
    async def process_book(self, book_code: str) -> bool:
        """
        Process all chapters in a specific book
        """
        try:
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
            
        except Exception as e:
            logger.error(f"Error processing book {book_code}: {e}")
            return False
    
    async def process_all_chapters(self, max_concurrent: int = 2) -> Dict[str, int]:
        """
        Process all chapters in the Bible with concurrency control
        """
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

async def main():
    """Main execution function"""
    try:
        aligner = HebrewBibleAligner()
        # Test with single book (1 Chronicles)
        success = await aligner.process_book('1Chr')
        print(f"Book processing: {'Success' if success else 'Failed'}")
        # Uncomment for full processing
        # summary = await aligner.process_all_chapters(max_concurrent=2)
        # print(f"Processing summary: {summary}")
    except Exception as e:
        logger.error(f"Main execution error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())