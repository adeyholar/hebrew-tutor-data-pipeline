# File: D:\AI\Gits\hebrew-tutor-data-pipeline\scripts\hebrew_alignment_tool.py
#!/usr/bin/env python3
"""
Hebrew Bible Audio-Text Alignment Tool
Uses CPU-based OpenAI Whisper for primary alignment, fallback to Azure Speech
Leverages existing local and Blob storage
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
from azure.storage.blob import BlobServiceClient
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech import SpeechConfig, AudioConfig
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import sys
import librosa
import soundfile as sf  # For WAV writing

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

# Force CPU (as per request)
device = torch.device("cpu")
logger.info(f"Using device: {device}")

# COMPLETE BOOK MAPPING - matches local and Blob structure
COMPLETE_BOOK_MAPPING = {
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
    '2Chr': {'folder': '39_2Chronicles', 'abbr': '2Chron'}
}

class HebrewBibleAligner:
    def __init__(self):
        self.subscription_key = os.getenv('AZURE_SPEECH_KEY')
        self.region = os.getenv('AZURE_SPEECH_REGION', 'eastus')
        self.storage_connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        
        if not self.subscription_key:
            logger.warning("AZURE_SPEECH_KEY not set; using Whisper only")

        self.speech_config = None
        if self.subscription_key:
            self.speech_config = SpeechConfig(
                subscription=self.subscription_key,
                region=self.region
            )
            self.speech_config.speech_recognition_language = "he-IL"
            self.speech_config.request_word_level_timestamps()
            self.speech_config.output_format = speechsdk.OutputFormat.Detailed

        self.blob_service_client = None
        self.container_client = None
        if self.storage_connection_string:
            try:
                self.blob_service_client = BlobServiceClient.from_connection_string(self.storage_connection_string)
                self.container_client = self.blob_service_client.get_container_client("hebrew-audio")
            except Exception as e:
                logger.info(f"Blob connection failed: {e}; using local audio")

        self.json_path = Path(os.getenv('HEBREW_BIBLE_JSON_PATH', 'D:/AI/Projects/HEBREW-TRAINING-AI-AGENT/TANACH/book/hebrew_bible_with_nikkud.json'))
        self.audio_base_dir = Path(os.getenv('AUDIO_BASE_DIR', 'D:/audio/tankh_audio_chp'))
        self.output_base_dir = Path(os.getenv('OUTPUT_BASE_DIR', 'D:/AI/Projects/HEBREW-TRAINING-AI-AGENT/TANACH/alignment'))

        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        self.bible_data = self._load_bible_data()
        logger.info(f"Initialized Hebrew Bible Aligner for {len(self.bible_data)} books")

        # Initialize Whisper with CPU
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small").to(device)
        self.processor = AutoProcessor.from_pretrained("openai/whisper-small")
        self.whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,  # Explicit tokenizer
            feature_extractor=self.processor.feature_extractor,
            device=device,
            batch_size=4  # Reduced batch_size for CPU
        )

    def _load_bible_data(self) -> Dict[str, List[List[List[str]]]]:
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'Aramaic' in data:
                del data['Aramaic']
            logger.info(f"Loaded Bible data with {len(data)} books")
            return data
        except FileNotFoundError:
            logger.error(f"Bible data file not found: {self.json_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in Bible data file: {e}")
            raise

    def _get_audio_filename(self, book_code: str, chapter: int) -> Path:
        if book_code not in COMPLETE_BOOK_MAPPING:
            raise ValueError(f"No mapping for book code {book_code}")
        mapping = COMPLETE_BOOK_MAPPING[book_code]
        folder_name = mapping['folder']
        abbr = mapping['abbr']
        num = folder_name.split('_')[0]
        audio_filename = f"{num}-{abbr}_{chapter:02d}.mp3"
        local_path = self.audio_base_dir / folder_name / audio_filename
        
        if local_path.exists():
            return local_path
        
        if self.container_client:
            blob_name = f"audio/{audio_filename}"
            blob_client = self.container_client.get_blob_client(blob_name)
            if blob_client.exists():
                download_path = tempfile.mktemp(suffix='.mp3')
                with open(download_path, "wb") as f:
                    f.write(blob_client.download_blob().readall())
                logger.info(f"Downloaded {audio_filename} from Blob")
                return Path(download_path)
        
        raise FileNotFoundError(f"Audio not found locally or in Blob: {audio_filename}")

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

    def _convert_to_wav(self, audio_file: Path) -> Path:
        """Convert MP3 to WAV for Azure compatibility"""
        wav_path = tempfile.mktemp(suffix='.wav')
        y, sr = librosa.load(str(audio_file), sr=16000)  # Azure prefers 16kHz
        sf.write(wav_path, y, sr)  # Use soundfile for writing
        logger.info(f"Converted {audio_file} to WAV for Azure")
        return Path(wav_path)

    async def _transcribe(self, audio_file: Path, reference_text: str) -> Optional[Dict]:
        """Primary: CPU Whisper, Fallback: Azure"""
        try:
            # Whisper transcription with CPU
            result = self.whisper_pipe(str(audio_file), return_timestamps="word", chunk_length_s=20, batch_size=4)  # Reduced for CPU
            processed_words = []
            total_confidence = 0
            word_count = 0
            total_duration = 0

            for chunk in result['chunks']:
                for w in chunk['text'].split():  # Refine splitting if needed
                    processed_word = {
                        'text': w,
                        'start': chunk['timestamp'][0] or 0.0,  # Handle None timestamps
                        'end': chunk['timestamp'][1] or 0.5,  # Handle None timestamps
                        'confidence': 0.8  # Placeholder
                    }
                    processed_words.append(processed_word)
                    total_confidence += processed_word['confidence']
                    word_count += 1
                    total_duration = max(total_duration, processed_word['end'])

            average_confidence = total_confidence / word_count if word_count > 0 else 0.0

            return {
                'words': processed_words,
                'confidence': average_confidence,
                'word_count': word_count,
                'total_duration': total_duration
            }
        except Exception as e:
            logger.warning(f"Whisper failed: {e}. Falling back to Azure.")
            if self.speech_config:
                return await self._transcribe_with_azure(audio_file, reference_text)
            return None

    async def _transcribe_with_azure(self, audio_file: Path, reference_text: str) -> Optional[Dict]:
        try:
            wav_file = self._convert_to_wav(audio_file)
            audio_config = AudioConfig(filename=str(wav_file))
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

            logger.info(f"Starting Azure transcription for {wav_file.name}")
            speech_recognizer.start_continuous_recognition()

            max_wait_time = 300
            wait_time = 0
            while len(transcription_results) == 0 and wait_time < max_wait_time:
                await asyncio.sleep(1)
                wait_time += 1

            speech_recognizer.stop_continuous_recognition()

            if not transcription_results:
                logger.warning(f"No transcription results for {wav_file.name}")
                return None

            return self._process_transcription_results(transcription_results, reference_text)

        except Exception as e:
            logger.error(f"Azure transcription error for {audio_file}: {e}")
            return None
        finally:
            if 'wav_file' in locals():
                os.remove(str(wav_file))  # Clean up WAV

    def _process_transcription_results(self, results: Any, reference_text: str) -> Dict:
        processed_words = []
        total_confidence = 0
        word_count = 0
        total_duration = 0

        if isinstance(results, list):
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
        nikkud_pattern = r'[\u0591-\u05C7]'
        normalized = re.sub(nikkud_pattern, '', text)
        normalized = re.sub(r'[^\u05D0-\u05EA\s]', '', normalized)
        return ' '.join(normalized.split()).strip()

    def _validate_output(self, output_data: Dict, audio_file: Path) -> bool:
        try:
            import librosa
            audio_duration = librosa.get_duration(path=str(audio_file))
            if output_data['total_duration'] > audio_duration * 1.1:
                logger.warning(f"Alignment duration {output_data['total_duration']} exceeds audio duration {audio_duration}")
                return False
            return True
        except Exception as e:
            logger.error(f"Validation error for {audio_file}: {e}")
            return False

    async def process_chapter(self, book_code: str, chapter_num: int) -> bool:
        try:
            logger.info(f"Processing {book_code} Chapter {chapter_num}")
            audio_file = self._get_audio_filename(book_code, chapter_num)
            chapter_text, verse_metadata = self._extract_chapter_text(book_code, chapter_num)
            transcription_result = await self._transcribe(audio_file, chapter_text)

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
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = []
        total_chapters = 0

        async def process_with_semaphore(book_code: str, chapter_num: int):
            async with semaphore:
                return await self.process_chapter(book_code, chapter_num)

        for book_code in self.bible_data:
            chapters = self.bible_data[book_code]
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
    try:
        aligner = HebrewBibleAligner()
        if len(sys.argv) > 1:
            command = sys.argv[1]
            if command == "test" and len(sys.argv) >= 4:
                book = sys.argv[2]
                chapter = int(sys.argv[3])
                result = await aligner.process_chapter(book, chapter)  # Direct await for single chapter
                print(f"Test result for {book} Ch{chapter}: {result}")
            else:
                print("Usage: python script.py test <book> <chapter>")
        else:
            summary = await aligner.process_all_chapters(max_concurrent=5)
            print(f"Processing summary: {summary}")

    except Exception as e:
        logger.error(f"Main execution error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())