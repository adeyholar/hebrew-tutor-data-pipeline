# Hebrew Tutor App - Audio-Text Force Alignment Pipeline (CORRECTED PATHS)
# Fixed audio folder mapping to match actual directory structure

import os
import sys
import json
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import traceback

# Dependencies
import librosa

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging for Hebrew text
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('hebrew_alignment.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class HebrewAlignmentTool:
    def __init__(self):
        """Initialize the Hebrew Alignment Tool with corrected audio paths."""
        logger.info("=== INITIALIZING HEBREW ALIGNMENT TOOL (CORRECTED PATHS) ===")
        
        # Paths
        self.base_path = Path("D:/AI/Projects/HEBREW-TRAINING-AI-AGENT/TANACH")
        self.audio_local_path = Path("D:/audio/tankh_audio_chp")
        self.json_path = self.base_path / "book" / "hebrew_bible_with_nikkud.json"
        self.output_path = self.base_path / "alignment"
        
        # Audio processing settings
        self.sample_rate = 16000
        
        # Load Bible data
        self.bible_data = self._load_bible_data()
        
        logger.info("=== INITIALIZATION COMPLETE ===")
    
    def _load_bible_data(self) -> Dict:
        """Load the Hebrew Bible JSON data with proper UTF-8 encoding."""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, dict):
                raise ValueError("Bible data must be a dictionary")
                
            total_chapters = sum(len(chapters) for chapters in data.values())
            logger.info(f"Loaded Bible data: {len(data)} books, {total_chapters} chapters")
            
            # Test Hebrew text
            if 'Gen' in data and len(data['Gen']) > 0 and len(data['Gen'][0]) > 0:
                first_verse = data['Gen'][0][0]
                if isinstance(first_verse, list):
                    first_verse_text = ' '.join(first_verse)
                else:
                    first_verse_text = str(first_verse)
                
                hebrew_chars = sum(1 for char in first_verse_text if '\u0590' <= char <= '\u05FF')
                if hebrew_chars > 0:
                    logger.info(f"Hebrew encoding verified: Genesis 1:1 has {hebrew_chars} Hebrew characters")
                else:
                    logger.warning("Hebrew characters not detected")
            
            return data
        except Exception as e:
            logger.error(f"Failed to load Bible data: {e}")
            return {}
    
    def get_audio_file_path(self, book_code: str, chapter: int) -> Optional[Path]:
        """Get the local audio file path with CORRECTED folder mappings."""
        
        # CORRECTED book folder mappings based on actual directory structure
        book_folders = {
            'Gen': '01_Genesis', 'Exod': '02_Exodus', 'Lev': '03_Leviticus',
            'Num': '04_Numbers', 'Deut': '05_Deuteronomy', 'Josh': '06_Joshua',
            'Judg': '07_Judges', 'Ruth': '30_Ruth', '1Sam': '08_1Samuel',  # FIXED Ruth!
            '2Sam': '09_2Samuel', '1Kgs': '10_1Kings', '2Kgs': '11_2Kings',
            '1Chr': '38_1Chronicles', '2Chr': '39_2Chronicles',  # CORRECTED!
            'Ezra': '36_Ezra', 'Neh': '37_Nehemiah', 'Esth': '34_Esther',  # CORRECTED!
            'Job': '28_Job', 'Ps': '27_Psalms', 'Prov': '29_Proverbs',  # CORRECTED!
            'Eccl': '32_Ecclesiastes', 'Song': '31_SongofSongs',  # CORRECTED!
            'Isa': '12_Isaiah', 'Jer': '13_Jeremiah', 'Lam': '33_Lamentations',  # CORRECTED!
            'Ezek': '14_Ezekiel', 'Dan': '35_Daniel',  # CORRECTED!
            'Hos': '15_Hosea', 'Joel': '16_Joel', 'Amos': '17_Amos',
            'Obad': '18_Obadiah', 'Jonah': '19_Jonah', 'Mic': '20_Micah',
            'Nah': '21_Nahum', 'Hab': '22_Habakkuk', 'Zeph': '23_Zephaniah',
            'Hag': '24_Haggai', 'Zech': '25_Zechariah', 'Mal': '26_Malachi'
        }
        
        # Audio filename mappings
        audio_codes = {
            'Gen': 'Gen', 'Exod': 'Exo', 'Lev': 'Lev', 'Num': 'Num', 'Deut': 'Deut',
            'Josh': 'Josh', 'Judg': 'Judg', 'Ruth': 'Ruth', '1Sam': '1Sam', '2Sam': '2Sam',
            '1Kgs': '1Kgs', '2Kgs': '2Kgs', '1Chr': '1Chron', '2Chr': '2Chron',  # CORRECTED!
            'Ezra': 'Ezra', 'Neh': 'Nehem', 'Esth': 'Esther', 'Job': 'Job',  # CORRECTED!
            'Ps': 'Psa', 'Prov': 'Prov', 'Eccl': 'Eccles', 'Song': 'Song',  # CORRECTED!
            'Isa': 'Isa', 'Jer': 'Jer', 'Lam': 'Lamen', 'Ezek': 'Ezek', 'Dan': 'Daniel',  # CORRECTED!
            'Hos': 'Hosea', 'Joel': 'Joel', 'Amos': 'Amos', 'Obad': 'Obad', 'Jonah': 'Jonah',
            'Mic': 'Micha', 'Nah': 'Nahum', 'Hab': 'Habak', 'Zeph': 'Zeph', 'Hag': 'Haggai',
            'Zech': 'Zech', 'Mal': 'Malachi'
        }
        
        if book_code not in book_folders or book_code not in audio_codes:
            logger.error(f"Unknown book code: {book_code}")
            return None
            
        folder_name = book_folders[book_code]
        audio_code = audio_codes[book_code]
        book_num = folder_name.split('_')[0]
        
        # Special case for Obadiah (single chapter book with no chapter number in filename)
        if book_code == 'Obad':
            filename = f"{book_num}-{audio_code}.mp3"
        # Handle special cases for Psalms (3-digit numbering)
        elif book_code == 'Ps':
            filename = f"{book_num}-{audio_code}_{chapter:03d}.mp3"
        else:
            filename = f"{book_num}-{audio_code}_{chapter:02d}.mp3"
        
        file_path = self.audio_local_path / folder_name / filename
        logger.debug(f"Looking for audio file: {file_path}")
        return file_path
    
    def validate_audio_file(self, audio_path: Path) -> Tuple[bool, float]:
        """Validate audio file."""
        try:
            if not audio_path.exists():
                logger.error(f"Audio file does not exist: {audio_path}")
                return False, 0.0
                
            file_size = audio_path.stat().st_size
            if file_size < 1000:
                logger.error(f"Audio file too small: {file_size} bytes")
                return False, 0.0
            
            duration = librosa.get_duration(path=str(audio_path))
            if duration > 0:
                logger.debug(f"Audio file valid: {duration:.2f}s, {file_size/1024:.1f}KB")
                return True, duration
            return False, 0.0
        except Exception as e:
            logger.error(f"Audio validation failed: {e}")
            return False, 0.0
    
    def extract_verses(self, book_code: str, chapter: int) -> List[Dict]:
        """Extract verses with Hebrew text preservation."""
        try:
            if book_code not in self.bible_data:
                logger.error(f"Book '{book_code}' not found in Bible data")
                return []
            
            book_data = self.bible_data[book_code]
            if not isinstance(book_data, list) or chapter < 1 or chapter > len(book_data):
                logger.error(f"Invalid chapter {chapter} for book {book_code} (max: {len(book_data)})")
                return []
            
            chapter_data = book_data[chapter - 1]
            if not isinstance(chapter_data, list):
                logger.error(f"Invalid chapter data format")
                return []
            
            # Convert word arrays to verse text preserving Hebrew
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
            
            if formatted_verses:
                first_verse = formatted_verses[0]['text']
                hebrew_chars = sum(1 for char in first_verse if '\u0590' <= char <= '\u05FF')
                logger.debug(f"Extracted {len(formatted_verses)} verses, first verse has {hebrew_chars} Hebrew chars")
            
            return formatted_verses
            
        except Exception as e:
            logger.error(f"Failed to extract verses: {e}")
            return []
    
    def create_estimated_alignment(self, verses: List[Dict], duration: float) -> Dict:
        """Create estimated alignment preserving Hebrew text."""
        if not verses:
            return {}
            
        logger.debug("Creating estimated alignment with Hebrew preservation")
        
        aligned_verses = []
        total_words = sum(len(verse.get('text', '').split()) for verse in verses)
        
        current_time = 0.0
        for i, verse in enumerate(verses):
            verse_text = verse.get('text', '')
            words = verse_text.split()
            word_count = len(words)
            
            # Allocate time based on word density
            if total_words > 0:
                verse_duration = (word_count / total_words) * duration
            else:
                verse_duration = duration / len(verses)
            
            verse_duration = max(verse_duration, 0.5)  # Minimum 0.5 seconds per verse
            
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
                        'confidence': 0.8
                    })
            
            aligned_verses.append({
                'verse_number': verse.get('verse_number', i + 1),
                'text': verse_text,
                'start': round(start_time, 2),
                'end': round(end_time, 2),
                'confidence': 0.8,
                'words': word_alignments
            })
        
        return {
            'total_duration': round(duration, 2),
            'overall_confidence': 0.8,
            'verse_count': len(verses),
            'verses': aligned_verses,
            'metadata': {
                'method': 'estimated_hebrew_preserved',
                'model': 'hebrew_preservation_corrected_paths',
                'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'note': 'Hebrew text preserved with corrected audio file paths'
            }
        }
    
    def save_alignment_json(self, book_code: str, chapter: int, alignment_data: Dict) -> bool:
        """Save alignment data with Hebrew preservation."""
        try:
            output_dir = self.output_path / book_code
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"alignment_{book_code}_{chapter:02d}.json"
            output_file = output_dir / filename
            
            # Write with UTF-8 encoding and ensure_ascii=False
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(alignment_data, f, indent=2, ensure_ascii=False)
            
            # Verify Hebrew preservation
            with open(output_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            if test_data.get('verses') and len(test_data['verses']) > 0:
                first_verse_text = test_data['verses'][0].get('text', '')
                hebrew_chars = sum(1 for char in first_verse_text if '\u0590' <= char <= '\u05FF')
                if hebrew_chars > 0:
                    logger.debug(f"Hebrew preserved in saved file: {hebrew_chars} chars")
                else:
                    logger.error("Hebrew text lost during save!")
                    return False
            
            logger.debug(f"Alignment saved: {output_file}")
            return True
                    
        except Exception as e:
            logger.error(f"Failed to save alignment: {e}")
            return False
    
    async def process_chapter(self, book_code: str, chapter: int) -> bool:
        """Process a single chapter with Hebrew preservation."""
        logger.info(f"Processing {book_code} Chapter {chapter}")
        
        try:
            # Extract verses with Hebrew text
            verses = self.extract_verses(book_code, chapter)
            if not verses:
                logger.error(f"No verses found for {book_code} {chapter}")
                return False
            
            # Validate audio file
            audio_path = self.get_audio_file_path(book_code, chapter)
            if not audio_path:
                logger.error(f"Could not determine audio path for {book_code} {chapter}")
                return False
                
            if not audio_path.exists():
                logger.error(f"Audio file not found: {audio_path}")
                return False
            
            is_valid, duration = self.validate_audio_file(audio_path)
            if not is_valid:
                logger.error(f"Invalid audio file: {audio_path}")
                return False
            
            # Create alignment with Hebrew preservation
            logger.debug("Creating alignment with Hebrew text preservation")
            alignment_data = self.create_estimated_alignment(verses, duration)
            
            if alignment_data:
                save_success = self.save_alignment_json(book_code, chapter, alignment_data)
                if save_success:
                    logger.info(f"✅ SUCCESS: {book_code} {chapter} processed ({duration:.1f}s, {len(verses)} verses)")
                    return True
            
            logger.error(f"Failed to process chapter {book_code} {chapter}")
            return False
            
        except Exception as e:
            logger.error(f"Error processing chapter {book_code} {chapter}: {e}")
            return False

    async def process_book(self, book_code: str) -> bool:
        """Process all chapters in a book."""
        logger.info(f"📚 Processing entire book: {book_code}")
        
        if book_code not in self.bible_data:
            logger.error(f"Book '{book_code}' not found")
            return False
        
        book_data = self.bible_data[book_code]
        chapter_count = len(book_data)
        logger.info(f"Book {book_code} has {chapter_count} chapters")
        
        success_count = 0
        failed_chapters = []
        start_time = time.time()
        
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
                    
                # Small delay for stability
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing {book_code} {chapter}: {e}")
                failed_chapters.append(chapter)
        
        total_time = time.time() - start_time
        logger.info(f"📊 Book {book_code} completed: {success_count}/{chapter_count} chapters successful")
        logger.info(f"⏱️ Total time: {total_time/60:.1f} minutes")
        
        if failed_chapters:
            logger.warning(f"⚠️ Failed chapters: {failed_chapters}")
        
        return success_count > 0

    async def process_all(self) -> bool:
        """Process all books in the correct order."""
        logger.info("🌍 Starting full processing of all Tanakh books")
        
        # Order books logically (Torah, Nevi'im, Ketuvim)
        book_order = [
            # Torah
            'Gen', 'Exod', 'Lev', 'Num', 'Deut',
            # Nevi'im Rishonim
            'Josh', 'Judg', '1Sam', '2Sam', '1Kgs', '2Kgs',
            # Nevi'im Aharonim
            'Isa', 'Jer', 'Ezek', 'Hos', 'Joel', 'Amos', 'Obad', 'Jonah', 'Mic', 'Nah', 'Hab', 'Zeph', 'Hag', 'Zech', 'Mal',
            # Ketuvim
            'Ps', 'Prov', 'Job', 'Song', 'Ruth', 'Lam', 'Eccl', 'Esth', 'Dan', 'Ezra', 'Neh', '1Chr', '2Chr'
        ]
        
        # Filter to only books that exist in our data
        available_books = [book for book in book_order if book in self.bible_data]
        
        total_chapters = sum(len(self.bible_data[book]) for book in available_books)
        logger.info(f"Processing {len(available_books)} books, {total_chapters} total chapters")
        
        success_count = 0
        failed_books = []
        start_time = time.time()
        
        for i, book_code in enumerate(available_books, 1):
            try:
                book_chapters = len(self.bible_data[book_code])
                logger.info(f"📖 Starting book {i}/{len(available_books)}: {book_code} ({book_chapters} chapters)")
                
                success = await self.process_book(book_code)
                if success:
                    success_count += 1
                    logger.info(f"✅ Book {book_code} completed successfully")
                else:
                    failed_books.append(book_code)
                    logger.warning(f"❌ Book {book_code} failed")
                    
                # Longer delay between books
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing book {book_code}: {e}")
                failed_books.append(book_code)
        
        total_time = time.time() - start_time
        logger.info(f"🏁 Full processing completed: {success_count}/{len(available_books)} books successful")
        logger.info(f"⏱️ Total time: {total_time/3600:.1f} hours")
        
        if failed_books:
            logger.warning(f"⚠️ Failed books: {failed_books}")
        
        return success_count > 0

async def main():
    """Main function."""
    try:
        tool = HebrewAlignmentTool()
        
        if len(sys.argv) == 1:
            # Process all books
            logger.info("🌍 Starting processing of ALL books")
            success = await tool.process_all()
            print(f"Full processing {'succeeded' if success else 'failed'}")
        elif len(sys.argv) == 3 and sys.argv[1] == "test":
            # Process entire book
            book_code = sys.argv[2]
            logger.info(f"📚 Testing entire book: {book_code}")
            success = await tool.process_book(book_code)
            print(f"Book processing {'succeeded' if success else 'failed'}")
        elif len(sys.argv) == 4 and sys.argv[1] == "test":
            # Process single chapter
            book_code = sys.argv[2]
            chapter = int(sys.argv[3])
            logger.info(f"🧪 Testing chapter: {book_code} {chapter}")
            success = await tool.process_chapter(book_code, chapter)
            print(f"Chapter processing {'succeeded' if success else 'failed'}")
        else:
            print("Usage:")
            print("  python hebrew_alignment_corrected.py                     # Process all books")
            print("  python hebrew_alignment_corrected.py test Gen            # Process entire Genesis")
            print("  python hebrew_alignment_corrected.py test Gen 1          # Process Genesis chapter 1")
            
    except KeyboardInterrupt:
        logger.info("⏹️ Processing interrupted by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())