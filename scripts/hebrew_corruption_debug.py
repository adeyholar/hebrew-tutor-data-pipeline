# Hebrew Corruption Debug Script
# Find exactly where Hebrew text gets corrupted

import json
import sys
from pathlib import Path

def debug_verse_extraction():
    """Debug the verse extraction process step by step."""
    
    print("=== DEBUGGING VERSE EXTRACTION ===")
    
    # Load Bible data
    bible_path = Path("D:/AI/Projects/HEBREW-TRAINING-AI-AGENT/TANACH/book/hebrew_bible_with_nikkud.json")
    
    with open(bible_path, 'r', encoding='utf-8') as f:
        bible_data = json.load(f)
    
    print(f"✅ Bible data loaded: {len(bible_data)} books")
    
    # Get Genesis Chapter 1
    book_code = 'Gen'
    chapter = 1
    
    if book_code not in bible_data:
        print(f"❌ Book '{book_code}' not found")
        return
    
    book_data = bible_data[book_code]
    print(f"📖 {book_code} has {len(book_data)} chapters")
    
    chapter_data = book_data[chapter - 1]
    print(f"📄 Chapter {chapter} has {len(chapter_data)} verses")
    
    # Debug first verse step by step
    print(f"\n=== DEBUGGING FIRST VERSE ===")
    
    verse_words = chapter_data[0]
    print(f"Raw verse data type: {type(verse_words)}")
    print(f"Raw verse data: {verse_words}")
    
    if isinstance(verse_words, list):
        print(f"Verse has {len(verse_words)} words")
        
        # Check each word
        for i, word in enumerate(verse_words[:5]):  # First 5 words
            print(f"  Word {i+1}: '{word}' (type: {type(word)})")
            # Check encoding
            hebrew_chars = sum(1 for char in str(word) if '\u0590' <= char <= '\u05FF')
            print(f"    Hebrew chars: {hebrew_chars}/{len(str(word))}")
        
        # Test joining
        joined_text = ' '.join(verse_words)
        print(f"\nJoined text: '{joined_text}'")
        
        # Check joined text encoding
        hebrew_chars = sum(1 for char in joined_text if '\u0590' <= char <= '\u05FF')
        print(f"Joined Hebrew chars: {hebrew_chars}/{len(joined_text)}")
        
        # Test creating verse dict like in the tool
        verse_dict = {
            'verse_number': 1,
            'text': joined_text
        }
        
        print(f"\nVerse dict text: '{verse_dict['text']}'")
        
        # Test JSON serialization
        json_str = json.dumps(verse_dict, ensure_ascii=False, indent=2)
        print(f"\nJSON serialization (first 200 chars):")
        print(json_str[:200] + "...")
        
        # Test JSON parsing
        parsed = json.loads(json_str)
        print(f"\nParsed back text: '{parsed['text']}'")
        
        if parsed['text'] == joined_text:
            print("✅ JSON round-trip preserved Hebrew text")
        else:
            print("❌ JSON round-trip corrupted Hebrew text")
            
        return verse_dict
    
    else:
        print(f"❌ Unexpected verse format: {type(verse_words)}")
        return None

def debug_alignment_creation():
    """Debug the alignment creation process."""
    
    print(f"\n=== DEBUGGING ALIGNMENT CREATION ===")
    
    # Get a proper verse
    verse_dict = debug_verse_extraction()
    if not verse_dict:
        return
    
    # Simulate creating alignment data like the tool does
    verses = [verse_dict]
    duration = 362.02
    
    # Test estimated alignment creation
    aligned_verses = []
    total_words = sum(len(verse.get('text', '').split()) for verse in verses)
    
    current_time = 0.0
    for i, verse in enumerate(verses):
        words = verse.get('text', '').split()
        word_count = len(words)
        
        print(f"Processing verse {i+1}:")
        print(f"  Text: '{verse.get('text', '')}'")
        print(f"  Word count: {word_count}")
        
        # Check individual words
        for j, word in enumerate(words[:3]):  # First 3 words
            print(f"    Word {j+1}: '{word}'")
            hebrew_chars = sum(1 for char in word if '\u0590' <= char <= '\u05FF')
            print(f"      Hebrew chars: {hebrew_chars}/{len(word)}")
        
        verse_duration = 5.76  # From actual data
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
                
                word_dict = {
                    'word': word,
                    'start': round(word_start, 2),
                    'end': round(word_end, 2),
                    'confidence': 0.1
                }
                word_alignments.append(word_dict)
                
                # Check first few words
                if j < 3:
                    print(f"    Word alignment {j+1}: '{word_dict['word']}'")
        
        aligned_verse = {
            'verse_number': verse.get('verse_number', i + 1),
            'text': verse.get('text', ''),
            'start': round(start_time, 2),
            'end': round(end_time, 2),
            'confidence': 0.1,
            'words': word_alignments
        }
        
        aligned_verses.append(aligned_verse)
    
    # Create final alignment data
    alignment_data = {
        'total_duration': round(duration, 2),
        'overall_confidence': 0.1,
        'verse_count': len(verses),
        'verses': aligned_verses,
        'metadata': {
            'method': 'estimated_improved',
            'model': 'debug_test',
            'created_at': '2025-08-14T20:40:00Z'
        }
    }
    
    print(f"\nFinal alignment data verse text: '{alignment_data['verses'][0]['text']}'")
    print(f"Final alignment data first word: '{alignment_data['verses'][0]['words'][0]['word']}'")
    
    # Test JSON output
    test_file = Path("debug_alignment_test.json")
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(alignment_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Test file written: {test_file}")
    
    # Read it back
    with open(test_file, 'r', encoding='utf-8') as f:
        recovered = json.load(f)
    
    recovered_text = recovered['verses'][0]['text']
    recovered_word = recovered['verses'][0]['words'][0]['word']
    
    print(f"Recovered verse text: '{recovered_text}'")
    print(f"Recovered first word: '{recovered_word}'")
    
    # Check if corruption occurred
    original_text = alignment_data['verses'][0]['text']
    if recovered_text == original_text:
        print("✅ File write/read preserved Hebrew text")
    else:
        print("❌ File write/read corrupted Hebrew text")
        print(f"Original:  '{original_text}'")
        print(f"Recovered: '{recovered_text}'")
    
    # Clean up
    test_file.unlink()
    
    return alignment_data

def main():
    """Run the corruption debugging."""
    
    print("Debugging Hebrew text corruption...\n")
    
    try:
        debug_alignment_creation()
        
        print(f"\n=== CONCLUSION ===")
        print("If all steps above show correct Hebrew text,")
        print("then the issue might be in:")
        print("1. Console/terminal display encoding")
        print("2. File viewer encoding")
        print("3. How the alignment tool processes transcription results")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()