# Hebrew Encoding Test Script
# Test the encoding fixes before running the full alignment tool

import json
import sys
from pathlib import Path

def test_hebrew_encoding():
    """Test Hebrew text encoding with the Bible data."""
    
    print("=== HEBREW ENCODING TEST ===")
    
    # Test Bible data loading
    bible_path = Path("D:/AI/Projects/HEBREW-TRAINING-AI-AGENT/TANACH/book/hebrew_bible_with_nikkud.json")
    
    if not bible_path.exists():
        print(f"❌ Bible file not found: {bible_path}")
        return False
    
    try:
        # Load with UTF-8 encoding like the converter
        with open(bible_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Bible data loaded: {len(data)} books")
        
        # Test Genesis 1:1
        if 'Gen' in data and len(data['Gen']) > 0 and len(data['Gen'][0]) > 0:
            first_verse = data['Gen'][0][0]
            if isinstance(first_verse, list):
                first_verse_text = ' '.join(first_verse)
            else:
                first_verse_text = str(first_verse)
            
            print(f"📖 Genesis 1:1: {first_verse_text}")
            
            # Check Hebrew characters
            hebrew_chars = sum(1 for char in first_verse_text if '\u0590' <= char <= '\u05FF')
            total_chars = len(first_verse_text)
            
            if hebrew_chars > 0:
                print(f"✅ Hebrew characters: {hebrew_chars}/{total_chars}")
                print("✅ Hebrew encoding is working correctly!")
                
                # Test JSON output with Hebrew
                test_output = {
                    "test_verse": first_verse_text,
                    "verse_number": 1,
                    "confidence": 0.8
                }
                
                # Test file writing
                test_file = Path("test_hebrew_encoding.json")
                with open(test_file, 'w', encoding='utf-8') as f:
                    json.dump(test_output, f, indent=2, ensure_ascii=False)
                
                # Read it back
                with open(test_file, 'r', encoding='utf-8') as f:
                    recovered = json.load(f)
                
                recovered_text = recovered["test_verse"]
                print(f"📄 File write/read test: {recovered_text[:50]}...")
                
                if recovered_text == first_verse_text:
                    print("✅ File encoding test passed!")
                    test_file.unlink()  # Clean up
                    return True
                else:
                    print("❌ File encoding test failed!")
                    return False
            else:
                print("❌ No Hebrew characters found - encoding issue!")
                return False
        else:
            print("❌ Genesis data not found")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_book_mapping():
    """Test book code mapping."""
    
    print("\n=== BOOK MAPPING TEST ===")
    
    bible_path = Path("D:/AI/Projects/HEBREW-TRAINING-AI-AGENT/TANACH/book/hebrew_bible_with_nikkud.json")
    
    try:
        with open(bible_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📚 Available books in Bible data:")
        for i, book in enumerate(sorted(data.keys())):
            chapter_count = len(data[book])
            print(f"  {i+1:2d}. {book:<8} ({chapter_count:2d} chapters)")
        
        # Test specific books
        test_books = ['Gen', 'Exod', 'Ps']
        for book in test_books:
            if book in data:
                chapters = len(data[book])
                first_chapter_verses = len(data[book][0])
                print(f"✅ {book}: {chapters} chapters, chapter 1 has {first_chapter_verses} verses")
            else:
                print(f"❌ {book}: not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Book mapping test failed: {e}")
        return False

def main():
    """Run all encoding tests."""
    
    print("Testing Hebrew encoding fixes...\n")
    
    # Test 1: Hebrew encoding
    encoding_ok = test_hebrew_encoding()
    
    # Test 2: Book mapping
    mapping_ok = test_book_mapping()
    
    print(f"\n=== RESULTS ===")
    if encoding_ok and mapping_ok:
        print("✅ All tests passed! The encoding fixes should work.")
        print("🚀 You can now run the fixed alignment tool.")
    else:
        print("❌ Some tests failed. Check the issues above.")
        
    return encoding_ok and mapping_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)