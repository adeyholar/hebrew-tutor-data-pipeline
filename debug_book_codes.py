#!/usr/bin/env python3
"""
Debug script to identify book code mismatches
"""
import json
from pathlib import Path

def debug_book_codes():
    """Debug book codes in JSON vs expected mappings"""
    
    # Load your JSON data
    json_path = Path('D:/AI/Projects/HEBREW-TRAINING-AI-AGENT/TANACH/book/hebrew_bible_with_nikkud.json')
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            bible_data = json.load(f)
        
        print("📚 Book codes found in your JSON file:")
        json_books = list(bible_data.keys())
        for i, book in enumerate(json_books, 1):
            chapters = len(bible_data[book])
            print(f"{i:2d}. {book:<12} ({chapters} chapters)")
        
        print(f"\nTotal books in JSON: {len(json_books)}")
        
        # Check audio directory structure
        audio_dir = Path('D:/audio/tankh_audio_chp')
        print(f"\n🎵 Audio directory structure check:")
        print(f"Audio base directory exists: {audio_dir.exists()}")
        
        if audio_dir.exists():
            subdirs = [d for d in audio_dir.iterdir() if d.is_dir()]
            print(f"Found {len(subdirs)} subdirectories:")
            for subdir in sorted(subdirs):
                mp3_files = list(subdir.glob("*.mp3"))
                print(f"  {subdir.name:<25} ({len(mp3_files)} MP3 files)")
        
        # Expected mappings from your code
        from hebrew_alignment_tool_fixed import COMPLETE_BOOK_MAPPING
        mapped_books = list(COMPLETE_BOOK_MAPPING.keys())
        
        print(f"\n🗺️  Books in mapping: {len(mapped_books)}")
        
        # Find mismatches
        json_set = set(json_books)
        mapped_set = set(mapped_books)
        
        missing_in_mapping = json_set - mapped_set
        extra_in_mapping = mapped_set - json_set
        
        if missing_in_mapping:
            print(f"\n❌ Book codes in JSON but NOT in mapping:")
            for book in sorted(missing_in_mapping):
                print(f"   {book}")
        
        if extra_in_mapping:
            print(f"\n⚠️  Book codes in mapping but NOT in JSON:")
            for book in sorted(extra_in_mapping):
                print(f"   {book}")
        
        # Perfect matches
        matches = json_set & mapped_set
        print(f"\n✅ Matching book codes: {len(matches)}")
        for book in sorted(matches):
            print(f"   {book}")
        
        # Sample audio file check
        print(f"\n🔍 Sample audio file existence check:")
        sample_books = list(matches)[:3] if matches else json_books[:3]
        
        for book_code in sample_books:
            if book_code in COMPLETE_BOOK_MAPPING:
                mapping = COMPLETE_BOOK_MAPPING[book_code]
                folder_name = mapping['folder']
                abbr = mapping['abbr']
                num = folder_name.split('_')[0]
                
                # Check first chapter
                audio_filename = f"{num}-{abbr}_01.mp3"
                audio_path = audio_dir / folder_name / audio_filename
                
                exists = audio_path.exists()
                status = "✅" if exists else "❌"
                print(f"   {status} {book_code}: {audio_path}")
        
    except FileNotFoundError:
        print(f"❌ JSON file not found: {json_path}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_book_codes()