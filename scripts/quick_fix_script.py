# Quick Fix Script - Addresses immediate issues
# Run this first to test if the fixes work

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# FIXED: Configure logging without emojis for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Force UTF-8 output
        logging.FileHandler('hebrew_alignment_test.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class QuickFixTest:
    def __init__(self):
        logger.info("=== QUICK FIX TEST - HEBREW ALIGNMENT ===")
        
        # GPU detection with better error handling
        self.device = torch.device("cpu")  # Start with CPU
        if torch.cuda.is_available():
            try:
                test_tensor = torch.tensor([1.0]).cuda()
                del test_tensor
                self.device = torch.device("cuda")
                logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            except Exception as e:
                logger.warning(f"GPU test failed: {e}")
        
        # FIXED: Use consistent data types
        self.model_name = "openai/whisper-small"  # Start with smaller model
        self.torch_dtype = torch.float32  # Consistent type
        self.sample_rate = 16000
        
        # Paths
        self.base_path = Path("D:/AI/Projects/HEBREW-TRAINING-AI-AGENT/TANACH")
        self.audio_local_path = Path("D:/audio/tankh_audio_chp")
        self.json_path = self.base_path / "book" / "hebrew_bible_with_nikkud.json"
        self.output_path = self.base_path / "alignment"
        
        self.processor = None
        self.model = None
        
        logger.info("Quick fix test initialized")
    
    def load_whisper_model(self):
        """Load Whisper with fixed data type consistency."""
        try:
            logger.info(f"Loading {self.model_name} on {self.device}...")
            
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            # FIXED: Consistent data type
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype
            )
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False
    
    def test_audio_loading(self):
        """Test audio loading capability."""
        try:
            # Test with Genesis 1
            audio_path = self.audio_local_path / "01_Genesis" / "01-Gen_01.mp3"
            if not audio_path.exists():
                logger.error(f"Audio file not found: {audio_path}")
                return False
            
            logger.info(f"Testing audio loading: {audio_path}")
            audio, sr = librosa.load(str(audio_path), sr=self.sample_rate)
            logger.info(f"Audio loaded: {len(audio)} samples, duration: {len(audio)/sr:.2f}s")
            return True
        except Exception as e:
            logger.error(f"Audio loading failed: {e}")
            return False
    
    def test_bible_data_loading(self):
        """Test Bible data loading."""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Bible data loaded: {len(data)} books")
            
            # Test Genesis extraction
            if 'Gen' in data:
                gen_data = data['Gen']
                logger.info(f"Genesis has {len(gen_data)} chapters")
                if len(gen_data) > 0:
                    logger.info(f"Genesis 1 has {len(gen_data[0])} verses")
                return True
            else:
                logger.error("Genesis not found in Bible data")
                return False
        except Exception as e:
            logger.error(f"Bible data loading failed: {e}")
            return False
    
    def test_file_saving(self):
        """Test the FIXED file saving mechanism."""
        try:
            test_data = {"test": "data", "timestamp": "2025-08-14"}
            output_dir = self.output_path / "Gen"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / "test_alignment.json"
            
            # FIXED: Better file handling
            if output_file.exists():
                backup_file = output_file.with_suffix('.json.backup')
                if backup_file.exists():
                    backup_file.unlink()
                output_file.rename(backup_file)
            
            # Write with unique temp file
            import time
            temp_file = output_file.with_suffix(f'.tmp.{os.getpid()}.{int(time.time())}')
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, indent=2, ensure_ascii=False)
            
            temp_file.rename(output_file)
            logger.info("File saving test successful")
            
            # Clean up
            if output_file.exists():
                output_file.unlink()
            return True
        except Exception as e:
            logger.error(f"File saving test failed: {e}")
            return False
    
    async def test_simple_transcription(self):
        """Test simple transcription with fixed data types."""
        try:
            if not self.load_whisper_model():
                return False
            
            # Load test audio
            audio_path = self.audio_local_path / "01_Genesis" / "01-Gen_01.mp3"
            if not audio_path.exists():
                logger.error("Test audio not found")
                return False
            
            logger.info("Loading audio for transcription test...")
            audio, sr = librosa.load(str(audio_path), sr=self.sample_rate)
            
            # FIXED: Process with consistent data types
            logger.info("Processing with Whisper...")
            inputs = self.processor(
                audio, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt"
            )
            
            # Ensure consistent data types
            inputs = {k: v.to(self.device).to(self.torch_dtype) for k, v in inputs.items()}
            
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    inputs["input_features"],
                    max_length=224,
                    num_beams=1,
                    language="hebrew",
                    task="transcribe"
                )
            
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
            logger.info(f"Transcription successful: {transcription[0][:100]}...")
            return True
            
        except Exception as e:
            logger.error(f"Transcription test failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            if "Input type" in str(e) and "bias type" in str(e):
                logger.error("Data type mismatch detected - this should be fixed in the main script")
            return False
    
    def run_all_tests(self):
        """Run all quick fix tests."""
        logger.info("=== RUNNING QUICK FIX TESTS ===")
        
        tests = [
            ("Bible Data Loading", self.test_bible_data_loading),
            ("Audio Loading", self.test_audio_loading),
            ("File Saving", self.test_file_saving),
        ]
        
        passed = 0
        for test_name, test_func in tests:
            logger.info(f"Running {test_name}...")
            try:
                if test_func():
                    logger.info(f"PASS: {test_name}")
                    passed += 1
                else:
                    logger.error(f"FAIL: {test_name}")
            except Exception as e:
                logger.error(f"ERROR: {test_name} - {e}")
        
        logger.info(f"Basic tests: {passed}/{len(tests)} passed")
        
        # Transcription test (async)
        logger.info("Running transcription test...")
        try:
            result = asyncio.run(self.test_simple_transcription())
            if result:
                logger.info("PASS: Transcription test")
                passed += 1
            else:
                logger.error("FAIL: Transcription test")
        except Exception as e:
            logger.error(f"ERROR: Transcription test - {e}")
        
        logger.info(f"=== TOTAL: {passed}/{len(tests)+1} tests passed ===")
        
        if passed == len(tests) + 1:
            logger.info("ALL TESTS PASSED! The main script should work now.")
        else:
            logger.warning("Some tests failed. Check the logs above.")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test = QuickFixTest()
        test.run_all_tests()
    else:
        print("Usage: python quick_fix_script.py test")

if __name__ == "__main__":
    main()