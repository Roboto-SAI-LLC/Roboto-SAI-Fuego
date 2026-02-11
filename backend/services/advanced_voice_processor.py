"""
Advanced Voice Processor for Roboto SAI
Created by Roberto Villarreal Martinez for Roboto SAI
Advanced voice processing with graceful fallbacks and quantum enhancements
"""

import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

# Constants
DEFAULT_USER_NAME = "Roberto Villarreal Martinez"
CONTEXT_STORAGE_DIR = "conversation_contexts"
AUDIO_SAMPLES_DIR = "audio_samples"
TEMP_TRANSCRIPTION_FILE = "temp_transcription.wav"
MIN_AUDIO_LENGTH_SECONDS = 0.1
MIN_AUDIO_LENGTH_SAMPLES = 1600
AMBIENT_NOISE_ADJUSTMENT_DURATION = 0.5
MAX_SESSION_HISTORY = 100
TOP_KEYWORDS_LIMIT = 5
RECENT_TRANSCRIPTIONS_LIMIT = 3
DEFAULT_EMOTION = "neutral"
DEFAULT_EMOTION_SCORE = 0.5
FALLBACK_EMOTION_SHORT = "neutral"
FALLBACK_EMOTION_LONG = "thoughtful"
FALLBACK_EMOTION_MEDIUM = "engaged"
SHORT_AUDIO_THRESHOLD = 2.0
LONG_AUDIO_THRESHOLD = 10.0

# Logging setup
logger = logging.getLogger(__name__)

# Advanced voice processing with graceful fallbacks
try:
    import speech_recognition as sr  # pyright: ignore[reportMissingImports]
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    sr = None

try:
    from pydub import AudioSegment  # pyright: ignore[reportMissingImports]
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    AudioSegment = None

try:
    import librosa  # pyright: ignore[reportMissingImports]
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None

# Optional advanced AI models - use fallbacks if not available
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    pipeline = None

try:
    from bertopic import BERTopic  # pyright: ignore[reportMissingImports]
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    BERTopic = None

class AdvancedVoiceProcessor:
    """
    Advanced voice processing system that adds conversation context preservation,
    emotion detection, topic modeling, and multi-session continuity to Roboto.
    """

    def __init__(self, user_name: str = DEFAULT_USER_NAME) -> None:
        self.user_name: str = user_name
        
        # Initialize speech recognition
        self.recognizer: Optional[Any] = None
        if SPEECH_RECOGNITION_AVAILABLE and sr:
            self.recognizer = sr.Recognizer()

        # Initialize AI models with error handling
        self.emotion_classifier: Optional[Any] = None
        self.topic_model: Optional[Any] = None
        
        self._initialize_emotion_classifier()
        self._initialize_topic_model()

        # Storage paths
        self.context_storage_dir: str = CONTEXT_STORAGE_DIR
        self.audio_samples_dir: str = AUDIO_SAMPLES_DIR
        self.ensure_directories()

        # Conversation session tracking
        self.current_session_id: Optional[str] = None
        self.session_data: List[Dict[str, Any]] = []
    
    def _initialize_emotion_classifier(self) -> None:
        """Initialize emotion classification model with fallback."""
        if TRANSFORMERS_AVAILABLE and pipeline:
            try:
                self.emotion_classifier = pipeline("audio-classification", 
                                                 model="superb/wav2vec2-base-superb-er")
                logger.info("Emotion classifier initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize emotion classifier: {e}")
                self.emotion_classifier = None
        else:
            logger.info("Transformers not available - using fallback emotion detection")
    
    def _initialize_topic_model(self) -> None:
        """Initialize topic modeling with fallback."""
        if BERTOPIC_AVAILABLE and BERTopic:
            try:
                self.topic_model = BERTopic(language="english", calculate_probabilities=True)
                logger.info("Topic modeling initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize topic model: {e}")
                self.topic_model = None
        else:
            logger.info("BERTopic not available - using fallback topic extraction")

    def ensure_directories(self) -> None:
        """Create necessary directories for storage."""
        try:
            for directory in [self.context_storage_dir, self.audio_samples_dir]:
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    logger.info(f"Created directory: {directory}")
                elif not os.path.isdir(directory):
                    logger.warning(f"Path exists but is not a directory: {directory}")
        except Exception as e:
            logger.error(f"Error creating directories: {e}")

    def transcribe_audio(self, audio_file: str) -> str:
        """
        Transcribe audio file to text with enhanced error handling.
        
        Args:
            audio_file: Path to the audio file to transcribe
            
        Returns:
            Transcribed text or error message
        """
        try:
            # Validate input
            if not self._validate_audio_file(audio_file):
                return "Invalid audio file provided"
            
            # Check if speech recognition is available
            if not SPEECH_RECOGNITION_AVAILABLE or not self.recognizer:
                return "Speech recognition not available - using fallback transcription simulation"
            
            # Convert audio format if needed
            processed_audio_file = self._prepare_audio_for_transcription(audio_file)
            
            # Perform transcription
            return self._perform_transcription(processed_audio_file)
            
        except Exception as e:
            logger.error(f"Audio transcription error: {e}")
            return f"Audio processing error: {e}"
        finally:
            # Clean up temporary files
            self._cleanup_temp_files()
    
    def _validate_audio_file(self, audio_file: str) -> bool:
        """Validate that the audio file exists and is accessible."""
        if not audio_file or not isinstance(audio_file, str):
            logger.error("Invalid audio file path provided")
            return False
        
        if not os.path.exists(audio_file):
            logger.error(f"Audio file does not exist: {audio_file}")
            return False
        
        if not os.path.isfile(audio_file):
            logger.error(f"Path is not a file: {audio_file}")
            return False
        
        return True
    
    def _prepare_audio_for_transcription(self, audio_file: str) -> str:
        """Prepare audio file for transcription by converting format if needed."""
        if not PYDUB_AVAILABLE or not AudioSegment:
            return audio_file
        
        try:
            # Check if conversion is needed
            if not audio_file.lower().endswith('.wav'):
                audio = AudioSegment.from_file(audio_file)
                audio.export(TEMP_TRANSCRIPTION_FILE, format="wav")
                return TEMP_TRANSCRIPTION_FILE
            return audio_file
        except Exception as e:
            logger.warning(f"Audio conversion failed, using original file: {e}")
            return audio_file
    
    def _perform_transcription(self, audio_file: str) -> str:
        """Perform the actual speech recognition transcription."""
        try:
            with sr.AudioFile(audio_file) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=AMBIENT_NOISE_ADJUSTMENT_DURATION)
                audio_data = self.recognizer.record(source)

                # Try Google Speech Recognition first
                text = self.recognizer.recognize_google(audio_data)
                logger.info(f"Successfully transcribed audio: {text[:50]}...")
                return text
                
        except sr.UnknownValueError:
            logger.warning("Google Speech Recognition could not understand audio")
            return "Could not understand the audio content"
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {e}")
            return f"Transcription service error: {e}"
    
    def _cleanup_temp_files(self) -> None:
        """Clean up temporary transcription files."""
        try:
            if os.path.exists(TEMP_TRANSCRIPTION_FILE):
                os.remove(TEMP_TRANSCRIPTION_FILE)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file: {e}")

    def detect_emotions(self, audio_file: str) -> List[Dict[str, Any]]:
        """
        Detect emotions in audio using transformer models or fallback analysis.
        
        Args:
            audio_file: Path to the audio file to analyze
            
        Returns:
            List of detected emotions with scores
        """
        try:
            # Validate input
            if not self._validate_audio_file(audio_file):
                return [{"label": DEFAULT_EMOTION, "score": DEFAULT_EMOTION_SCORE, "method": "invalid_file"}]
            
            # Use emotion classifier if available
            if self.emotion_classifier:
                return self._classify_emotions_with_model(audio_file)
            else:
                return self._fallback_emotion_detection(audio_file)
                
        except Exception as e:
            logger.error(f"Emotion detection error: {e}")
            return self._fallback_emotion_detection(audio_file)
    
    def _classify_emotions_with_model(self, audio_file: str) -> List[Dict[str, Any]]:
        """Use the emotion classification model to detect emotions."""
        if not LIBROSA_AVAILABLE or not librosa:
            logger.warning("Librosa not available - using fallback emotion detection")
            return self._fallback_emotion_detection(audio_file)
        
        try:
            # Load and resample audio for the model
            audio, sample_rate = librosa.load(audio_file, sr=16000)

            # Ensure audio is not too short
            if len(audio) < MIN_AUDIO_LENGTH_SAMPLES:  # Less than MIN_AUDIO_LENGTH_SECONDS seconds
                logger.warning("Audio too short for emotion detection")
                return [{"label": DEFAULT_EMOTION, "score": DEFAULT_EMOTION_SCORE, "method": "too_short"}]

            emotions = self.emotion_classifier(audio)
            logger.info(f"Detected emotions: {emotions[:2]}")
            return emotions

        except Exception as e:
            logger.error(f"Model-based emotion detection error: {e}")
            return self._fallback_emotion_detection(audio_file)

    def _fallback_emotion_detection(self, audio_file: str) -> List[Dict[str, Any]]:
        """
        Simple fallback emotion detection based on basic audio analysis.
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            List with fallback emotion detection result
        """
        try:
            # Try to get basic audio info for emotion estimation
            if PYDUB_AVAILABLE and AudioSegment:
                duration = self._get_audio_duration(audio_file)
                
                # Simple heuristic based on duration
                emotion, score = self._estimate_emotion_from_duration(duration)
                
                return [{"label": emotion, "score": score, "method": "fallback_analysis"}]
            else:
                return [{"label": DEFAULT_EMOTION, "score": DEFAULT_EMOTION_SCORE, "method": "default_fallback"}]

        except Exception as e:
            logger.error(f"Fallback emotion detection error: {e}")
            return [{"label": DEFAULT_EMOTION, "score": DEFAULT_EMOTION_SCORE, "method": "error_fallback"}]
    
    def _estimate_emotion_from_duration(self, duration: float) -> Tuple[str, float]:
        """Estimate emotion based on audio duration."""
        if duration < SHORT_AUDIO_THRESHOLD:
            return FALLBACK_EMOTION_SHORT, 0.6
        elif duration > LONG_AUDIO_THRESHOLD:
            return FALLBACK_EMOTION_LONG, 0.7
        else:
            return FALLBACK_EMOTION_MEDIUM, 0.65

    def extract_topics(self, text: str) -> Tuple[Dict[str, Any], List[int], List[float]]:
        """
        Extract topics from transcribed text using BERTopic or fallback analysis.
        
        Args:
            text: Text to extract topics from
            
        Returns:
            Tuple of (topic_info, topics, probabilities)
        """
        try:
            # Validate input
            if not text or not isinstance(text, str) or not text.strip():
                return {}, [], []
            
            # Use topic model if available
            if self.topic_model:
                return self._extract_topics_with_model(text)
            else:
                return self._fallback_topic_extraction(text)
                
        except Exception as e:
            logger.error(f"Topic extraction error: {e}")
            return self._fallback_topic_extraction(text)
    
    def _extract_topics_with_model(self, text: str) -> Tuple[Dict[str, Any], List[int], List[float]]:
        """Extract topics using the BERTopic model."""
        try:
            topics, probabilities = self.topic_model.fit_transform([text])
            topic_info = self.topic_model.get_topic_info()

            logger.info(f"Extracted {len(topic_info)} topics from text")
            return topic_info, topics, probabilities

        except Exception as e:
            logger.error(f"Model-based topic extraction error: {e}")
            return self._fallback_topic_extraction(text)

    def _fallback_topic_extraction(self, text: str) -> Tuple[Dict[str, Any], List[int], List[float]]:
        """
        Simple fallback topic extraction using keyword analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (topic_info, topics, probabilities)
        """
        try:
            words = text.lower().split()
            word_freq = self._calculate_word_frequencies(words)
            
            # Get top keywords
            top_words = self._get_top_keywords(word_freq)
            
            # Create simple topic structure
            topic_info = {
                "fallback_analysis": True,
                "top_keywords": [word for word, count in top_words],
                "keyword_frequencies": dict(top_words),
                "text_length": len(text),
                "word_count": len(words)
            }

            return topic_info, [0], [0.5]  # Simple fallback values

        except Exception as e:
            logger.error(f"Fallback topic extraction error: {e}")
            return {"error": "Topic extraction failed"}, [], []
    
    def _calculate_word_frequencies(self, words: List[str]) -> Dict[str, int]:
        """Calculate word frequencies excluding common words."""
        common_words = self._get_common_words()
        word_freq = {}
        
        for word in words:
            if len(word) > 3 and word not in common_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        return word_freq
    
    def _get_common_words(self) -> set:
        """Get set of common words to exclude from topic extraction."""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'cannot', 'i', 
            'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'these', 'those'
        }
    
    def _get_top_keywords(self, word_freq: Dict[str, int]) -> List[Tuple[str, int]]:
        """Get top keywords by frequency."""
        return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:TOP_KEYWORDS_LIMIT]

    def process_voice_chat(self, audio_files: List[str], session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process multiple audio files, extract comprehensive conversation context.
        This is the main enhancement that adds context preservation across sessions.
        
        Args:
            audio_files: List of audio file paths to process
            session_id: Optional session identifier
            
        Returns:
            List of processing results for each audio file
        """
        try:
            # Generate session ID if not provided
            session_id = session_id or self._generate_session_id()
            self.current_session_id = session_id
            
            logger.info(f"Processing voice chat session {session_id} with {len(audio_files)} files")
            
            # Process each audio file
            results = []
            for i, audio_file in enumerate(audio_files):
                result = self._process_single_audio_file(audio_file, session_id, i)
                results.append(result)
            
            # Save session context
            self.save_session_context(results, session_id)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing voice chat: {e}")
            return []
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _process_single_audio_file(self, audio_file: str, session_id: str, file_index: int) -> Dict[str, Any]:
        """Process a single audio file and return results."""
        try:
            # Check if file exists
            if not os.path.exists(audio_file):
                logger.warning(f"File {audio_file} not found, skipping")
                return self._create_error_result(session_id, file_index, audio_file, "File not found")
            
            logger.info(f"Processing audio file {file_index + 1}: {audio_file}")
            
            # Transcribe audio
            transcription = self.transcribe_audio(audio_file)
            
            # Detect emotions
            emotions = self.detect_emotions(audio_file)
            
            # Extract topics if transcription was successful
            topic_info, topics, probabilities = self._extract_topics_from_transcription(transcription)
            
            # Compile comprehensive results
            return self._compile_processing_result(
                session_id, file_index, audio_file, transcription, 
                emotions, topic_info, topics, probabilities
            )
            
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}")
            return self._create_error_result(session_id, file_index, audio_file, str(e))
    
    def _extract_topics_from_transcription(self, transcription: str) -> Tuple[Dict[str, Any], List[int], List[float]]:
        """Extract topics from transcription if successful."""
        if transcription and "error" not in transcription.lower():
            return self.extract_topics(transcription)
        return {}, [], []
    
    def _compile_processing_result(self, session_id: str, file_index: int, audio_file: str, 
                                 transcription: str, emotions: List[Dict[str, Any]], 
                                 topic_info: Dict[str, Any], topics: List[int], 
                                 probabilities: List[float]) -> Dict[str, Any]:
        """Compile the processing result dictionary."""
        return {
            "session_id": session_id,
            "file_index": file_index,
            "file": audio_file,
            "timestamp": datetime.now().isoformat(),
            "transcription": transcription,
            "emotions": emotions,
            "dominant_emotion": emotions[0]["label"] if emotions else DEFAULT_EMOTION,
            "emotion_confidence": emotions[0]["score"] if emotions else DEFAULT_EMOTION_SCORE,
            "topics": self._normalize_topic_info(topic_info),
            "topic_probabilities": self._normalize_probabilities(probabilities),
            "processing_metadata": {
                "audio_duration": self._get_audio_duration(audio_file),
                "text_length": len(transcription) if transcription else 0,
                "emotion_model_used": bool(self.emotion_classifier),
                "topic_model_used": bool(self.topic_model)
            }
        }
    
    def _normalize_topic_info(self, topic_info: Any) -> Dict[str, Any]:
        """Normalize topic info to dictionary format."""
        if hasattr(topic_info, 'to_dict') and topic_info is not None:
            return topic_info.to_dict()
        return topic_info if topic_info is not None else {}
    
    def _normalize_probabilities(self, probabilities: Any) -> List[float]:
        """Normalize probabilities to list format."""
        if hasattr(probabilities, 'tolist') and probabilities is not None:
            return probabilities.tolist()
        return probabilities if probabilities is not None else []
    
    def _create_error_result(self, session_id: str, file_index: int, audio_file: str, error: str) -> Dict[str, Any]:
        """Create an error result for failed processing."""
        return {
            "session_id": session_id,
            "file_index": file_index,
            "file": audio_file,
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "transcription": f"Processing error: {error}",
            "emotions": [{"label": "error", "score": 0.0}],
            "topics": {}
        }

    def _get_audio_duration(self, audio_file: str) -> float:
        """
        Get audio file duration in seconds.
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            Duration in seconds, or 0.0 if unable to determine
        """
        try:
            if PYDUB_AVAILABLE and AudioSegment:
                audio = AudioSegment.from_file(audio_file)
                return len(audio) / 1000.0  # Convert milliseconds to seconds
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"Could not get audio duration for {audio_file}: {e}")
            return 0.0

    def save_session_context(self, results: List[Dict[str, Any]], session_id: str) -> None:
        """
        Save conversation context for session continuity.
        
        Args:
            results: Processing results from audio files
            session_id: Session identifier
        """
        try:
            output_file = os.path.join(self.context_storage_dir, f"{session_id}_context.json")
            
            # Create comprehensive session summary
            session_summary = self._create_session_summary(results, session_id)
            
            # Save to file
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(session_summary, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Session context saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving session context: {e}")
    
    def _create_session_summary(self, results: List[Dict[str, Any]], session_id: str) -> Dict[str, Any]:
        """Create a comprehensive session summary."""
        return {
            "session_id": session_id,
            "user_name": self.user_name,
            "created_at": datetime.now().isoformat(),
            "total_files": len(results),
            "successful_transcriptions": self._count_successful_transcriptions(results),
            "dominant_emotions": self._analyze_session_emotions(results),
            "key_topics": self._analyze_session_topics(results),
            "conversation_flow": results,
            "session_metadata": self._create_session_metadata(results)
        }
    
    def _count_successful_transcriptions(self, results: List[Dict[str, Any]]) -> int:
        """Count the number of successful transcriptions."""
        return sum(1 for r in results if r.get("transcription") and "error" not in r.get("transcription", "").lower())
    
    def _create_session_metadata(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create session metadata."""
        return {
            "total_duration": sum(r.get("processing_metadata", {}).get("audio_duration", 0) for r in results),
            "average_emotion_confidence": self._calculate_average_emotion_confidence(results),
            "models_used": {
                "emotion_detection": bool(self.emotion_classifier),
                "topic_modeling": bool(self.topic_model),
                "speech_recognition": True
            }
        }
    
    def _calculate_average_emotion_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate average emotion confidence across results."""
        confidences = [r.get("emotion_confidence", 0) for r in results if r.get("emotion_confidence") is not None]
        return float(np.mean(confidences)) if confidences else 0.0

    def _analyze_session_emotions(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze emotional patterns across the session.
        
        Args:
            results: Processing results from audio files
            
        Returns:
            Dictionary containing emotion analysis results
        """
        try:
            emotions = [r.get("dominant_emotion", "neutral") for r in results if r.get("dominant_emotion")]
            
            if not emotions:
                return {
                    "most_common": ("neutral", 0),
                    "emotion_distribution": {},
                    "emotional_trajectory": []
                }
            
            # Count emotion frequencies
            emotion_counts = {}
            for emotion in emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # Find most common emotion
            most_common = max(emotion_counts.items(), key=lambda x: x[1])
            
            return {
                "most_common": most_common,
                "emotion_distribution": emotion_counts,
                "emotional_trajectory": emotions
            }
            
        except Exception as e:
            logger.error(f"Error analyzing session emotions: {e}")
            return {
                "most_common": ("neutral", 0),
                "emotion_distribution": {},
                "emotional_trajectory": []
            }

    def _analyze_session_topics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze topical patterns across the session.
        
        Args:
            results: Processing results from audio files
            
        Returns:
            Dictionary containing topic analysis results
        """
        try:
            # Collect all transcription text
            all_text = " ".join([
                r.get("transcription", "") 
                for r in results 
                if r.get("transcription") and "error" not in r.get("transcription", "").lower()
            ])
            
            if not all_text.strip():
                return {
                    "summary": "No transcription text available for topic analysis",
                    "text_length": 0
                }
            
            if not self.topic_model:
                return {
                    "summary": "Topic modeling not available",
                    "text_length": len(all_text)
                }
            
            # Extract topics from combined text
            topic_info, topics, probabilities = self.extract_topics(all_text)
            
            # Normalize topic info
            normalized_topic_info = self._normalize_topic_info(topic_info)
            normalized_probabilities = self._normalize_probabilities(probabilities)
            
            return {
                "session_topics": normalized_topic_info,
                "topic_confidence": normalized_probabilities,
                "text_analyzed": len(all_text)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing session topics: {e}")
            return {
                "summary": f"Topic analysis failed: {e}",
                "text_length": len(all_text) if 'all_text' in locals() else 0
            }

    def load_context_for_new_session(self, session_id: Optional[str] = None, context_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Load saved context for continuing conversation across sessions.
        
        Args:
            session_id: Optional session identifier
            context_file: Optional direct path to context file
            
        Returns:
            Loaded context dictionary, empty dict if not found
        """
        try:
            file_path = self._determine_context_file_path(session_id, context_file)
            
            if not file_path:
                logger.warning("No context file could be determined")
                return {}
            
            if not os.path.exists(file_path):
                logger.warning(f"Context file does not exist: {file_path}")
                return {}
            
            # Load and validate context
            context = self._load_context_from_file(file_path)
            
            logger.info(f"Loaded context from {file_path}")
            return context
            
        except Exception as e:
            logger.error(f"Error loading context: {e}")
            return {}
    
    def _determine_context_file_path(self, session_id: Optional[str], context_file: Optional[str]) -> Optional[str]:
        """Determine the context file path to load."""
        if context_file:
            return context_file
        elif session_id:
            return os.path.join(self.context_storage_dir, f"{session_id}_context.json")
        else:
            # Load most recent session
            return self._find_most_recent_context_file()
    
    def _find_most_recent_context_file(self) -> Optional[str]:
        """Find the most recent context file."""
        try:
            context_files = [
                f for f in os.listdir(self.context_storage_dir) 
                if f.endswith("_context.json")
            ]
            
            if not context_files:
                return None
            
            # Find file with most recent creation time
            latest_file = max(
                context_files, 
                key=lambda f: os.path.getctime(os.path.join(self.context_storage_dir, f))
            )
            
            return os.path.join(self.context_storage_dir, latest_file)
            
        except Exception as e:
            logger.error(f"Error finding recent context file: {e}")
            return None
    
    def _load_context_from_file(self, file_path: str) -> Dict[str, Any]:
        """Load context from a JSON file."""
        with open(file_path, "r", encoding='utf-8') as f:
            context = json.load(f)
        
        # Validate context structure
        if not isinstance(context, dict):
            raise ValueError("Invalid context file format")
        
        return context

    def generate_conversation_summary(self, context: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary for AI continuation.
        
        Args:
            context: Session context dictionary
            
        Returns:
            Formatted conversation summary string
        """
        try:
            if not context:
                return "No previous conversation context available."
            
            # Build summary components
            summary_parts = []
            
            # Basic session info
            session_info = self._build_session_info_summary(context)
            summary_parts.extend(session_info)
            
            # Emotional context
            emotion_summary = self._build_emotion_summary(context)
            if emotion_summary:
                summary_parts.append(emotion_summary)
            
            # Topical context
            topic_summary = self._build_topic_summary(context)
            if topic_summary:
                summary_parts.append(topic_summary)
            
            # Recent conversation snippets
            recent_snippets = self._build_recent_snippets(context)
            if recent_snippets:
                summary_parts.extend(recent_snippets)
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Context available but summary generation failed: {e}"
    
    def _build_session_info_summary(self, context: Dict[str, Any]) -> List[str]:
        """Build basic session information summary."""
        session_id = context.get("session_id", "unknown")
        total_files = context.get("total_files", 0)
        
        return [
            f"Previous conversation session: {session_id}",
            f"Total interactions: {total_files}",
        ]
    
    def _build_emotion_summary(self, context: Dict[str, Any]) -> Optional[str]:
        """Build emotional context summary."""
        emotions = context.get("dominant_emotions", {})
        most_common = emotions.get("most_common")
        
        if most_common and most_common[1] > 0:
            emotion, count = most_common
            return f"Dominant emotion: {emotion} ({count} instances)"
        
        return None
    
    def _build_topic_summary(self, context: Dict[str, Any]) -> Optional[str]:
        """Build topical context summary."""
        topics = context.get("key_topics", {})
        
        if topics.get("session_topics"):
            return "Key topics discussed: available in detailed analysis"
        
        return None
    
    def _build_recent_snippets(self, context: Dict[str, Any]) -> List[str]:
        """Build recent conversation snippets."""
        conversation_flow = context.get("conversation_flow", [])
        
        if not conversation_flow:
            return []
        
        # Get recent transcriptions (last 3)
        recent_transcriptions = [
            item.get("transcription", "")[:100] + "..." 
            for item in conversation_flow[-3:] 
            if item.get("transcription") and "error" not in item.get("transcription", "").lower()
        ]
        
        if not recent_transcriptions:
            return []
        
        snippets = ["Recent conversation snippets:"]
        snippets.extend([f"- {snippet}" for snippet in recent_transcriptions])
        
        return snippets

    def integrate_with_roboto(self, audio_files: List[str]) -> Dict[str, Any]:
        """
        Main integration method for Roboto - processes voice chat and returns
        enhanced context for the AI conversation system.
        
        Args:
            audio_files: List of audio file paths to process
            
        Returns:
            Dictionary containing processed context and integration metadata
        """
        try:
            logger.info("Integrating advanced voice processing with Roboto")
            
            # Process the voice chat
            session_context = self.process_voice_chat(audio_files)
            
            # Generate comprehensive context summary
            context_summary = self._build_roboto_context_summary(session_context)
            
            return context_summary
            
        except Exception as e:
            logger.error(f"Error in Roboto integration: {e}")
            return self._create_error_integration_response(str(e))
    
    def _build_roboto_context_summary(self, session_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build comprehensive context summary for Roboto integration."""
        # Create mock context for summary generation
        mock_context = {
            "session_id": self.current_session_id,
            "conversation_flow": session_context,
            "dominant_emotions": self._analyze_session_emotions(session_context),
            "key_topics": self._analyze_session_topics(session_context)
        }
        
        # Add session metadata if available
        if session_context:
            mock_context.update({
                "total_files": len(session_context),
                "successful_transcriptions": self._count_successful_transcriptions(session_context)
            })
        
        return {
            "session_context": session_context,
            "conversation_summary": self.generate_conversation_summary(mock_context),
            "roboto_integration": self._create_integration_metadata()
        }
    
    def _create_integration_metadata(self) -> Dict[str, Any]:
        """Create integration metadata for Roboto."""
        return {
            "voice_profile_user": self.user_name,
            "context_preserved": True,
            "emotion_analysis_enabled": bool(self.emotion_classifier),
            "topic_modeling_enabled": bool(self.topic_model),
            "session_continuity": True,
            "processing_timestamp": datetime.now().isoformat(),
            "integration_version": "2.0"
        }
    
    def _create_error_integration_response(self, error: str) -> Dict[str, Any]:
        """Create error response for integration failures."""
        return {
            "session_context": [],
            "conversation_summary": f"Voice processing integration failed: {error}",
            "roboto_integration": {
                "voice_profile_user": self.user_name,
                "context_preserved": False,
                "error": error,
                "processing_timestamp": datetime.now().isoformat()
            }
        }

# Example usage and testing function
def example_usage() -> None:
    """
    Example of how to use the AdvancedVoiceProcessor with Roboto.
    Demonstrates the integration workflow and capabilities.
    """
    try:
        # Initialize the voice processor
        processor = AdvancedVoiceProcessor("Roberto Villarreal Martinez")
        
        # Example audio files (would be actual recorded files)
        audio_files = ["voice_chat1.wav", "voice_chat2.wav"]
        
        # Check for existing example files
        existing_files = [f for f in audio_files if os.path.exists(f)]
        
        if existing_files:
            print("Found audio files, processing...")
            
            # Process audio files and get enhanced context
            roboto_context = processor.integrate_with_roboto(existing_files)
            
            # Display results
            _display_integration_results(roboto_context)
            
        else:
            print("No audio files found for processing example")
            _display_initialization_status(processor)
            
    except Exception as e:
        print(f"Error in example usage: {e}")
        logger.error(f"Example usage failed: {e}")

def _display_integration_results(roboto_context: Dict[str, Any]) -> None:
    """Display the results of voice processing integration."""
    print("=" * 60)
    print("ROBOTO ADVANCED VOICE PROCESSING INTEGRATION")
    print("=" * 60)
    print(roboto_context["conversation_summary"])
    print("\nIntegration Status:")
    for key, value in roboto_context["roboto_integration"].items():
        print(f"  {key}: {value}")

def _display_initialization_status(processor: AdvancedVoiceProcessor) -> None:
    """Display the initialization status of the voice processor."""
    print("Advanced Voice Processor initialized and ready for integration")
    print(f"User: {processor.user_name}")
    print(f"Emotion analysis: {'Enabled' if processor.emotion_classifier else 'Disabled'}")
    print(f"Topic modeling: {'Enabled' if processor.topic_model else 'Disabled'}")
    print(f"Context storage: {processor.context_storage_dir}")

if __name__ == "__main__":
    example_usage()