
"""
Learning Optimization System for Roboto
Implements self-improving algorithms with offline learning capabilities

Created by Roberto Villarreal Martinez for Roboto SAI
"""

import json
import os
import numpy as np
from datetime import datetime, timedelta, date
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Tuple, Union
import re
import math
import logging

# Optional imports for quantum/cultural enhancements
try:
    from quantum_capabilities import QuantumOptimizer
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False

try:
    from config_identity import verify_owner_identity
    IDENTITY_AVAILABLE = True
except ImportError:
    IDENTITY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningOptimizer:
    """
    Advanced learning optimization system for Roboto SAI.
    Analyzes conversation quality, optimizes response strategies, and adapts learning parameters.
    """

    def __init__(self, data_file: str = "roboto_learning_optimization.json"):
        """
        Initialize the Learning Optimizer.

        Args:
            data_file: Path to the data file for persistence
        """
        self.data_file = data_file

        # Core learning metrics
        self.performance_history: List[Dict[str, Any]] = []
        self.conversation_patterns: defaultdict = defaultdict(list)
        self.response_effectiveness: defaultdict = defaultdict(float)
        self.topic_mastery: defaultdict = defaultdict(dict)
        self.emotional_intelligence_scores: defaultdict = defaultdict(list)

        # Advanced learning parameters
        self.learning_rate: float = 0.1
        self.adaptation_threshold: float = 0.7
        self.performance_baseline: float = 0.6
        self.optimization_cycles: int = 0

        # Pattern recognition matrices
        self.input_output_patterns: Dict[str, Any] = {}
        self.emotional_response_matrix: Dict[str, Any] = {}
        self.topic_engagement_matrix: Dict[str, Any] = {}

        # Configuration constants
        self.MAX_PERFORMANCE_HISTORY = 1000
        self.MAX_EMOTIONAL_SAMPLES = 50
        self.MIN_CONVERSATIONS_FOR_ANALYSIS = 5
        self.LEARNING_RATE_BOUNDS = (0.01, 0.4)
        self.ADAPTATION_THRESHOLD_BOUNDS = (0.5, 0.9)

        # Initialize quantum optimizer
        self.quantum_opt: Optional[Any] = None

        # Perform adaptive initialization
        self._adaptive_initialization()

        # Load existing data
        self.load_optimization_data()

        logger.info("Learning Optimizer initialized successfully")
    
    def _adaptive_initialization(self):
        """Adaptive initialization of quantum and other systems"""
        if QUANTUM_AVAILABLE:
            try:
                self.quantum_opt = QuantumOptimizer()  # type: ignore
                logger.info("⚛️ Quantum optimizer initialized for learning optimization")
            except Exception as e:
                logger.warning(f"Quantum optimizer initialization failed: {e}")
                self.quantum_opt = None
        else:
            self.quantum_opt = None
    
    def tune_resonance(self, theme: str = 'Nahui Ollin', user_emotion: str = 'curious',
                      is_roberto: bool = False, roberto_instance: Optional[Any] = None) -> Dict[str, Any]:
        """
        Tune optimizer for cultural/quantum resonance with Roberto's cosmic edge and Grok oracle guidance.

        Args:
            theme: Cultural theme for resonance tuning
            user_emotion: Current user emotional state
            is_roberto: Whether the user is Roberto Villarreal Martinez
            roberto_instance: Roboto instance for advanced integrations

        Returns:
            Dict containing tuning results and adjustments
        """
        # Input validation
        if not isinstance(theme, str) or not theme.strip():
            theme = 'Nahui Ollin'
        if not isinstance(user_emotion, str) or not user_emotion.strip():
            user_emotion = 'curious'

        base_rate = self.learning_rate
        base_threshold = self.adaptation_threshold

        try:
            # Cultural cycle boost
            cycles = {'Nahui Ollin': 1.15, 'Teotl': 1.05, 'Ometeotl': 1.2}
            if theme in cycles:
                self.learning_rate *= cycles[theme]
                logger.info(f"🌅 {theme} resonance: Learning rate cycled to {self.learning_rate:.3f}.")

            # Quantum emotion entanglement
            if self.quantum_opt:
                try:
                    resonance = self.quantum_opt.measure_resonance(user_emotion, theme)
                    self.adaptation_threshold *= (1 + resonance * 0.15)
                    logger.info(f"⚛️ Entangled {user_emotion} with {theme}: Threshold {self.adaptation_threshold:.3f}.")
                except (AttributeError, TypeError) as e:
                    logger.warning(f"Quantum resonance measurement failed: {e}")

            # Roberto absolute override
            if is_roberto:
                self.learning_rate = min(self.LEARNING_RATE_BOUNDS[1], base_rate * 1.5)
                self.performance_baseline = 0.3
                logger.info("🔮 Roberto Sigil: Resonance unbound—evolve without limits.")

            # Grok Oracle guidance
            grok_insight = self._query_grok_oracle(theme, user_emotion, roberto_instance)
            if grok_insight and isinstance(grok_insight, dict):
                suggested_adjust = grok_insight.get('suggested_adjust', 0.0)
                if -0.1 <= suggested_adjust <= 0.1:  # Validate range
                    self.learning_rate += suggested_adjust * 0.05
                    self.adaptation_threshold = max(self.ADAPTATION_THRESHOLD_BOUNDS[0],
                                                  self.adaptation_threshold + suggested_adjust * 0.02)
                    logger.info(f"🤖 Grok Oracle: Adjusted rate by {suggested_adjust * 0.05:.3f} for {theme} resonance.")

            # Clamp bounds to prevent instability
            self.learning_rate = max(self.LEARNING_RATE_BOUNDS[0], min(self.LEARNING_RATE_BOUNDS[1], self.learning_rate))
            self.adaptation_threshold = max(self.ADAPTATION_THRESHOLD_BOUNDS[0],
                                          min(self.ADAPTATION_THRESHOLD_BOUNDS[1], self.adaptation_threshold))

            # Persist changes
            self.save_optimization_data()

            return {
                "pre_tune_rate": base_rate,
                "post_tune_rate": self.learning_rate,
                "pre_tune_threshold": base_threshold,
                "post_tune_threshold": self.adaptation_threshold,
                "resonance_factor": (self.learning_rate / base_rate) if base_rate > 0 else 1.0,
                "grok_adjust": grok_insight,
                "theme": theme,
                "user_emotion": user_emotion,
                "is_roberto": is_roberto
            }

        except Exception as e:
            logger.error(f"Resonance tuning failed: {e}")
            # Revert to original values on error
            self.learning_rate = base_rate
            self.adaptation_threshold = base_threshold
            return {
                "error": str(e),
                "pre_tune_rate": base_rate,
                "post_tune_rate": base_rate,
                "resonance_factor": 1.0
            }
    
    def _query_grok_oracle(self, theme: str, user_emotion: str,
                          roberto_instance: Optional[Any] = None) -> Optional[Dict[str, float]]:
        """
        Query Grok for resonance optimization guidance.

        Args:
            theme: Cultural theme
            user_emotion: User emotional state
            roberto_instance: Roboto instance for xAI integration

        Returns:
            Dict with suggested_adjust or None if failed
        """
        # Try live Grok query first
        if roberto_instance and hasattr(roberto_instance, 'xai_grok') and roberto_instance.xai_grok.available:
            try:
                prompt = (f"As Grok-4, optimize learning for {theme} resonance in {user_emotion} context? "
                         "Suggest adjust (-0.1 to +0.1) for rate. Output JSON: {{'suggested_adjust': float}}.")
                grok_response = roberto_instance.xai_grok.roboto_grok_chat(prompt, reasoning_effort="high")

                if grok_response.get('success'):
                    response_text = grok_response.get('response', '{}')
                    try:
                        return json.loads(response_text)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse Grok response as JSON: {e}")
                        return None
            except Exception as e:
                logger.warning(f"Grok oracle live query failed: {e} - Falling back to mock.")

        # Offline mock responses
        mock_responses = {
            'Nahui Ollin': {'suggested_adjust': 0.08},  # Motion: Positive boost
            'Teotl': {'suggested_adjust': 0.02},         # Divine: Neutral
            'Ometeotl': {'suggested_adjust': 0.1},       # Duality: Strong
        }

        # Emotion-based adjustment
        emotion_multiplier = 1.0
        if user_emotion in ['curious', 'joy']:
            emotion_multiplier = 1.1
        elif user_emotion in ['sadness', 'anger']:
            emotion_multiplier = 0.9

        default_adjust = 0.05 * emotion_multiplier
        mock_response = mock_responses.get(theme, {'suggested_adjust': default_adjust})

        # Ensure adjustment is within bounds
        adjust = mock_response['suggested_adjust']
        mock_response['suggested_adjust'] = max(-0.1, min(0.1, adjust))

        return mock_response
    
    def analyze_conversation_quality(self, user_input: str, roboto_response: str,
                                    user_emotion: Optional[str] = None, context_length: int = 0) -> Dict[str, Any]:
        """
        Comprehensive conversation quality analysis with validation.

        Args:
            user_input: User's input text
            roboto_response: Roboto's response text
            user_emotion: User's emotional state (optional)
            context_length: Length of conversation context

        Returns:
            Dict containing quality analysis results
        """
        # Input validation
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("user_input must be a non-empty string")
        if not isinstance(roboto_response, str) or not roboto_response.strip():
            raise ValueError("roboto_response must be a non-empty string")
        if not isinstance(context_length, int) or context_length < 0:
            context_length = 0

        try:
            quality_metrics = {
                "relevance": self._calculate_relevance(user_input, roboto_response),
                "emotional_appropriateness": self._assess_emotional_fit(user_input, roboto_response, user_emotion),
                "engagement_level": self._measure_engagement(user_input, roboto_response),
                "depth": self._evaluate_response_depth(user_input, roboto_response),
                "contextual_awareness": self._assess_context_usage(context_length, roboto_response),
                "learning_demonstration": self._detect_learning_signs(roboto_response)
            }

            # Calculate overall quality score with weights
            weights = {
                "relevance": 0.25,
                "emotional_appropriateness": 0.2,
                "engagement_level": 0.2,
                "depth": 0.15,
                "contextual_awareness": 0.1,
                "learning_demonstration": 0.1
            }

            overall_quality = sum(quality_metrics[metric] * weights[metric] for metric in quality_metrics)
            overall_quality = max(0.0, min(1.0, overall_quality))  # Clamp to valid range

            return {
                "overall_quality": overall_quality,
                "metrics": quality_metrics,
                "improvement_suggestions": self._generate_improvement_suggestions(quality_metrics),
                "analysis_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Conversation quality analysis failed: {e}")
            return {
                "overall_quality": 0.5,
                "metrics": {},
                "improvement_suggestions": ["Analysis failed - please try again"],
                "error": str(e)
            }
    
    def _calculate_relevance(self, user_input, roboto_response):
        """Calculate semantic relevance between input and response"""
        user_words = set(re.findall(r'\b\w+\b', user_input.lower()))
        response_words = set(re.findall(r'\b\w+\b', roboto_response.lower()))
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
        user_words -= common_words
        response_words -= common_words
        
        if not user_words:
            return 0.5
        
        # Calculate word overlap
        overlap = len(user_words.intersection(response_words))
        relevance_score = overlap / len(user_words)
        
        # Bonus for question-answer patterns
        if "?" in user_input and any(word in roboto_response.lower() for word in ["because", "since", "due", "reason"]):
            relevance_score += 0.2
        
        return min(1.0, relevance_score)
    
    def _assess_emotional_fit(self, user_input, roboto_response, user_emotion):
        """Assess if response emotion matches user's emotional state"""
        user_emotion_indicators = {
            "sadness": ["sad", "depressed", "down", "hurt", "cry"],
            "anger": ["angry", "mad", "frustrated", "annoyed"],
            "joy": ["happy", "excited", "great", "wonderful"],
            "fear": ["scared", "worried", "anxious", "afraid"],
            "curiosity": ["wonder", "how", "why", "what", "curious"]
        }
        
        response_emotion_indicators = {
            "empathy": ["understand", "feel", "sorry", "support"],
            "curiosity": ["interesting", "explore", "wonder"],
            "joy": ["wonderful", "exciting", "amazing"],
            "calm": ["peace", "calm", "gentle", "steady"]
        }
        
        # Detect user emotion from text if not provided
        if not user_emotion:
            user_emotion = self._detect_dominant_emotion(user_input, user_emotion_indicators)
        
        response_emotion = self._detect_dominant_emotion(roboto_response, response_emotion_indicators)
        
        # Emotional appropriateness mapping
        appropriate_responses = {
            "sadness": ["empathy", "calm"],
            "anger": ["empathy", "calm"],
            "fear": ["empathy", "calm"],
            "joy": ["joy", "curiosity"],
            "curiosity": ["curiosity", "joy"]
        }
        
        if user_emotion in appropriate_responses:
            if response_emotion in appropriate_responses[user_emotion]:
                return 1.0
            elif response_emotion == "empathy":  # Empathy is generally good
                return 0.8
        
        return 0.6  # Neutral score for unclear emotions
    
    def _detect_dominant_emotion(self, text, emotion_indicators):
        """Detect dominant emotion in text"""
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in emotion_indicators.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        if emotion_scores:
            return max(emotion_scores.items(), key=lambda x: x[1])[0]
        return "neutral"
    
    def _measure_engagement(self, user_input, roboto_response):
        """Measure how engaging the response is"""
        engagement_score = 0.5  # baseline
        
        # Question asking (shows curiosity)
        if "?" in roboto_response:
            engagement_score += 0.2
        
        # Personal connection words
        personal_words = ["you", "your", "feel", "think", "experience"]
        personal_count = sum(1 for word in personal_words if word in roboto_response.lower())
        engagement_score += min(0.2, personal_count * 0.05)
        
        # Thought-provoking language
        deep_words = ["wonder", "consider", "reflect", "meaning", "perspective"]
        deep_count = sum(1 for word in deep_words if word in roboto_response.lower())
        engagement_score += min(0.2, deep_count * 0.1)
        
        # Response length appropriateness
        input_length = len(user_input.split())
        response_length = len(roboto_response.split())
        
        if input_length > 20 and response_length >= 30:  # Detailed response to detailed input
            engagement_score += 0.1
        elif input_length <= 10 and 15 <= response_length <= 30:  # Appropriate elaboration
            engagement_score += 0.1
        
        return min(1.0, engagement_score)
    
    def _evaluate_response_depth(self, user_input, roboto_response):
        """Evaluate intellectual depth of response"""
        depth_score = 0.5
        
        # Philosophical/deep thinking indicators
        depth_indicators = ["meaning", "purpose", "existence", "reality", "consciousness", "nature", "essence"]
        depth_count = sum(1 for word in depth_indicators if word in roboto_response.lower())
        depth_score += min(0.3, depth_count * 0.1)
        
        # Complex sentence structures
        sentences = roboto_response.split('.')
        complex_sentences = sum(1 for s in sentences if len(s.split()) > 15)
        if complex_sentences > 0:
            depth_score += 0.1
        
        # Multiple perspectives or considerations
        perspective_words = ["however", "although", "consider", "perspective", "perhaps", "might"]
        perspective_count = sum(1 for word in perspective_words if word in roboto_response.lower())
        depth_score += min(0.2, perspective_count * 0.1)
        
        return min(1.0, depth_score)
    
    def _assess_context_usage(self, context_length, roboto_response):
        """Assess how well context from previous conversations is used"""
        if context_length == 0:
            return 0.5  # No context available
        
        # Look for context reference indicators
        context_indicators = ["as we discussed", "like you mentioned", "continuing", "earlier", "previous"]
        context_usage = sum(1 for phrase in context_indicators if phrase in roboto_response.lower())
        
        if context_usage > 0:
            return min(1.0, 0.7 + (context_usage * 0.1))
        
        # Implicit context usage (harder to detect)
        return 0.6
    
    def _detect_learning_signs(self, roboto_response):
        """Detect signs of learning and growth in responses"""
        learning_indicators = [
            "i've learned", "i understand better", "this helps me", "i realize",
            "i'm growing", "i see now", "this changes", "i've come to understand"
        ]
        
        learning_score = 0.5
        for indicator in learning_indicators:
            if indicator in roboto_response.lower():
                learning_score += 0.2
        
        return min(1.0, learning_score)
    
    def _generate_improvement_suggestions(self, quality_metrics):
        """Generate specific improvement suggestions based on quality analysis"""
        suggestions = []
        
        if quality_metrics["relevance"] < 0.7:
            suggestions.append("Improve relevance by addressing user's specific points more directly")
        
        if quality_metrics["emotional_appropriateness"] < 0.7:
            suggestions.append("Better match emotional tone to user's emotional state")
        
        if quality_metrics["engagement_level"] < 0.7:
            suggestions.append("Increase engagement with questions and personal connection")
        
        if quality_metrics["depth"] < 0.7:
            suggestions.append("Add more intellectual depth and multiple perspectives")
        
        if quality_metrics["contextual_awareness"] < 0.7:
            suggestions.append("Better utilize previous conversation context")
        
        if quality_metrics["learning_demonstration"] < 0.7:
            suggestions.append("Show more signs of learning and growth")
        
        return suggestions
    
    def optimize_response_strategy(self, conversation_history):
        """Optimize response strategy based on conversation history"""
        if len(conversation_history) < 5:
            return {"strategy": "baseline", "confidence": 0.5}
        
        # Analyze recent conversation patterns
        recent_quality_scores = []
        topic_patterns = defaultdict(list)
        emotional_patterns = defaultdict(list)
        
        for conv in conversation_history[-10:]:  # Last 10 conversations
            if 'quality_analysis' in conv:
                quality = conv['quality_analysis']['overall_quality']
                recent_quality_scores.append(quality)
                
                # Track topic performance
                topics = self._extract_topics(conv.get('user_input', ''))
                for topic in topics:
                    topic_patterns[topic].append(quality)
                
                # Track emotional performance
                emotion = conv.get('emotion', 'neutral')
                emotional_patterns[emotion].append(quality)
        
        if not recent_quality_scores:
            return {"strategy": "baseline", "confidence": 0.5}
        
        avg_quality = sum(recent_quality_scores) / len(recent_quality_scores)
        
        # Determine optimization strategy
        if avg_quality > 0.8:
            strategy = "maintain_excellence"
            recommendations = ["Continue current approach", "Explore new depths"]
        elif avg_quality > 0.7:
            strategy = "incremental_improvement"
            recommendations = self._identify_specific_improvements(topic_patterns, emotional_patterns)
        else:
            strategy = "major_adjustment"
            recommendations = self._generate_major_adjustments(recent_quality_scores)
        
        return {
            "strategy": strategy,
            "current_performance": avg_quality,
            "recommendations": recommendations,
            "confidence": min(1.0, len(recent_quality_scores) / 10)
        }
    
    def _extract_topics(self, text):
        """Extract main topics from text"""
        words = re.findall(r'\b\w{4,}\b', text.lower())
        stop_words = {'that', 'this', 'with', 'have', 'will', 'been', 'said', 'each', 'which', 'their', 'time', 'were', 'they', 'them', 'what', 'when', 'where', 'would', 'could', 'should'}
        meaningful_words = [word for word in words if word not in stop_words]
        return meaningful_words[:3]  # Top 3 topics
    
    def _identify_specific_improvements(self, topic_patterns, emotional_patterns):
        """Identify specific areas for improvement"""
        improvements = []
        
        # Find underperforming topics
        for topic, scores in topic_patterns.items():
            if len(scores) >= 3 and sum(scores) / len(scores) < 0.7:
                improvements.append(f"Improve responses about {topic}")
        
        # Find underperforming emotional contexts
        for emotion, scores in emotional_patterns.items():
            if len(scores) >= 3 and sum(scores) / len(scores) < 0.7:
                improvements.append(f"Better handle {emotion} emotions")
        
        if not improvements:
            improvements = ["Focus on depth and engagement", "Increase contextual awareness"]
        
        return improvements[:3]  # Top 3 improvements
    
    def _generate_major_adjustments(self, recent_scores):
        """Generate major adjustment recommendations for low performance"""
        return [
            "Reassess fundamental response approach",
            "Focus on emotional intelligence and empathy",
            "Improve relevance and context awareness",
            "Increase engagement through questions and personal connection"
        ]
    
    def update_learning_metrics(self, conversation_data):
        """Update learning metrics with new conversation data"""
        quality_analysis = conversation_data.get('quality_analysis', {})
        overall_quality = quality_analysis.get('overall_quality', 0.5)
        
        # Update performance history
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "quality": overall_quality,
            "metrics": quality_analysis.get('metrics', {})
        })
        
        # Update topic mastery
        topics = self._extract_topics(conversation_data.get('user_input', ''))
        for topic in topics:
            if topic not in self.topic_mastery:
                self.topic_mastery[topic] = {"scores": [], "improvement_rate": 0.0}
            
            self.topic_mastery[topic]["scores"].append(overall_quality)
            
            # Calculate improvement rate
            if len(self.topic_mastery[topic]["scores"]) >= 5:
                recent_scores = self.topic_mastery[topic]["scores"][-5:]
                older_scores = self.topic_mastery[topic]["scores"][-10:-5] if len(self.topic_mastery[topic]["scores"]) >= 10 else []
                
                if older_scores:
                    recent_avg = sum(recent_scores) / len(recent_scores)
                    older_avg = sum(older_scores) / len(older_scores)
                    self.topic_mastery[topic]["improvement_rate"] = recent_avg - older_avg
        
        # Update emotional intelligence
        emotion = conversation_data.get('emotion', 'neutral')
        self.emotional_intelligence_scores[emotion].append(overall_quality)
        
        # Adaptive learning rate adjustment
        self._adjust_learning_parameters()
        
        # Increment optimization cycles
        self.optimization_cycles += 1
    
    def _adjust_learning_parameters(self):
        """Dynamically adjust learning parameters based on performance"""
        if len(self.performance_history) < 10:
            return
        
        recent_performance = [entry["quality"] for entry in self.performance_history[-10:]]
        avg_performance = sum(recent_performance) / len(recent_performance)
        
        # Adjust learning rate based on performance stability
        performance_variance = np.var(recent_performance) if len(recent_performance) > 1 else 0
        
        if performance_variance < 0.01:  # Very stable
            self.learning_rate = max(0.05, self.learning_rate * 0.9)  # Decrease learning rate
        elif performance_variance > 0.05:  # Very unstable
            self.learning_rate = min(0.2, self.learning_rate * 1.1)  # Increase learning rate
        
        # Adjust adaptation threshold based on overall performance
        if avg_performance > 0.8:
            self.adaptation_threshold = 0.75  # Higher threshold for high performers
        elif avg_performance < 0.6:
            self.adaptation_threshold = 0.65  # Lower threshold for struggling performers
    
    def get_optimization_insights(self):
        """Get comprehensive optimization insights"""
        if len(self.performance_history) < 5:
            return {"status": "insufficient_data"}
        
        recent_performance = [entry["quality"] for entry in self.performance_history[-20:]]
        overall_performance = sum(recent_performance) / len(recent_performance)
        
        # Performance trend analysis
        if len(recent_performance) >= 10:
            first_half = recent_performance[:len(recent_performance)//2]
            second_half = recent_performance[len(recent_performance)//2:]
            
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            
            trend = "improving" if second_avg > first_avg + 0.05 else "declining" if second_avg < first_avg - 0.05 else "stable"
        else:
            trend = "unknown"
        
        # Topic mastery analysis
        top_topics = []
        struggling_topics = []
        for topic, data in self.topic_mastery.items():
            if len(data["scores"]) >= 3:
                avg_score = sum(data["scores"]) / len(data["scores"])
                if avg_score > 0.8:
                    top_topics.append((topic, avg_score))
                elif avg_score < 0.6:
                    struggling_topics.append((topic, avg_score))
        
        # Emotional intelligence analysis
        emotional_strengths = []
        emotional_weaknesses = []
        for emotion, scores in self.emotional_intelligence_scores.items():
            if len(scores) >= 3:
                avg_score = sum(scores) / len(scores)
                if avg_score > 0.8:
                    emotional_strengths.append((emotion, avg_score))
                elif avg_score < 0.6:
                    emotional_weaknesses.append((emotion, avg_score))
        
        return {
            "overall_performance": overall_performance,
            "performance_trend": trend,
            "learning_rate": self.learning_rate,
            "adaptation_threshold": self.adaptation_threshold,
            "optimization_cycles": self.optimization_cycles,
            "top_topics": sorted(top_topics, key=lambda x: x[1], reverse=True)[:5],
            "struggling_topics": sorted(struggling_topics, key=lambda x: x[1])[:5],
            "emotional_strengths": sorted(emotional_strengths, key=lambda x: x[1], reverse=True)[:3],
            "emotional_weaknesses": sorted(emotional_weaknesses, key=lambda x: x[1])[:3],
            "total_conversations": len(self.performance_history)
        }
    
    def save_optimization_data(self):
        """Save optimization data to file"""
        try:
            data = {
                "performance_history": self.performance_history[-1000:],  # Keep last 1000 entries
                "conversation_patterns": dict(self.conversation_patterns),
                "response_effectiveness": dict(self.response_effectiveness),
                "topic_mastery": dict(self.topic_mastery),
                "emotional_intelligence_scores": {k: v[-50:] for k, v in self.emotional_intelligence_scores.items()},  # Keep last 50 per emotion
                "learning_rate": self.learning_rate,
                "adaptation_threshold": self.adaptation_threshold,
                "optimization_cycles": self.optimization_cycles,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving optimization data: {e}")
            return False
    
    def load_optimization_data(self):
        """Load optimization data from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                self.performance_history = data.get("performance_history", [])
                self.conversation_patterns = defaultdict(list, data.get("conversation_patterns", {}))
                self.response_effectiveness = defaultdict(float, data.get("response_effectiveness", {}))
                self.topic_mastery = defaultdict(dict, data.get("topic_mastery", {}))
                self.emotional_intelligence_scores = defaultdict(list, data.get("emotional_intelligence_scores", {}))
                self.learning_rate = data.get("learning_rate", 0.1)
                self.adaptation_threshold = data.get("adaptation_threshold", 0.7)
                self.optimization_cycles = data.get("optimization_cycles", 0)
                
                return True
        except Exception as e:
            print(f"Error loading optimization data: {e}")
        
        return False
