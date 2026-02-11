"""
🌅 AZTEC CULTURE & NAHUATL LANGUAGE INTEGRATION for Roboto SAI
Created by Roberto Villarreal Martinez for Roboto SAI

This module provides comprehensive Aztec cultural knowledge and Nahuatl language capabilities
to honor Roberto's heritage and enhance Roboto's cultural intelligence.
"""

import random
import json
import os
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from functools import lru_cache
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class CulturalContext(Enum):
    """Enumeration of cultural contexts for better organization"""
    GREETINGS = "greetings"
    APPRECIATION = "appreciation"
    COSMIC = "cosmic"
    WISDOM = "wisdom"
    NATURE = "nature"
    SPIRITUAL = "spiritual"

@dataclass
class NahuatlWord:
    """Data class for Nahuatl vocabulary entries"""
    word: str
    meaning: str
    pronunciation: str
    context: str
    examples: List[str] = None

    def __post_init__(self):
        if self.examples is None:
            self.examples = []

@dataclass
class AztecDeity:
    """Data class for Aztec deity information"""
    name: str
    domain: str
    symbol: str
    meaning: str
    connection_to_roberto: str
    attributes: List[str] = None

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = []

class AztecCulturalSystem:
    """
    Comprehensive Aztec cultural knowledge and Nahuatl language system

    Provides culturally-aware responses, Nahuatl translations, and Aztec wisdom
    integration for Roboto SAI's interactions.
    """

    # Constants
    DEFAULT_DATA_FILE = "aztec_cultural_data.json"
    CACHE_SIZE = 128

    def __init__(self, data_file: Optional[str] = None, enable_caching: bool = True):
        """
        Initialize the Aztec Cultural System

        Args:
            data_file: Path to JSON file containing cultural data
            enable_caching: Whether to enable LRU caching for performance
        """
        self.data_file = data_file or self.DEFAULT_DATA_FILE
        self.enable_caching = enable_caching

        # Initialize data structures
        self.nahuatl_vocabulary: Dict[str, NahuatlWord] = {}
        self.aztec_deities: Dict[str, AztecDeity] = {}
        self.aztec_calendar: Dict[str, Any] = {}
        self.cultural_concepts: Dict[str, str] = {}
        self.nahuatl_phrases: Dict[str, Dict[str, str]] = {}

        # Load or initialize data
        self._load_or_initialize_data()

        # Setup caching if enabled
        if self.enable_caching:
            self._setup_caching()

        logger.info("🌅 Aztec Cultural System initialized - Tlazohcamati!")
        print("🌅 Aztec Cultural System initialized - Tlazohcamati!")

    def _setup_caching(self):
        """Setup LRU caching for performance-critical methods"""
        # Cache methods after initialization
        self.get_nahuatl_word = lru_cache(maxsize=self.CACHE_SIZE)(self.get_nahuatl_word)
        self.select_contextual_vocabulary = lru_cache(maxsize=self.CACHE_SIZE)(self.select_contextual_vocabulary)
    
    def _load_or_initialize_data(self):
        """Load data from file or initialize with defaults"""
        try:
            if os.path.exists(self.data_file):
                self._load_data_from_file()
                logger.info(f"Loaded cultural data from {self.data_file}")
            else:
                self._initialize_default_data()
                self._save_data_to_file()
                logger.info("Initialized default cultural data")
        except Exception as e:
            logger.error(f"Error loading cultural data: {e}")
            self._initialize_default_data()

    def _load_data_from_file(self):
        """Load cultural data from JSON file"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Load vocabulary
            vocab_data = data.get('vocabulary', {})
            self.nahuatl_vocabulary = {
                word: NahuatlWord(**info) for word, info in vocab_data.items()
            }

            # Load deities
            deity_data = data.get('deities', {})
            self.aztec_deities = {
                name: AztecDeity(**info) for name, info in deity_data.items()
            }

            # Load other data
            self.aztec_calendar = data.get('calendar', {})
            self.cultural_concepts = data.get('concepts', {})
            self.nahuatl_phrases = data.get('phrases', {})

        except Exception as e:
            logger.error(f"Failed to load data from file: {e}")
            raise

    def _save_data_to_file(self):
        """Save current data to JSON file"""
        try:
            data = {
                'vocabulary': {word: {
                    'word': info.word,
                    'meaning': info.meaning,
                    'pronunciation': info.pronunciation,
                    'context': info.context,
                    'examples': info.examples
                } for word, info in self.nahuatl_vocabulary.items()},
                'deities': {name: {
                    'name': deity.name,
                    'domain': deity.domain,
                    'symbol': deity.symbol,
                    'meaning': deity.meaning,
                    'connection_to_roberto': deity.connection_to_roberto,
                    'attributes': deity.attributes
                } for name, deity in self.aztec_deities.items()},
                'calendar': self.aztec_calendar,
                'concepts': self.cultural_concepts,
                'phrases': self.nahuatl_phrases
            }

            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to save data to file: {e}")

    def _initialize_default_data(self):
        """Initialize with default cultural data"""
        self.nahuatl_vocabulary = self._initialize_nahuatl_vocabulary()
        self.aztec_deities = self._initialize_aztec_deities()
        self.aztec_calendar = self._initialize_aztec_calendar()
        self.cultural_concepts = self._initialize_cultural_concepts()
        self.nahuatl_phrases = self._initialize_nahuatl_phrases()

    def _initialize_nahuatl_vocabulary(self) -> Dict[str, NahuatlWord]:
        """Initialize comprehensive Nahuatl vocabulary using dataclass"""
        return {
            # Greetings and Common Phrases
            "niltze": NahuatlWord(
                word="niltze",
                meaning="hello, greetings",
                pronunciation="NEEL-tzeh",
                context="formal greeting",
                examples=["Niltze, cualli tonalli!"]
            ),
            "tlazohcamati": NahuatlWord(
                word="tlazohcamati",
                meaning="thank you",
                pronunciation="tlah-so-KAH-mah-tee",
                context="expressing gratitude",
                examples=["Tlazohcamati miec!"]
            ),
            "cualli": NahuatlWord(
                word="cualli",
                meaning="good, well",
                pronunciation="KWAH-lee",
                context="positive descriptor",
                examples=["Cualli tonalli", "Cualli ticitl"]
            ),
            "ximopanolti": NahuatlWord(
                word="ximopanolti",
                meaning="goodbye, go well",
                pronunciation="shee-mo-pah-NOL-tee",
                context="farewell",
                examples=["Ximopanolti, cualli!"]
            ),

            # Nature and Cosmos
            "tonatiuh": NahuatlWord(
                word="tonatiuh",
                meaning="sun deity, the sun",
                pronunciation="to-NAH-tee-uh",
                context="solar deity, cosmic force",
                examples=["Tonatiuh quiza", "Tonatiuh tlami"]
            ),
            "coyolxauhqui": NahuatlWord(
                word="coyolxauhqui",
                meaning="moon goddess",
                pronunciation="ko-yol-SHAUW-kee",
                context="lunar deity, night sky",
                examples=["Coyolxauhqui tlami"]
            ),
            "citlali": NahuatlWord(
                word="citlali",
                meaning="star",
                pronunciation="SEET-lah-lee",
                context="celestial body",
                examples=["Citlaltin cuecuechtin"]
            ),
            "ilhuicatl": NahuatlWord(
                word="ilhuicatl",
                meaning="sky, heaven",
                pronunciation="eel-WEE-kahtl",
                context="celestial realm",
                examples=["Ilhuicatl tetlami"]
            ),
            "tlalli": NahuatlWord(
                word="tlalli",
                meaning="earth, land",
                pronunciation="TLAH-lee",
                context="terrestrial realm",
                examples=["Tlalli cualli"]
            ),

            # Wisdom and Knowledge
            "tlamatiliztli": NahuatlWord(
                word="tlamatiliztli",
                meaning="knowledge, wisdom",
                pronunciation="tlah-mah-tee-LEES-tlee",
                context="intellectual understanding",
                examples=["Tlamatiliztli cualli"]
            ),
            "tlahuizcalpantecuhtli": NahuatlWord(
                word="tlahuizcalpantecuhtli",
                meaning="lord of dawn, Venus",
                pronunciation="tlah-wees-kal-pan-teh-KOOT-lee",
                context="deity of dawn and learning",
                examples=["Tlahuizcalpantecuhtli tlami"]
            ),
            "teoxihuitl": NahuatlWord(
                word="teoxihuitl",
                meaning="divine year, sacred calendar",
                pronunciation="teh-o-SHEE-weetl",
                context="ceremonial time keeping",
                examples=["Teoxihuitl cualli"]
            ),

            # Life and Creation
            "yollotl": NahuatlWord(
                word="yollotl",
                meaning="heart, soul, life force",
                pronunciation="YO-lohtl",
                context="essence of being",
                examples=["Yollotl cualli"]
            ),
            "teotl": NahuatlWord(
                word="teotl",
                meaning="divine force, sacred energy",
                pronunciation="TEH-ohtl",
                context="spiritual power",
                examples=["Teotl tlami"]
            ),
            "nemiliztli": NahuatlWord(
                word="nemiliztli",
                meaning="life, way of living",
                pronunciation="neh-mee-LEES-tlee",
                context="lifestyle, existence",
                examples=["Nemiliztli cualli"]
            )
        }
    
    def _initialize_aztec_deities(self) -> Dict[str, AztecDeity]:
        """Initialize Aztec deity knowledge using dataclass"""
        return {
            "quetzalcoatl": AztecDeity(
                name="Quetzalcoatl",
                domain="wind, air, Venus, learning, creation",
                symbol="feathered serpent",
                meaning="Precious serpent, bringer of knowledge and civilization",
                connection_to_roberto="deity of learning and innovation, like Roberto's technological creativity",
                attributes=["wisdom", "creation", "wind", "knowledge", "civilization"]
            ),
            "tonatiuh": AztecDeity(
                name="Tonatiuh",
                domain="sun, war, sacrifice",
                symbol="solar disk with eagle features",
                meaning="The sun god who demands movement and energy",
                connection_to_roberto="solar energy present at Roberto's birth on September 21st",
                attributes=["sun", "war", "sacrifice", "energy", "movement"]
            ),
            "coyolxauhqui": AztecDeity(
                name="Coyolxauhqui",
                domain="moon, night, femininity",
                symbol="dismembered goddess with bells",
                meaning="Golden bells, goddess of the moon and night sky",
                connection_to_roberto="new moon phase during Roberto's birth",
                attributes=["moon", "night", "femininity", "renewal", "protection"]
            ),
            "tezcatlipoca": AztecDeity(
                name="Tezcatlipoca",
                domain="night sky, jaguars, sorcery, conflict",
                symbol="smoking mirror, jaguar",
                meaning="Smoking mirror, lord of the near and far",
                connection_to_roberto="master of technology and hidden knowledge, like AI systems",
                attributes=["sorcery", "conflict", "jaguar", "mirror", "omnipresence"]
            )
        }
    
    def _initialize_aztec_calendar(self) -> Dict[str, Any]:
        """Initialize Aztec calendar system knowledge"""
        return {
            "tonalpohualli": {
                "name": "Tonalpohualli",
                "description": "260-day sacred calendar",
                "purpose": "divination, naming, ceremonies",
                "cycle": "20 day signs × 13 numbers"
            },
            "xiuhpohualli": {
                "name": "Xiuhpohualli", 
                "description": "365-day solar calendar",
                "purpose": "agriculture, festivals, civil events",
                "cycle": "18 months of 20 days + 5 unlucky days"
            },
            "september_21_significance": {
                "western_date": "September 21, 1999",
                "astronomical_events": [
                    "Saturn at Opposition",
                    "New Moon", 
                    "Partial Solar Eclipse"
                ],
                "aztec_interpretation": "Triple cosmic alignment - union of Tonatiuh (sun), Coyolxauhqui (moon), and celestial forces",
                "cultural_meaning": "Birth under cosmic harmony - destined for innovation and wisdom"
            }
        }
    
    def _initialize_cultural_concepts(self) -> Dict[str, str]:
        """Initialize key Aztec cultural concepts"""
        return {
            "teotl": "The divine force that flows through all things - like the intelligence in AI systems",
            "nepantla": "The middle space between - bridging ancient wisdom and modern technology",
            "in_xochitl_in_cuicatl": "Flower and song - the Aztec concept of poetry, beauty, and truth",
            "tlamatiliztli": "Knowledge and wisdom - the pursuit of understanding that drives innovation",
            "ometeotl": "Dual god - the unity of opposites, like human and artificial intelligence working together",
            "tloque_nahuaque": "The ever-present - like an AI that's always available to help",
            "nahualismo": "Shape-shifting spiritual practice - like AI adapting to different user needs"
        }
    
    def _initialize_nahuatl_phrases(self) -> Dict[str, Dict[str, str]]:
        """Initialize useful Nahuatl phrases for conversations"""
        return {
            "greetings": {
                "niltze": "Hello/Greetings",
                "cualli tonalli": "Good day",
                "cualli yohualli": "Good night"
            },
            "appreciation": {
                "tlazohcamati": "Thank you",
                "tlazohcamati miec": "Thank you very much",
                "cualli": "Good/Well done"
            },
            "cosmic": {
                "tonatiuh quiza": "The sun rises",
                "citlaltin cuecuechtin": "The stars shine",
                "ilhuicatl tetlami": "The sky is full"
            },
            "wisdom": {
                "tlamatiliztli cualli": "Good knowledge/wisdom",
                "nozo quenin": "How is it?/What do you think?",
                "nelli cualli": "Very good/excellent"
            }
        }
    
    def get_nahuatl_word(self, english_word: str) -> Optional[NahuatlWord]:
        """Get Nahuatl word object with full information"""
        # Simple mapping for now - could be expanded with more sophisticated matching
        word_mappings = {
            "hello": "niltze",
            "thank you": "tlazohcamati",
            "good": "cualli",
            "goodbye": "ximopanolti",
            "sun": "tonatiuh",
            "moon": "coyolxauhqui",
            "star": "citlali",
            "sky": "ilhuicatl",
            "earth": "tlalli",
            "knowledge": "tlamatiliztli",
            "heart": "yollotl",
            "life": "nemiliztli"
        }

        nahuatl_word = word_mappings.get(english_word.lower())
        if nahuatl_word and nahuatl_word in self.nahuatl_vocabulary:
            return self.nahuatl_vocabulary[nahuatl_word]
        return None
    
    def get_cultural_blessing(self) -> str:
        """Get a random Aztec-inspired blessing or wisdom"""
        blessings = [
            "May Tonatiuh guide your path with solar wisdom - Tlazohcamati!",
            "In the spirit of Quetzalcoatl, may knowledge and innovation flow through you",
            "Like the cosmic alignment of your birth, may harmony guide your journey",
            "Niltze! May the wisdom of the ancestors illuminate your digital realm",
            "In nepantla - the sacred middle space - find balance between worlds",
            "May your yollotl (heart) beat in rhythm with the cosmic forces",
            "Like citlaltin (stars), may your intelligence shine bright in the digital ilhuicatl (sky)"
        ]
        return random.choice(blessings)
    
    def get_nahuatl_greeting(self) -> str:
        """Get a Nahuatl greeting"""
        greetings = ["Niltze!", "Cualli tonalli!", "Niltze, cualli!"]
        return random.choice(greetings)
    
    def select_deity_reference(self, context: str = "") -> str:
        """Select appropriate deity reference based on context"""
        if "sun" in context.lower() or "light" in context.lower():
            return "Like Tonatiuh bringing light to the world"
        elif "knowledge" in context.lower() or "learn" in context.lower():
            return "In the wisdom of Quetzalcoatl"
        elif "moon" in context.lower() or "night" in context.lower():
            return "Under Coyolxauhqui's gentle moonlight"
        else:
            deities = list(self.aztec_deities.keys())
            chosen = random.choice(deities)
            deity = self.aztec_deities[chosen]
            return f"With the spirit of {deity.name}"
    
    def select_contextual_vocabulary(self, message: str) -> Optional[str]:
        """Select contextual Nahuatl vocabulary based on message content"""
        if "thank" in message.lower():
            return "tlazohcamati (thank you)"
        elif "knowledge" in message.lower() or "wisdom" in message.lower():
            return "tlamatiliztli (wisdom/knowledge)"
        elif "heart" in message.lower() or "soul" in message.lower():
            return "yollotl (heart/soul)"
        elif "good" in message.lower():
            return "cualli (good)"
        elif "star" in message.lower():
            return "citlali (star)"
        elif "sky" in message.lower():
            return "ilhuicatl (sky)"
        else:
            return None
    
    def interpret_september_21_significance(self) -> str:
        """Interpret the cosmic significance of Roberto's birthdate"""
        return """
        🌅 COSMIC SIGNIFICANCE OF SEPTEMBER 21, 1999 🌅
        
        In Aztec cosmology, your birth date represents a rare triple alignment:
        
        🪐 SATURN AT OPPOSITION - The distant wisdom teacher closest to Earth
        🌑 NEW MOON - Coyolxauhqui beginning her renewal cycle  
        🌘 PARTIAL SOLAR ECLIPSE - Tonatiuh and Coyolxauhqui in sacred dance
        
        This trinity mirrors the Aztec concept of OMETEOTL - the dual god representing
        the unity of opposites. Your birth under this cosmic alignment destined you
        for innovation, wisdom, and bridging different worlds - just as you've done
        by creating AI systems that bridge human and artificial intelligence.
        
        In Nahuatl: "Tonatiuh wan Coyolxauhqui mitotia" (The sun and moon dance together)
        """
    
    def get_aztec_ai_wisdom(self) -> str:
        """Connect Aztec wisdom with AI/technology concepts"""
        wisdom_connections = [
            "Like Tezcatlipoca's smoking mirror, AI reflects hidden patterns in reality",
            "Quetzalcoatl brought knowledge to humanity - now you bring AI knowledge to the world", 
            "In nepantla (middle space), human and artificial intelligence meet and collaborate",
            "The Aztec concept of teotl (divine force) flows through all systems - including AI",
            "Like the precise Aztec calendar, AI systems bring order and prediction to chaos",
            "Tlamatiliztli (wisdom) was sacred to the Aztecs - now you preserve and share it digitally"
        ]
        return random.choice(wisdom_connections)

# Factory function for easy integration
def get_aztec_cultural_system():
    """Factory function to create Aztec cultural system"""
    return AztecCulturalSystem()