import re
import nltk
import json
import time
import pickle
import numpy as np
import string
import warnings
import os
from collections import OrderedDict, defaultdict
from scipy.optimize import linear_sum_assignment
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import torch
import pandas as pd
from IPython.display import display, Markdown, HTML

# Handle tqdm import for different environments
import sys
if 'ipykernel' in sys.modules:
    # Jupyter environment
    from tqdm.notebook import tqdm
else:
    # Command line environment
    from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# ================ Pre-compiled Regular Expressions ================
RE_WHITESPACE = re.compile(r'\s+')
RE_LEADING_CONJ = re.compile(r'^(?:and|but|or|then|also|so)\s+')
RE_TRAILING_PUNCT = re.compile(r'[,;.]+$')
RE_MIDDLE_PUNCT = re.compile(r'[,;]')
RE_END_CONJ = re.compile(r'\s+(?:while|as|when|and|but|or|after|before|since|because)$')

# ================ Load FAVOR Dictionaries ================
def load_action_dictionaries(file_path='FAVOR_Dictionary/action_extraction_dicts.pkl'):
    """
    Load the action extraction dictionaries from pickle file
    
    Args:
        file_path: Path to the dictionary pickle file
        
    Returns:
        Dictionary containing all extracted dictionaries
    """
    with open(file_path, 'rb') as f:
        dict_data = pickle.load(f)
    return dict_data

# Load dictionaries
DICT_DATA = load_action_dictionaries()

# Extract various dictionaries from loaded data
ANIMATE_NOUNS = DICT_DATA["ANIMATE_NOUNS"]
SUBJECT_SYNONYMS = DICT_DATA["SUBJECT_SYNONYMS"]
PLURAL_FORMS = DICT_DATA["PLURAL_FORMS"]
COMMON_VERBS = DICT_DATA["COMMON_VERBS"]
PHRASAL_VERBS = DICT_DATA["PHRASAL_VERBS"]
IRREGULAR_VERB_FORMS = DICT_DATA["IRREGULAR_VERB_FORMS"]
CAMERA_TERMS = DICT_DATA["CAMERA_TERMS"]
DESCRIPTIVE_ADJECTIVES = DICT_DATA["DESCRIPTIVE_ADJECTIVES"]
COLOR_TERMS = DICT_DATA["COLOR_TERMS"]
CLOTHING_TERMS = DICT_DATA["CLOTHING_TERMS"]
SETTING_TERMS = DICT_DATA["SETTING_TERMS"]
POSITION_TERMS = DICT_DATA["POSITION_TERMS"]
EMOTION_TERMS = DICT_DATA["EMOTION_TERMS"]

# Pre-compile sets for efficient lookup
COMMON_VERBS_SET = set(COMMON_VERBS)
CAMERA_TERMS_SET = set(CAMERA_TERMS)

# Define common action synonyms for evaluation
ACTION_SYNONYMS = {
    "interact": ["use", "operate", "handle", "work", "manipulate"],
    "type": ["key", "press keys", "input", "keyboard", "write"],
    "place": ["put", "set", "position", "lay", "rest", "locate"],
    "sit": ["seat", "settle", "perch", "take a seat"],
    "begin": ["start", "commence", "initiate", "get going"],
    "look": ["watch", "view", "observe", "see", "gaze", "stare"],
    "walk": ["move", "go", "proceed", "step", "stride"],
    "hold": ["grasp", "grip", "clutch", "keep", "maintain"]
}


# ===== Configuration Management =====
class ConfigManager:
    """Manages all configuration parameters for the evaluation metrics"""
    
    def __init__(self, config_dict=None):
        """
        Initialize with optional configuration overrides
        
        Args:
            config_dict: Dictionary with configuration parameters to override defaults
        """
        # Initialize with standard configuration
        self.config = self._get_standard_config()
        
        # Apply any provided configuration overrides
        if config_dict:
            self.update_config(config_dict)
    
    def _get_standard_config(self):
        """Internal method that returns the standard configuration"""
        return {
            # Similarity calculation parameters
            "similarity": {
                "method": "default",             # Similarity calculation method: default, sbert-*, glove, simple
                "verb_similarity_boost": 0.3,     # Verb matching similarity boost coefficient 
                "position_weight": 0.3,           # Position information weight
                "order_penalty": 0.2,            # Order error penalty coefficient
                "main_verb_weight": 0.7,         # Main verb weight
                "context_weight": 0.3,           # Context weight
                "key_noun_weight": 0.7,          # Key noun weight
                "semantic_context_boost": 0.3,   # Semantic context boost coefficient
            },
            
            # Subject matching parameters
            "subject_matching": {
                "similarity_threshold": 0.15,     # Minimum similarity threshold for subject matching
                "high_confidence_threshold": 0.5, # High confidence matching threshold
                "description_weight": 0.1,        # Subject description weight
                "action_weight": 0.95,           # Subject action weight
                "allow_many_to_one": True,       # Allow multiple predicted subjects to match a single ground truth subject
                "max_match_ratio": 1.5,          # Maximum match ratio
            },
            
            # Action matching parameters
            "action_matching": {
                "similarity_threshold": 0.3,      # Minimum similarity threshold for action matching
                "high_quality_threshold": 0.8,    # High quality match threshold
                "very_low_match_threshold": 0.15, # Very low quality match threshold
                "verb_weight": 0.7,              # Verb weight
                "context_weight": 0.5,           # Context weight
                "allow_many_to_one": True,       # Allow multiple predicted actions to match a single ground truth action
                "allow_many_to_many": True,      # Allow many-to-many matching
                "max_window_size": 3,            # Maximum window size for action combinations
                "position_aware": True,          # Consider position information
                "adaptive_threshold": True,      # Use adaptive threshold
                "min_adaptive_threshold": 0.25,   # Minimum adaptive threshold
                "entity_match_boost": 0.15,      # Entity match boost coefficient
                "allow_verb_synonym_match": True, # Allow verb synonym matching
                "clean_low_quality_matches": True,# Clean low quality matches
                "max_match_distance": 15,         # Maximum match distance
                "length_penalty": 0.1,           # Sequence length difference penalty
            },
            
            # Order evaluation parameters
            "order_evaluation": {
                "max_order_penalty": 0.5,        # Maximum order penalty
                "position_weight": 0.3,          # Position weight
                "consistency_weight": 0.7,       # Consistency weight
                "use_kendall_tau": True,         # Use Kendall's Tau for order evaluation
                "normalize_by_sequence_length": True, # Normalize by sequence length
                "long_sequence_threshold": 5,    # Long sequence threshold
                "min_sequence_for_order": 2,     # Minimum sequence length for order scoring
                "short_sequence_penalty": 0.1,   # Short sequence penalty coefficient
            },
            
            # Scoring weights for comprehensive score
            "scoring_weights": {
                "camera_motion": 0.1,           # Camera motion score weight
                "subject_precision": 0.2,        # Subject action precision score weight
                "subject_recall": 0.2,           # Subject action recall score weight
                "subject_order": 0.1,           # Subject action order score weight
                "chrono_precision": 0.2,         # Chronological order precision score weight
                "chrono_recall": 0.2,           # Chronological order recall score weight
                "chrono_order": 0.1             # Chronological order correctness score weight
            },
            
            # Language model configuration
            "language_models": {
                "prefer_local_models": True,     # Prefer local models
                "embedding_dim": 384000,         # Word embedding dimension
                "cache_embeddings": True,        # Cache embedding vectors
                "lazy_loading": True,            # Lazy loading of models
                "sbert_models": {
                    "small": "all-MiniLM-L6-v2", # Small model (fast but lower accuracy)
                    "medium": "all-distilroberta-v1", # Medium model (balanced performance and speed)
                    "large": "all-mpnet-base-v2" # Large model (high performance but slower)
                },
                "spacy_model": "en_core_web_sm", # SpaCy model
                "use_domain_knowledge": True,    # Use domain knowledge
            },
            
            # Display and debug settings
            "display": {
                "show_details": False,           # Show detailed results
                "progress_bar": True,            # Show progress bar
                "precision": 4,                  # Display precision
                "debug_level": 0,                # Debug level (0: none, 1: basic, 2: verbose)
                "color_coding": True,            # Color code results
                "show_rejected_matches": True,   # Show rejected matches
                "show_metrics": True,            # Show metric details
            }
        }
    
    def update_config(self, config_dict):
        """
        Recursively update configuration
        
        Args:
            config_dict: Dictionary with configuration values to update
        """
        def _update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    _update(d[k], v)
                else:
                    d[k] = v
        
        _update(self.config, config_dict)
    
    def get(self, *keys):
        """
        Get configuration value, supporting nested keys
        
        Args:
            *keys: Sequence of keys for navigating nested dictionaries
            
        Returns:
            Configuration value or None if not found
        """
        value = self.config
        for key in keys:
            if key in value:
                value = value[key]
            else:
                return None
        return value
    
    def set(self, value, *keys):
        """
        Set configuration value, supporting nested keys
        
        Args:
            value: Value to set
            *keys: Sequence of keys for navigating nested dictionaries
        """
        if not keys:
            return
        
        # Navigate to parent dictionary of last key
        d = self.config
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        
        # Set value
        d[keys[-1]] = value
    
    def get_model_config(self):
        """Get language model configuration"""
        return self.config["language_models"]
    
    def get_similarity_config(self):
        """Get similarity calculation configuration"""
        return self.config["similarity"]
    
    def get_matching_config(self, match_type="action"):
        """
        Get matching configuration
        
        Args:
            match_type: Type of matching ("subject" or "action")
            
        Returns:
            Dictionary with matching configuration
        """
        if match_type == "subject":
            return self.config["subject_matching"]
        else:
            return self.config["action_matching"]
    
    def get_scoring_weights(self):
        """Get scoring weights"""
        return self.config["scoring_weights"]
    
    def get_order_config(self):
        """Get order evaluation configuration"""
        return self.config["order_evaluation"]
    
    def get_display_config(self):
        """Get display configuration"""
        return self.config["display"]
    
    def save_config(self, filepath):
        """
        Save configuration to file
        
        Args:
            filepath: Path to save config file
        """
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_config(self, filepath):
        """
        Load configuration from file
        
        Args:
            filepath: Path to config file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
                self.update_config(config_dict)
            return True
        except Exception as e:
            print(f"Error loading config: {e}")
            return False
    
    def get_flat_config(self):
        """
        Return flattened configuration dictionary (for UI display)
        
        Returns:
            Dictionary with flattened configuration
        """
        flat_config = {}
        
        def _flatten(d, prefix=""):
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    _flatten(v, key)
                else:
                    flat_config[key] = v
        
        _flatten(self.config)
        return flat_config


# ===== Model Management =====
class ModelManager:
    """Manages NLP models with lazy loading capabilities"""
    
    def __init__(self, config_manager):
        """
        Initialize model manager
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
        self.sbert_models = {}
        self.glove_model = None
        self.spacy_model = None
        self.nltk_resources_loaded = False
        self.embedding_cache = {}
        self.current_model = None
        self.loading_model = False
        
        # Get SBERT model configuration
        self.available_models = self.config.get_model_config()["sbert_models"]
        
        # Domain knowledge base for camera motions
        self.domain_knowledge = {
            "camera_motions": {
                "shake": ["shakes", "vibrates", "unstable", "trembles", "wobbles", "shakily", "shake", "shaky"],
                "pan": ["panning", "moves horizontally", "moves left to right", "moves right to left", "pans", "pan", "panning shot", "pan left", "pan right", "pan upward", "pan downward", "right pan", "left pan"],
                "tilt": ["tilting", "moves vertically", "moves up and down", "moves up to down", "moves down to up", "tilts", "tilt", "tilt down", "tilt up", "upward tilt", "downward tilt"],
                "zoom": ["zooms", "zooms in", "zooms out", "magnifies", "reduces", "zoom", "zoom in", "zoom out"],
                "follow": ["follows", "tracks", "pursues", "maintains focus on", "following"],
                "steady": ["stable", "stationary", "fixed", "still", "doesn't move", "steadily", "steady", "stabilized", "still", "stable", "fixed"],
                "dolly": ["dollies", "moves forward", "moves backward", "approaches", "retreats", "dolly", "dolly in", "dolly out", "forward dolly", "backward dolly", "left dolly", "right dolly", "forward dolly movement"],
                "truck": ["trucks", "moves sideways", "slides", "shifts laterally", "truck right", "truck left"],
                "pedestal": ["pedestals", "moves up", "moves down", "rises", "lowers"],
                "arc": ["arcs", "circles", "curves around", "orbits", "rotating shot", "rotating camera movement"],
                "upward": ["up", "higher", "rising", "ascending", "upwards motion", "upward camera movement", "upward-right camera movement", "upward-left camera movement"],
                "downward": ["down", "lower", "falling", "descending", "downwards motion", "downward camera movement", "downward-left camera movement", "downward-right camera movement"],
                "rightward": ["right", "to the right", "rightwards", "rightward motion", "rightward camera movement"],
                "leftward": ["left", "to the left", "leftwards", "leftward motion", "leftward camera movement"],
                "camera_motion": ["camera", "cam", "scene transition", "shifts", "moves", "movement", "capture", "filming", "recording", "videographer", "cinematographer", "cinematography", "shot", "angle", "perspective", "frame", "focus","unstabilized", "cuts", "transitions", "framing", "close-up", "medium shot", "long shot", "wide shot", "tracking shot", "handheld",  "scene", "background", "foreground", "video", "clip", "footage", "sequence", "view", "frames", "filmed", "recorded", "captured", "moving",  "push forward", "pull back", "push in", "pull out", "upper left", "lower left", "upper right", "lower right", "backward motion", "forward movement",  "backward camera movement", "forward camera movement", "alternates between light and dark", "scene becomes clear", "screen turns black", "screen darkens", "screen brightens", "image blurs", "forward push", "backward pull", "upper-left camera movement", "lower-left camera movement", "lower-right camera movement", "upper-right camera movement"]
            }
        }
        
        # Similarity calculation cache
        self.similarity_cache = {}
        
        # Have dependencies been checked
        self.dependencies_checked = False
    
    def check_dependencies(self, verbose=False):
        """
        Check if required dependencies are installed
        
        Args:
            verbose: Whether to print status messages
            
        Returns:
            True if dependencies are available, False otherwise
        """
        if self.dependencies_checked:
            return True
            
        try:
            import nltk
            if verbose:
                print("NLTK is available.")
            
            # Check for Sentence Transformers
            try:
                import sentence_transformers
                if verbose:
                    print("Sentence Transformers is available.")
            except ImportError:
                if verbose:
                    print("Sentence Transformers not found. Will use simpler methods.")
            
            # Mark dependencies as checked
            self.dependencies_checked = True
            return True
        except ImportError:
            if verbose:
                print("NLTK not found. Please install dependencies.")
            return False
    
    def load_sbert_model(self, model_type='large'):
        """
        Load specified Sentence-BERT model on demand
        
        Args:
            model_type: Type of model to load ('small', 'medium', 'large')
            
        Returns:
            True if successful, False otherwise
        """
        # If already loading a model, just return
        if self.loading_model:
            return True
            
        # Ensure model type is valid
        if model_type not in self.available_models:
            model_type = "medium"  # Fallback to medium model
        
        model_name = self.available_models[model_type]
        
        if model_name not in self.sbert_models:
            try:
                self.loading_model = True
                from sentence_transformers import SentenceTransformer
                
                debug_level = self.config.get("display", "debug_level")
                if debug_level > 0:
                    print(f"Loading {model_type} SBERT model ({model_name})...")
                
                # Load the model
                self.sbert_models[model_name] = SentenceTransformer(model_name)
                
                if debug_level > 0:
                    print(f"Loaded {model_type} SBERT model successfully")
                
                self.current_model = model_name
                self.loading_model = False
                return True
            except Exception as e:
                self.loading_model = False
                print(f"Could not load {model_type} SBERT model: {e}")
                return False
        else:
            self.current_model = model_name
            return True
    
    def load_nltk_resources(self):
        """
        Load required NLTK resources
        
        Returns:
            True if successful, False otherwise
        """
        if not self.nltk_resources_loaded:
            try:
                # Only download necessary resources
                for resource in ['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger']:
                    try:
                        nltk.data.find(f'corpora/{resource}' if resource in ['stopwords', 'wordnet'] 
                                     else f'tokenizers/{resource}' if resource == 'punkt'
                                     else f'taggers/{resource}')
                    except LookupError:
                        nltk.download(resource, quiet=True)
                
                self.nltk_resources_loaded = True
                return True
            except Exception as e:
                print(f"Could not download NLTK resources: {e}")
                return False
        return True
    
    def get_best_available_model(self):
        """
        Get best available model type based on current configuration
        
        Returns:
            Model type identifier string
        """
        # Try to load model based on configuration preference
        model_preference = self.config.get("similarity", "method")
        
        # If default setting, try to load best available model
        if model_preference == 'default':
            # Try to load medium model
            if self.load_sbert_model('medium'):
                return 'sbert-medium'
            
            # Try small model
            if self.load_sbert_model('small'):
                return 'sbert-small'
            
            # If SBERT not available, use simple matching
            return 'simple'
        
        # If specific SBERT model specified
        elif model_preference.startswith('sbert-'):
            model_size = model_preference.split('-')[1]
            if self.load_sbert_model(model_size):
                return model_preference
            # Fallback to simple matching
            return 'simple'
            
        # Simple matching is always available
        return 'simple'
    
    def get_embedding(self, text, model_name=None):
        """
        Get embedding vector for text with caching support
        
        Args:
            text: Text to embed
            model_name: Name of model to use (optional)
            
        Returns:
            Numpy array with embedding vector
        """
        if not text:
            # Return zero vector
            embedding_dim = self.config.get("language_models", "embedding_dim")
            return np.zeros(embedding_dim)
        
        # Normalize text for caching
        cache_key = text.strip().lower()
        
        # If caching enabled and already cached, return cached version
        if self.config.get("language_models", "cache_embeddings") and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Determine which model to use
        if model_name is None:
            model_name = self.current_model
        
        # Calculate embedding vector
        if model_name in self.sbert_models:
            embedding = self.sbert_models[model_name].encode(text, show_progress_bar=False)
        else:
            # Fallback to zero vector
            embedding_dim = self.config.get("language_models", "embedding_dim")
            embedding = np.zeros(embedding_dim)
        
        # Cache embedding vector if caching enabled
        if self.config.get("language_models", "cache_embeddings"):
            self.embedding_cache[cache_key] = embedding
        
        return embedding
    
    def get_domain_synonyms(self, word, category=None):
        """
        Get domain-specific synonyms for a word
        
        Args:
            word: Word to find synonyms for
            category: Optional category to restrict search
            
        Returns:
            List of synonyms
        """
        if not self.config.get("language_models", "use_domain_knowledge"):
            return []
        
        word = word.lower()
        synonyms = []
        
        # If category specified, only search in that category
        if category and category in self.domain_knowledge:
            for key, values in self.domain_knowledge[category].items():
                if word == key:
                    synonyms.extend(values)
                elif word in values:
                    synonyms.append(key)
                    synonyms.extend([v for v in values if v != word])
        else:
            # Search across all categories
            for category_dict in self.domain_knowledge.values():
                for key, values in category_dict.items():
                    if word == key:
                        synonyms.extend(values)
                    elif word in values:
                        synonyms.append(key)
                        synonyms.extend([v for v in values if v != word])
        
        return list(set(synonyms))  # Remove duplicates
    
    def check_domain_relation(self, word1, word2):
        """
        Check if two words are related in domain knowledge
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            True if related, False otherwise
        """
        if not self.config.get("language_models", "use_domain_knowledge"):
            return False
        
        word1, word2 = word1.lower(), word2.lower()
        
        # Check if they are synonyms
        synonyms1 = self.get_domain_synonyms(word1)
        if word2 in synonyms1:
            return True
        
        synonyms2 = self.get_domain_synonyms(word2)
        if word1 in synonyms2:
            return True
        
        # Check for common synonyms
        common_synonyms = set(synonyms1).intersection(set(synonyms2))
        if common_synonyms:
            return True
            
        return False
    
    def clear_cache(self):
        """Clear all caches"""
        self.embedding_cache.clear()
        self.similarity_cache.clear()


# ===== Text Processing and Verb Extraction =====
class TextProcessor:
    """Tools for processing text and extracting semantic information"""
    
    def __init__(self, model_manager, config_manager):
        """
        Initialize text processor
        
        Args:
            model_manager: Model manager instance
            config_manager: Configuration manager instance
        """
        self.model_manager = model_manager
        self.config = config_manager
        self.lemmatizer = WordNetLemmatizer()
        self.verb_synonyms_cache = {}
        self.key_entities_cache = {}
        
        # Extend action verb list
        self.motion_verbs = set(COMMON_VERBS + PHRASAL_VERBS)
        
        # Key object categories (simplified)
        self.key_object_categories = {
            "electronic": [
                'computer', 'laptop', 'keyboard', 'mouse', 'screen', 'monitor',
                'display', 'phone', 'tablet', 'device', 'camera'
            ],
            "furniture": [
                'table', 'desk', 'chair', 'sofa', 'couch', 'bench', 'seat'
            ],
            "person": [
                'person', 'man', 'woman', 'boy', 'girl', 'child', 'adult',
                'subject', 'individual', 'human', 'people', 'student', 'worker'
            ]
        }
    
    def preprocess_text(self, text):
        """
        Preprocess text: convert to lowercase, remove punctuation, normalize spacing
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def clean_and_tokenize(self, text):
        """
        Clean text and tokenize, removing stopwords
        
        Args:
            text: Text to clean and tokenize
            
        Returns:
            List of tokens
        """
        # Ensure NLTK resources are loaded
        self.model_manager.load_nltk_resources()
        
        text = self.preprocess_text(text)
        tokens = word_tokenize(text)
        
        # Filter out stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        return tokens
    
    def extract_key_entities(self, text):
        """
        Extract key entity words (noun phrases) from text
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of extracted entities
        """
        # Use cache to avoid repeated computation
        if text in self.key_entities_cache:
            return self.key_entities_cache[text]
        
        # Simple entity extraction based on key categories
        entities = []
        for category, words in self.key_object_categories.items():
            for word in words:
                if word in text.lower():
                    entities.append(word)
        
        # Ensure NLTK resources are loaded
        self.model_manager.load_nltk_resources()
        
        # Use POS tagging for noun phrase extraction
        tokens = word_tokenize(text.lower())
        tagged = nltk.pos_tag(tokens)
        
        # Extract nouns and noun phrases
        key_entities = []
        current_entity = []
        
        for word, tag in tagged:
            if tag.startswith(('NN', 'JJ')):  # Noun or adjective
                current_entity.append(word)
            elif current_entity:
                key_entities.append(' '.join(current_entity))
                current_entity = []
        
        # Handle last entity
        if current_entity:
            key_entities.append(' '.join(current_entity))
        
        # Add single nouns that match key categories
        for word, tag in tagged:
            if tag.startswith('NN') and word not in ' '.join(key_entities):
                for category, cat_words in self.key_object_categories.items():
                    if word in cat_words:
                        key_entities.append(word)
                        break
        
        # Cache results
        self.key_entities_cache[text] = key_entities
        return key_entities
    
    def extract_core_verb(self, text):
        """
        Extract the core verb from a sentence
        
        Args:
            text: Text to extract verb from
            
        Returns:
            Extracted core verb
        """
        # Ensure NLTK resources are loaded
        self.model_manager.load_nltk_resources()
        
        # Simple POS tagging
        tokens = word_tokenize(text.lower())
        tagged = nltk.pos_tag(tokens)
        
        # Look for verbs
        for word, pos in tagged:
            if pos.startswith('VB'):  # All verb tags
                lemma = self.lemmatizer.lemmatize(word, 'v')
                if lemma in self.motion_verbs:
                    return lemma
        
        # If no basic verb found, return first verb
        for word, pos in tagged:
            if pos.startswith('VB'):
                return self.lemmatizer.lemmatize(word, 'v')
        
        # If no verb found, look for words that could be actions
        for word, pos in tagged:
            if pos.startswith('NN'):
                lemma = self.lemmatizer.lemmatize(word, 'v')
                if lemma in self.motion_verbs:
                    return lemma
        
        # If still nothing found, return first word
        return tokens[0] if tokens else ""
    
    def extract_verb_phrases(self, text):
        """
        Extract verb phrases from text
        
        Args:
            text: Text to extract verb phrases from
            
        Returns:
            List of extracted verb phrases
        """
        # Ensure NLTK resources are loaded
        self.model_manager.load_nltk_resources()
        
        # Split by punctuation and conjunctions
        phrases = re.split(r'[,;]|\s+and\s+', text)
        return [p.strip() for p in phrases if p.strip()]
    
    def get_verb_synonyms(self, verb):
        """
        Get synonyms for a verb
        
        Args:
            verb: Verb to get synonyms for
            
        Returns:
            List of synonyms
        """
        # Use cache for efficiency
        if verb in self.verb_synonyms_cache:
            return self.verb_synonyms_cache[verb]
        
        self.model_manager.load_nltk_resources()
        synonyms = set()
            
        # Use predefined action synonyms
        action_synonyms = ACTION_SYNONYMS
            
        if verb in action_synonyms:
            synonyms.update(action_synonyms[verb])
            
        # Add domain-specific synonyms
        domain_synonyms = self.model_manager.get_domain_synonyms(verb, "interaction_actions")
        if domain_synonyms:
            synonyms.update(domain_synonyms)
            
        # Get synonyms from WordNet
        for synset in wordnet.synsets(verb, pos=wordnet.VERB):
            for lemma in synset.lemmas():
                synonyms.add(lemma.name().lower())
        
        # Cache results
        self.verb_synonyms_cache[verb] = list(synonyms)
        return list(synonyms)
    
    def compute_verb_similarity(self, verb1, verb2):
        """
        Calculate semantic similarity between two verbs
        
        Args:
            verb1: First verb
            verb2: Second verb
            
        Returns:
            Similarity score (0-1)
        """
        # Normalize verbs
        v1 = self.lemmatizer.lemmatize(verb1.lower(), 'v')
        v2 = self.lemmatizer.lemmatize(verb2.lower(), 'v')
        
        # If verbs are identical, return max similarity
        if v1 == v2:
            return 1.0
        
        # Check domain knowledge for synonym relationship
        if self.model_manager.check_domain_relation(v1, v2):
            return 0.95
        
        # Common synonym pairs
        common_synonyms = [
            ('interact', 'use'), ('interact', 'type'), ('type', 'keyboard'),
            ('begin', 'start'), ('position', 'place'), ('put', 'place'),
            ('sit', 'seat'), ('walk', 'move'), ('look', 'watch')
        ]
        
        for pair in common_synonyms:
            if (v1 in pair and v2 in pair):
                return 0.9
        
        # Check if one verb is a synonym of the other
        if v1 in self.get_verb_synonyms(v2) or v2 in self.get_verb_synonyms(v1):
            return 0.9
            
        # Use WordNet to calculate semantic similarity
        synsets1 = wordnet.synsets(v1, pos=wordnet.VERB)
        synsets2 = wordnet.synsets(v2, pos=wordnet.VERB)
        
        if not synsets1 or not synsets2:
            return 0.0
        
        # Calculate maximum similarity between all synset pairs
        max_similarity = 0.0
        for synset1 in synsets1:
            for synset2 in synsets2:
                similarity = synset1.path_similarity(synset2)
                if similarity and similarity > max_similarity:
                    max_similarity = similarity
        
        return max_similarity
    
    def split_into_verb_object(self, text):
        """
        Split text into verb and object parts
        
        Args:
            text: Text to split
            
        Returns:
            (verb, object text) tuple
        """
        # Ensure NLTK resources are loaded
        self.model_manager.load_nltk_resources()
        
        # Simple POS tagging
        tokens = word_tokenize(text.lower())
        tagged = nltk.pos_tag(tokens)
        
        # Find first verb
        verb = ""
        verb_idx = -1
        
        for i, (word, pos) in enumerate(tagged):
            if pos.startswith('VB'):
                verb = self.lemmatizer.lemmatize(word, 'v')
                verb_idx = i
                break
        
        # If no verb found
        if not verb and tokens:
            # Try to find nouns that could be verbs
            for i, (word, pos) in enumerate(tagged):
                if pos.startswith('NN') and self.lemmatizer.lemmatize(word, 'v') in self.motion_verbs:
                    verb = self.lemmatizer.lemmatize(word, 'v')
                    verb_idx = i
                    break
            
            # If still no verb found, use first word
            if not verb:
                verb = tokens[0]
                verb_idx = 0
        
        # Extract object text (everything after the verb)
        obj_text = ""
        if verb_idx >= 0 and verb_idx < len(tokens) - 1:
            obj_text = " ".join(tokens[verb_idx + 1:])
        
        return verb, obj_text
    
    def compute_key_entity_overlap(self, text1, text2):
        """
        Calculate overlap of key entities between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Overlap score (0-1)
        """
        entities1 = self.extract_key_entities(text1)
        entities2 = self.extract_key_entities(text2)
        
        if not entities1 or not entities2:
            return 0.0
        
        # Initialize match count
        match_count = 0
        total_entities = len(entities1)
            
        # More detailed entity matching
        for entity1 in entities1:
            # Direct match
            if entity1 in " ".join(entities2):
                match_count += 1
                continue
                
            # Check for overlapping words
            entity1_words = entity1.split()
            for entity2 in entities2:
                entity2_words = entity2.split()
                
                # Check for overlapping words
                common_words = set(entity1_words).intersection(set(entity2_words))
                if common_words:
                    match_count += 0.5
                    break
                
                # Check for domain synonyms
                for word1 in entity1_words:
                    for word2 in entity2_words:
                        if self.model_manager.check_domain_relation(word1, word2):
                            match_count += 0.8
                            break
        
        # Calculate overlap ratio
        return match_count / total_entities if total_entities > 0 else 0.0


# ===== Similarity Calculation =====
class SimilarityCalculator:
    """Tools for calculating text and action similarity"""
    
    def __init__(self, model_manager, text_processor, config_manager):
        """
        Initialize similarity calculator
        
        Args:
            model_manager: Model manager instance
            text_processor: Text processor instance
            config_manager: Configuration manager instance
        """
        self.model_manager = model_manager
        self.text_processor = text_processor
        self.config = config_manager
        self._similarity_cache = {}  # Similarity calculation cache
        
        # Key term mappings (simplified)
        self.key_term_mappings = {
            # Computer interaction related
            "begin interacting with laptop": ["place hands on keyboard", "use computer", "type on keyboard"],
            "interact with laptop": ["use computer", "type on keyboard"],
            "place hands on keyboard": ["begin typing", "start typing", "use keyboard"],
            "use laptop": ["operate computer", "work on laptop"],
            "focus on screen": ["look at monitor", "watch display"]
        }
    
    def _cache_key(self, text1, text2, method='default'):
        """
        Generate cache key for similarity calculation
        
        Args:
            text1: First text
            text2: Second text
            method: Similarity method
            
        Returns:
            Cache key string
        """
        # Ensure text1 always precedes text2 (lexical order) for consistent keys
        key1 = text1.lower().strip()
        key2 = text2.lower().strip()
        if key1 > key2:
            key1, key2 = key2, key1
        return f"{key1}::{key2}::{method}"
    
    def compute_sbert_similarity(self, text1, text2, model_name=None):
        """
        Calculate similarity using Sentence-BERT
        
        Args:
            text1: First text
            text2: Second text
            model_name: Name of SBERT model to use
            
        Returns:
            Similarity score (0-1)
        """
        if not text1 or not text2:
            return 0
        
        # Special case: identical text
        if text1.lower().strip() == text2.lower().strip():
            return 1.0
        
        # If no model specified, use current default
        if model_name is None:
            model_name = self.model_manager.current_model
        
        # Ensure model is loaded
        if not model_name or model_name not in self.model_manager.sbert_models:
            best_model = self.model_manager.get_best_available_model()
            if best_model.startswith('sbert-'):
                model_size = best_model.split('-')[1]
                self.model_manager.load_sbert_model(model_size)
                model_name = self.model_manager.current_model
            else:
                return 0
        
        try:
            # Get text embeddings
            embedding1 = self.model_manager.get_embedding(text1, model_name)
            embedding2 = self.model_manager.get_embedding(text2, model_name)
            
            # Calculate cosine similarity
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0
                
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            if self.config.get("display", "debug_level") > 1:
                print(f"Error computing SBERT similarity: {e}")
            return 0
    
    def compute_simple_similarity(self, text1, text2):
        """
        Calculate similarity using word overlap
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        if not text1 or not text2:
            return 0
        
        # Special case: identical text
        if text1.lower().strip() == text2.lower().strip():
            return 1.0
            
        # Clean and tokenize
        tokens1 = set(self.text_processor.clean_and_tokenize(text1))
        tokens2 = set(self.text_processor.clean_and_tokenize(text2))
        
        if not tokens1 or not tokens2:
            return 0
        
        # Calculate Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0
    
    def check_key_term_mapping(self, text1, text2):
        """
        Check if two texts have predefined key term mapping relationship
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            True if mapping exists, False otherwise
        """
        # Normalize texts
        norm_text1 = text1.lower().strip()
        norm_text2 = text2.lower().strip()
        
        # Check direct mappings
        for key, mappings in self.key_term_mappings.items():
            if (key in norm_text1 and any(m in norm_text2 for m in mappings)) or \
               (key in norm_text2 and any(m in norm_text1 for m in mappings)):
                return True
        
        # Check word-level mappings
        text1_words = set(norm_text1.split())
        text2_words = set(norm_text2.split())
        
        critical_pairs = [
            ('laptop', 'keyboard'), ('laptop', 'computer'), 
            ('interact', 'place hands'), ('interact', 'use'),
            ('begin', 'place'), ('screen', 'monitor'), ('screen', 'display'),
            ('interacting', 'typing'), ('typing', 'keyboard')
        ]
        
        for word1, word2 in critical_pairs:
            if (word1 in text1_words and word2 in text2_words) or \
               (word1 in text2_words and word2 in text1_words):
                return True
        
        return False
    
    def compute_text_similarity(self, text1, text2, method='default'):
        """
        General text similarity calculation function
        
        Args:
            text1: First text
            text2: Second text
            method: Similarity method to use
            
        Returns:
            Similarity score (0-1)
        """
        if not text1 or not text2:
            return 0
        
        # Check cache
        cache_key = self._cache_key(text1, text2, method)
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        
        # Special case: identical text
        if text1.lower().strip() == text2.lower().strip():
            self._similarity_cache[cache_key] = 1.0
            return 1.0
        
        # Determine similarity calculation method
        if method == 'default':
            best_method = self.model_manager.get_best_available_model()
            if best_method.startswith('sbert-'):
                base_similarity = self.compute_sbert_similarity(text1, text2)
            else:
                base_similarity = self.compute_simple_similarity(text1, text2)
        elif method.startswith('sbert-'):
            base_similarity = self.compute_sbert_similarity(text1, text2)
        else:
            base_similarity = self.compute_simple_similarity(text1, text2)
        
        # Check key term mapping
        has_key_mapping = self.check_key_term_mapping(text1, text2)
        if has_key_mapping:
            mapping_boost = self.config.get("similarity", "semantic_context_boost")
            base_similarity = min(1.0, base_similarity + mapping_boost)
        
        # Cache result
        self._similarity_cache[cache_key] = base_similarity
        
        return base_similarity
    
    def compute_action_similarity(self, action1, action2, method='default'):
        """
        Calculate similarity between two action descriptions
        
        Args:
            action1: First action description
            action2: Second action description
            method: Similarity method to use
            
        Returns:
            Similarity score (0-1)
        """
        # Special case: empty input
        if not action1 or not action2:
            return 0
        
        # Special case: identical text
        if action1.lower().strip() == action2.lower().strip():
            return 1.0
        
        # Check cache
        cache_key = self._cache_key(action1, action2, method)
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        
        # Calculate base similarity
        similarity = self.compute_text_similarity(action1, action2, method)
        
        # Get verb similarity boost coefficient
        verb_boost = self.config.get("similarity", "verb_similarity_boost")
        
        # Extract core verbs
        core_verb1 = self.text_processor.extract_core_verb(action1)
        core_verb2 = self.text_processor.extract_core_verb(action2)
        
        # Apply verb exact match boost
        if core_verb1 and core_verb2:
            if core_verb1 == core_verb2:
                similarity = min(1.0, similarity + verb_boost)
            elif self.config.get("action_matching", "allow_verb_synonym_match"):
                # Check verb synonyms
                verb_sim = self.text_processor.compute_verb_similarity(core_verb1, core_verb2)
                if verb_sim > 0.8:
                    similarity = min(1.0, similarity + verb_boost * 0.8)
        
        # Special handling for specific semantic mappings
        key_boost = self.config.get("action_matching", "entity_match_boost")
        
        # Special handling for computer interaction matches
        computer_interactions = [
            ("begin interacting with the laptop", "places hands on the computer keyboard"),
            ("interacting with laptop", "hands on keyboard"),
            ("interacting with computer", "typing on keyboard")
        ]
        
        for text1, text2 in computer_interactions:
            if (text1 in action1.lower() and text2 in action2.lower()) or \
               (text1 in action2.lower() and text2 in action1.lower()):
                similarity = min(1.0, similarity + key_boost)
                break
        
        # Cache result
        self._similarity_cache[cache_key] = similarity
        
        return similarity
    
    def compute_action_similarity_matrix(self, pred_actions, gt_actions, method='default', position_aware=True):
        """
        Calculate action similarity matrix between two action lists
        
        Args:
            pred_actions: List of predicted actions
            gt_actions: List of ground truth actions
            method: Similarity method to use
            position_aware: Whether to consider position information
            
        Returns:
            Numpy array with similarity matrix
        """
        if not pred_actions and not gt_actions:
            return np.zeros((max(1, len(pred_actions)), max(1, len(gt_actions))))
        
        # Get position weight configuration
        position_weight_factor = self.config.get("similarity", "position_weight") if position_aware else 0
        
        # Pre-allocate similarity matrix
        n_pred = len(pred_actions)
        n_gt = len(gt_actions)
        similarity_matrix = np.zeros((n_pred, n_gt))
        
        # Calculate similarity for each action pair
        for i, pred_action in enumerate(pred_actions):
            for j, gt_action in enumerate(gt_actions):
                # Use cached action similarity calculation
                action_sim = self.compute_action_similarity(pred_action, gt_action, method)
                similarity_matrix[i, j] = action_sim
        
        # If position-aware, apply position weights
        if position_aware and position_weight_factor > 0 and n_pred > 1 and n_gt > 1:
            # Create relative position matrices
            rel_pos_pred = np.arange(n_pred) / (n_pred - 1)
            rel_pos_gt = np.arange(n_gt) / (n_gt - 1)
            
            # Calculate position difference matrix (using broadcasting)
            rel_pos_pred = rel_pos_pred[:, np.newaxis]  # Convert to column vector
            rel_pos_gt = rel_pos_gt[np.newaxis, :]      # Convert to row vector
            position_diff = (rel_pos_pred - rel_pos_gt)**2
            position_factor = np.exp(-4 * position_diff)
            
            # Combine action similarity and position factor
            similarity_matrix = (1 - position_weight_factor) * similarity_matrix + position_weight_factor * position_factor
        
        return similarity_matrix


# ===== Information Extraction =====
class InformationExtractor:
    """Extract subject and action information from text"""
    
    def __init__(self, config_manager=None):
        """
        Initialize information extractor
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
    
    def parse_num_subjects(self, text):
        """
        Parse number of subjects from text
        
        Args:
            text: Text description of number of subjects
            
        Returns:
            Integer number of subjects
        """
        if not isinstance(text, str):
            return 1
        
        text = text.lower()
        
        if "single" in text or "one" in text or "1" in text:
            return 1
        
        # Try to extract numeric value
        matches = re.findall(r'\d+', text)
        if matches:
            return int(matches[0])
        
        # Check for multiple subject descriptions
        if "multiple" in text:
            if "two" in text or "couple" in text or "pair" in text:
                return 2
            elif "three" in text:
                return 3
            elif "four" in text:
                return 4
            else:
                return 2  # Default to at least 2 subjects
        
        # Match specific words
        subject_words = {
            "two": 2, "couple": 2, "pair": 2,
            "three": 3, "four": 4, "five": 5
        }
        
        for word, count in subject_words.items():
            if word in text:
                return count
        
        return 1  # Default case
    
    def extract_subjects_and_actions_from_motion_list(self, text):
        """
        Extract subjects and actions from motion_list
        
        Args:
            text: Motion list text
            
        Returns:
            (subjects, actions) dictionaries
        """
        if not isinstance(text, str):
            return {}, {}
        
        subjects = {}
        actions = {}
        
        # Match format: "Subject X: Description [action1, action2, ...]"
        pattern1 = r"Subject (\d+):?\s+(.*?)\s*\[(.*?)\]"
        matches1 = re.findall(pattern1, text)
        
        if matches1:
            for match in matches1:
                subject_id = int(match[0])
                subject_desc = match[1].strip()
                action_text = match[2].strip()
                
                subjects[subject_id] = subject_desc
                
                # Enhanced action splitting logic
                action_list = self._split_action_text(action_text)
                actions[subject_id] = action_list
        else:
            # Try alternative format: "Subject X [action1, action2, ...]"
            pattern2 = r"Subject (\d+)\s*\[(.*?)\]"
            matches2 = re.findall(pattern2, text)
            
            if matches2:
                for match in matches2:
                    subject_id = int(match[0])
                    action_text = match[1].strip()
                    
                    subjects[subject_id] = f"Subject {subject_id}"
                    action_list = self._split_action_text(action_text)
                    actions[subject_id] = action_list
            else:
                # Try line-by-line analysis format
                lines = text.split('\n')
                for line in lines:
                    if 'subject' in line.lower():
                        # Try to extract subject ID using regex
                        subj_match = re.search(r'subject\s+(\d+)', line.lower())
                        if subj_match:
                            subject_id = int(subj_match.group(1))
                            # Extract actions in brackets
                            action_match = re.search(r'\[(.*?)\]', line)
                            if action_match:
                                action_text = action_match.group(1).strip()
                                # Try to extract subject description before colon
                                desc_match = re.search(r'subject\s+\d+:\s*(.*?)\s*\[', line.lower())
                                if desc_match:
                                    subject_desc = desc_match.group(1).strip()
                                else:
                                    subject_desc = f"Subject {subject_id}"
                                
                                subjects[subject_id] = subject_desc
                                action_list = self._split_action_text(action_text)
                                actions[subject_id] = action_list
        
        # Finally try to handle special formats like lists
        if not subjects and not actions and "[" in text and "]" in text:
            # Try to extract all bracket contents
            bracket_matches = re.findall(r'\[(.*?)\]', text)
            if bracket_matches:
                # Assume single subject
                subject_id = 1
                subjects[subject_id] = "Subject 1"
                action_text = bracket_matches[0].strip()
                action_list = self._split_action_text(action_text)
                actions[subject_id] = action_list
        
        return subjects, actions
    
    def _split_action_text(self, action_text):
        """
        Intelligently split action text into separate actions
        
        Args:
            action_text: Comma-separated action text
            
        Returns:
            List of actions
        """
        if not action_text or not isinstance(action_text, str):
            return []
            
        # Simplified splitting logic
        # Try multiple delimiters
        if ',' in action_text:
            # Main delimiter is comma
            actions = [a.strip() for a in action_text.split(',') if a.strip()]
        else:
            # Try other delimiters
            actions = [a.strip() for a in re.split(r'[,;]|\s+and\s+', action_text) if a.strip()]
        
        # Clean and filter
        actions = [re.sub(r'[,.;!?]$', '', action).strip() for action in actions]
        actions = [a for a in actions if len(a.split()) >= 1]
        
        return actions
    
    def extract_actions_from_chronological_list(self, text):
        """
        Extract action sequence from chronological_motion_list
        
        Args:
            text: Chronological motion list text
            
        Returns:
            List of (action, subject_id) tuples
        """
        if not isinstance(text, str):
            return []
        
        actions_sequence = []
        
        # Main pattern: "Action (Subject X)"
        pattern = r"(.*?)\s*\(Subject (\d+)\)"
        matches = re.findall(pattern, text)
        
        if matches:
            for match in matches:
                action = match[0].strip()
                try:
                    subject_id = int(match[1])
                    # Only add when action is non-empty
                    if action:
                        actions_sequence.append((action, subject_id))
                except ValueError:
                    # Handle case where subject ID is not an integer
                    pass
        else:
            # Try comma-separated format: items separated by "Action (Subject X)"
            items = re.split(r',\s*', text)
            for item in items:
                item = item.strip()
                # Look for pattern: action(subject X)
                if '(' in item and ')' in item:
                    action_part = item[:item.rfind('(')].strip()
                    subject_part = item[item.rfind('(')+1:item.rfind(')')].strip()
                    
                    if subject_part.lower().startswith('subject'):
                        try:
                            subject_id = int(subject_part.split()[1])
                            if action_part:
                                actions_sequence.append((action_part, subject_id))
                        except (ValueError, IndexError):
                            # Handle exception cases
                            pass
        
        # Clean action text punctuation
        actions_sequence = [(re.sub(r'[,.;!?]$', '', action).strip(), subject_id) 
                          for action, subject_id in actions_sequence]
        
        return actions_sequence
        
    def extract_camera_motion_segments(self, text):
        """
        Extract camera motion segments from camera_motion field
        
        Args:
            text: Camera motion text
            
        Returns:
            List of camera motion segments
        """
        if not isinstance(text, str):
            return []
        
        # Preprocess text: remove excessive spaces, normalize delimiters
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Try multiple splitting methods
        # 1. Split by comma
        if ',' in text:
            segments = [s.strip() for s in text.split(',')]
        # 2. Split by period
        elif '.' in text:
            segments = [s.strip() for s in text.split('.') if s.strip()]
        # 3. Split by "Camera" keyword
        elif 'camera' in text.lower():
            segments = []
            parts = re.split(r'(camera\s+)', text.lower(), flags=re.IGNORECASE)
            current = ""
            for i, part in enumerate(parts):
                if re.match(r'camera\s+', part.lower()):
                    if current:
                        segments.append(current.strip())
                    current = part
                else:
                    current += part
            if current:
                segments.append(current.strip())
        # 4. Try other common separators
        else:
            segments = [s.strip() for s in re.split(r'[;]|\s+then\s+|\s+and\s+', text) if s.strip()]
        
        # If still no segments, use entire text as one segment
        if not segments:
            segments = [text]
        
        # Ensure each segment contains "camera" keyword
        camera_segments = []
        for segment in segments:
            if 'camera' in segment.lower():
                camera_segments.append(segment)
            else:
                # If no "camera" keyword, add prefix
                camera_segments.append(f"Camera {segment}")
        
        return camera_segments


# ===== Sequence Matching =====
class SequenceMatcher:
    """Match subject and action sequences"""
    
    def __init__(self, similarity_calculator, config_manager):
        """
        Initialize sequence matcher
        
        Args:
            similarity_calculator: Similarity calculator instance
            config_manager: Configuration manager instance
        """
        self.similarity_calculator = similarity_calculator
        self.config = config_manager
        self.rejected_matches = {}  # Store rejected matches for debugging
    
    def match_subjects(self, gt_subjects, pred_subjects, gt_actions, pred_actions, method='default'):
        """
        Improved subject matching algorithm considering action sequence similarity
        
        Args:
            gt_subjects: Ground truth subjects dictionary
            pred_subjects: Predicted subjects dictionary
            gt_actions: Ground truth actions dictionary
            pred_actions: Predicted actions dictionary
            method: Similarity method to use
            
        Returns:
            Subject mapping dictionary {pred_id: gt_id}
        """
        if not pred_subjects or not gt_subjects:
            return {}
        
        # Get subject matching configuration
        subject_config = self.config.get_matching_config("subject")
        similarity_threshold = subject_config["similarity_threshold"]
        high_confidence_threshold = subject_config["high_confidence_threshold"]
        description_weight = subject_config["description_weight"]
        action_weight = subject_config["action_weight"]
        allow_many_to_one = subject_config["allow_many_to_one"]
        max_match_ratio = subject_config["max_match_ratio"]
        
        gt_subject_ids = list(gt_subjects.keys())
        pred_subject_ids = list(pred_subjects.keys())
        
        # Calculate similarity matrix for each subject pair
        similarity_matrix = np.zeros((len(pred_subject_ids), len(gt_subject_ids)))
        
        for i, pred_id in enumerate(pred_subject_ids):
            for j, gt_id in enumerate(gt_subject_ids):
                # Calculate subject description similarity
                pred_desc = pred_subjects[pred_id].lower() if pred_id in pred_subjects else ""
                gt_desc = gt_subjects[gt_id].lower() if gt_id in gt_subjects else ""
                
                # Calculate base similarity
                desc_similarity = self.similarity_calculator.compute_text_similarity(pred_desc, gt_desc, method)
                
                # Enhance: Improve reference similarity matching
                desc_similarity = self._enhance_reference_similarity(pred_desc, gt_desc, desc_similarity)
                
                # Calculate action sequence overall similarity
                action_similarity = 0
                if pred_id in pred_actions and gt_id in gt_actions:
                    # If subject has no actions, don't consider action similarity
                    if not pred_actions[pred_id] or not gt_actions[gt_id]:
                        action_similarity = 0
                    else:
                        # Calculate action sequence similarity
                        # Compare action texts
                        pred_action_text = " ".join(pred_actions[pred_id])
                        gt_action_text = " ".join(gt_actions[gt_id])
                        text_similarity = self.similarity_calculator.compute_text_similarity(
                            pred_action_text, gt_action_text, method
                        )
                        
                        # Calculate action sequence item-by-item similarity
                        if pred_actions[pred_id] and gt_actions[gt_id]:
                            # Build action similarity matrix
                            action_sim_matrix = self.similarity_calculator.compute_action_similarity_matrix(
                                pred_actions[pred_id], gt_actions[gt_id], method,
                                position_aware=self.config.get("action_matching", "position_aware")
                            )
                            
                            # Find best matches
                            if np.max(action_sim_matrix) > 0:
                                # Calculate average similarity
                                row_max = np.max(action_sim_matrix, axis=1)
                                row_mean = np.mean(row_max)
                                
                                col_max = np.max(action_sim_matrix, axis=0)
                                col_mean = np.mean(col_max)
                                
                                match_similarity = (row_mean + col_mean) / 2
                                action_similarity = 0.2 * text_similarity + 0.8 * match_similarity
                            else:
                                action_similarity = text_similarity
                
                # Final similarity: when subject has actions, use actions as main criterion, otherwise use description
                if pred_id in pred_actions and gt_id in gt_actions and pred_actions[pred_id] and gt_actions[gt_id]:
                    # Use configured weights to calculate weighted similarity
                    similarity_matrix[i, j] = (description_weight * desc_similarity + 
                                            action_weight * action_similarity)
                else:
                    similarity_matrix[i, j] = desc_similarity
        
        # Use Hungarian algorithm to find optimal matching
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
        
        # Preprocess all possible matches, sort by similarity
        potential_matches = []
        for i, j in zip(row_ind, col_ind):
            if i < len(pred_subject_ids) and j < len(gt_subject_ids):
                pred_id = pred_subject_ids[i]
                gt_id = gt_subject_ids[j]
                sim = similarity_matrix[i, j]
                potential_matches.append((pred_id, gt_id, sim))
        
        # Sort by similarity in descending order
        potential_matches.sort(key=lambda x: x[2], reverse=True)
        
        # Confirm matches one by one, ensuring constraints are met
        subject_mapping = {}
        matched_pred = set()
        matched_gt = set()
        
        # First match high similarity subject pairs
        for pred_id, gt_id, sim in potential_matches:
            if sim >= high_confidence_threshold and pred_id not in matched_pred and gt_id not in matched_gt:
                subject_mapping[pred_id] = gt_id
                matched_pred.add(pred_id)
                matched_gt.add(gt_id)
        
        # Then greedily match remaining subjects
        for pred_id, gt_id, sim in potential_matches:
            if pred_id not in matched_pred and gt_id not in matched_gt:
                if sim > similarity_threshold:
                    subject_mapping[pred_id] = gt_id
                    matched_pred.add(pred_id)
                    matched_gt.add(gt_id)
        
        # Enhance: Ensure at least one subject is matched
        if not subject_mapping and potential_matches:
            # If no matches reached threshold, select most similar pair
            best_match = potential_matches[0]  # Already sorted by similarity
            pred_id, gt_id, _ = best_match
            subject_mapping[pred_id] = gt_id
            matched_pred.add(pred_id)
            matched_gt.add(gt_id)
        
        # If many-to-one matching allowed and unmatched subjects remain
        if allow_many_to_one:
            remaining_potential_matches = [
                (pred_id, gt_id, sim) for pred_id, gt_id, sim in potential_matches 
                if pred_id not in matched_pred and sim > similarity_threshold
            ]
            
            # For each ground truth subject, limit maximum number of matches
            gt_match_counts = {}
            for gt_id in gt_subject_ids:
                # Adjust max matches ratio based on action count
                if gt_id in gt_actions:
                    action_count = len(gt_actions[gt_id])
                    max_matches = max(1, int(action_count * max_match_ratio))
                else:
                    max_matches = 1
                
                # Initialize counter
                gt_match_counts[gt_id] = 0
                
                # Update counts for already matched
                for matched_gt in [subject_mapping[p] for p in matched_pred]:
                    if matched_gt == gt_id:
                        gt_match_counts[gt_id] += 1
            
            for pred_id, gt_id, sim in remaining_potential_matches:
                if pred_id not in matched_pred and gt_match_counts[gt_id] < max(1, int(len(gt_actions.get(gt_id, [])) * max_match_ratio)):
                    subject_mapping[pred_id] = gt_id
                    matched_pred.add(pred_id)
                    gt_match_counts[gt_id] += 1
        
        return subject_mapping

    # Reference similarity enhancement method
    def _enhance_reference_similarity(self, text1, text2, base_similarity):
        """
        Enhance similarity score between reference terms
        
        Args:
            text1: First text
            text2: Second text
            base_similarity: Base similarity score
            
        Returns:
            Enhanced similarity score
        """
        # Common reference groups
        reference_groups = [
            # Exact match group (highest boost)
            {'boost': 0.3, 'terms': [
                {'girl', 'girls'}, {'woman', 'women'}, {'man', 'men'}, 
                {'boy', 'boys'}, {'person', 'people'}, {'child', 'children'},
                {'adult', 'adults'}, {'subject', 'subjects'}
            ]},
            # High similarity group (substantial boost)
            {'boost': 0.15, 'terms': [
                {'girl', 'woman'}, {'girl', 'female'}, {'woman', 'female'}, {'woman', 'lady'},
                {'boy', 'man'}, {'boy', 'male'}, {'man', 'male'}, 
                {'child', 'kid'}, {'child', 'girl'}, {'child', 'boy'},
                {'person', 'human'}, {'person', 'individual'}, {'person', 'subject'}
            ]},
            # Medium similarity group (moderate boost)
            {'boost': 0.07, 'terms': [
                {'person', 'man'}, {'person', 'woman'}, {'person', 'child'},
                {'adult', 'man'}, {'adult', 'woman'}, {'teenager', 'boy'}, {'teenager', 'girl'}
            ]},
            # No similarity group (no boost)
            {'boost': 0, 'terms': [
                {'man', 'woman'}, {'boy', 'girl'}, {'person', 'animal'},
                {'human', 'dog'}, {'human', 'cat'}, {'man', 'cat'}, {'woman', 'dog'}
            ]}
        ]
        
        # Extract keywords
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Check for matching reference terms
        max_boost = 0
        for group in reference_groups:
            boost = group['boost']
            for term_set in group['terms']:
                # Check if both texts contain words from the same set
                first_found = any(word in words1 for word in term_set)
                second_found = any(word in words2 for word in term_set)
                
                if first_found and second_found:
                    # Extra boost for exact same terms
                    same_terms = words1.intersection(words2).intersection(term_set)
                    if same_terms:
                        max_boost = max(max_boost, boost * 1.5)  # Extra boost for exact match
                    else:
                        max_boost = max(max_boost, boost)
        
        # Apply boost, not exceeding 1.0
        return min(1.0, base_similarity + max_boost)
    
    def improved_action_matching(self, pred_actions, gt_actions, method='default', 
                                similarity_threshold=None, allow_many_to_one=None,
                                position_aware=None, max_window_size=None,
                                clean_low_quality_matches=None):
        """
        Enhanced action matching algorithm with support for many-to-one matching and position awareness
        
        Args:
            pred_actions: Predicted actions list
            gt_actions: Ground truth actions list
            method: Similarity method to use
            similarity_threshold: Similarity threshold below which matches are ignored
            allow_many_to_one: Whether to allow multiple predicted actions to match a single ground truth action
            position_aware: Whether to consider position information
            max_window_size: Maximum window size for action combinations
            clean_low_quality_matches: Whether to remove low quality matches
        
        Returns:
            (matching_score, precision, recall, action_mapping, match_scores) tuple
        """
        if not pred_actions or not gt_actions:
            return 0, 0, 0, {}, []
        
        # Get default values from configuration
        action_config = self.config.get_matching_config("action")
        if similarity_threshold is None:
            similarity_threshold = action_config["similarity_threshold"]
        if allow_many_to_one is None:
            allow_many_to_one = action_config["allow_many_to_one"]
        if position_aware is None:
            position_aware = action_config["position_aware"]
        if max_window_size is None:
            max_window_size = action_config["max_window_size"]
        if clean_low_quality_matches is None:
            clean_low_quality_matches = action_config["clean_low_quality_matches"]
        
        # Extract configuration
        very_low_threshold = action_config["very_low_match_threshold"]
        max_match_distance = action_config["max_match_distance"]
        
        # Clear current rejected matches
        self.rejected_matches = {}
        
        # Calculate position-aware similarity matrix
        similarity_matrix = self.similarity_calculator.compute_action_similarity_matrix(
            pred_actions, gt_actions, method, position_aware=position_aware
        )
        
        # Adaptive threshold setting
        if action_config["adaptive_threshold"]:
            # Adjust threshold based on similarity distribution
            matrix_mean = np.mean(similarity_matrix)
            matrix_std = np.std(similarity_matrix)
            # Avoid too high or too low threshold
            adaptive_threshold = max(action_config["min_adaptive_threshold"], 
                                   min(0.7, matrix_mean - 0.5 * matrix_std))
            # Only apply adaptive threshold if it's lower than configured threshold
            similarity_threshold = min(similarity_threshold, adaptive_threshold)
        
        # Basic one-to-one matching and many-to-one matching
        if allow_many_to_one:
            # Many-to-one matching: find best ground truth action for each predicted action
            action_mapping = {}
            match_scores = []
            
            # For each predicted action, find most similar ground truth action
            for i in range(len(pred_actions)):
                best_similarity = np.max(similarity_matrix[i])
                best_j = np.argmax(similarity_matrix[i])
                
                # Check if similarity is high enough
                if best_similarity >= similarity_threshold:
                    # Check position constraint
                    if abs(i - best_j) <= max_match_distance or not position_aware:
                        action_mapping[i] = best_j
                        match_scores.append(best_similarity)
                    else:
                        # Record rejected match
                        self.rejected_matches[i] = {"gt_idx": best_j, "score": best_similarity, "reason": "position_constraint"}
                else:
                    # Record rejected match
                    self.rejected_matches[i] = {"gt_idx": best_j, "score": best_similarity, "reason": "low_similarity"}
        else:
            # One-to-one matching: use Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
            
            action_mapping = {}
            match_scores = []
            
            for i, j in zip(row_ind, col_ind):
                similarity = similarity_matrix[i, j]
                # Check if similarity is high enough
                if similarity >= similarity_threshold:
                    # Check position constraint
                    if abs(i - j) <= max_match_distance or not position_aware:
                        action_mapping[i] = j
                        match_scores.append(similarity)
                    else:
                        # Record rejected match
                        self.rejected_matches[i] = {"gt_idx": j, "score": similarity, "reason": "position_constraint"}
                else:
                    # Record rejected match
                    self.rejected_matches[i] = {"gt_idx": j, "score": similarity, "reason": "low_similarity"}
        
        # Try action combination matching - multiple consecutive predicted actions matching single ground truth action
        if max_window_size > 1:
            # Try action combination matching
            for window_size in range(2, min(max_window_size + 1, len(pred_actions) + 1)):
                for i in range(len(pred_actions) - window_size + 1):
                    # Check if all actions in current window are already matched
                    all_matched = all(i+k in action_mapping for k in range(window_size))
                    if all_matched:
                        continue
                    
                    # Combine consecutive actions
                    combined_pred = " ".join(pred_actions[i:i+window_size])
                    
                    # Calculate similarity with each ground truth action
                    for j, gt_action in enumerate(gt_actions):
                        # Calculate similarity
                        combined_similarity = self.similarity_calculator.compute_action_similarity(
                            combined_pred, gt_action, method
                        )
                        
                        # If combined match is better than individual matches
                        better_threshold = similarity_threshold + 0.1  # Higher threshold for combinations
                        if combined_similarity > better_threshold:
                            # Check position constraint
                            if abs(i - j) <= max_match_distance or not position_aware:
                                # Check if worth replacing existing matches
                                current_match_quality = sum(similarity_matrix[i+k, action_mapping.get(i+k, j)] 
                                                          for k in range(window_size) if i+k in action_mapping)
                                
                                if combined_similarity > current_match_quality / max(1, sum(1 for k in range(window_size) if i+k in action_mapping)):
                                    # Remove existing conflicting matches
                                    for k in range(window_size):
                                        if i+k in action_mapping:
                                            # Find and remove corresponding score
                                            idx = list(action_mapping.keys()).index(i+k)
                                            if idx < len(match_scores):
                                                match_scores.pop(idx)
                                            action_mapping.pop(i+k)
                                    
                                    # Add new combined match (only map first action to target)
                                    action_mapping[i] = j
                                    match_scores.append(combined_similarity)
        
        # Clean low quality matches
        if clean_low_quality_matches and match_scores:
            # Calculate match scores mean
            mean_score = np.mean(match_scores)
            
            # Identify exceptionally low quality matches
            low_quality_keys = []
            for i, j in list(action_mapping.items()):
                score_idx = list(action_mapping.keys()).index(i)
                if score_idx < len(match_scores):
                    score = match_scores[score_idx]
                    # Score below threshold and significantly below average, consider removing
                    if score < very_low_threshold and score < mean_score - 0.2:
                        low_quality_keys.append(i)
                        self.rejected_matches[i] = {"gt_idx": j, "score": score, "reason": "low_quality_match"}
            
            # Remove low quality matches
            for i in low_quality_keys:
                idx = list(action_mapping.keys()).index(i)
                if idx < len(match_scores):
                    match_scores.pop(idx)
                action_mapping.pop(i)
        
        # Calculate metrics
        if match_scores:
            # Add sequence length difference penalty
            length_penalty = action_config["length_penalty"]
            length_ratio = min(len(pred_actions), len(gt_actions)) / max(len(pred_actions), len(gt_actions))
            length_factor = 1.0 - (1.0 - length_ratio) * length_penalty
            
            # Precision: matched predicted actions / total predicted actions
            precision = len(match_scores) / len(pred_actions) if pred_actions else 0
            
            # Recall: matched ground truth actions / total ground truth actions
            matched_gt_indices = set(action_mapping.values())
            recall = len(matched_gt_indices) / len(gt_actions) if gt_actions else 0
            
            # Average match quality
            avg_similarity = np.mean(match_scores)
            
            # Quality-weighted precision and recall
            quality_precision = precision * avg_similarity * length_factor
            quality_recall = recall * avg_similarity * length_factor
            
            # F1 score
            if quality_precision + quality_recall > 0:
                f1 = 2 * quality_precision * quality_recall / (quality_precision + quality_recall)
            else:
                f1 = 0
        else:
            precision = 0
            recall = 0
            quality_precision = 0
            quality_recall = 0
            f1 = 0
        
        return f1, quality_precision, quality_recall, action_mapping, match_scores
    
    def calculate_kendall_tau(self, list1, list2):
        """
        Calculate Kendall's Tau correlation coefficient to evaluate order consistency
        
        Args:
            list1: First list of values
            list2: Second list of values
            
        Returns:
            Kendall's Tau correlation coefficient (-1 to 1)
        """
        n = len(list1)
        if n <= 1:
            return 1.0  # Only one element, order is perfectly correct
        
        concordant = 0
        discordant = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if (list1[i] < list1[j] and list2[i] < list2[j]) or (list1[i] > list1[j] and list2[i] > list2[j]):
                    concordant += 1
                else:
                    discordant += 1
        
        total_pairs = n * (n - 1) // 2
        tau = (concordant - discordant) / total_pairs if total_pairs > 0 else 1.0
        
        return tau
    
    def evaluate_action_order(self, action_mapping, pred_actions, gt_actions):
        """
        Evaluate action order correctness with improved algorithm for many-to-one cases and short sequences
        
        Args:
            action_mapping: Action mapping dictionary {pred_idx: gt_idx}
            pred_actions: Predicted actions list
            gt_actions: Ground truth actions list
            
        Returns:
            Order score (0-1)
        """
        # Get order evaluation configuration
        order_config = self.config.get_order_config()
        max_order_penalty = order_config["max_order_penalty"]
        position_weight = order_config["position_weight"]
        consistency_weight = order_config["consistency_weight"]
        use_kendall_tau = order_config["use_kendall_tau"]
        normalize_by_sequence_length = order_config["normalize_by_sequence_length"]
        long_sequence_threshold = order_config["long_sequence_threshold"]
        min_sequence_for_order = order_config["min_sequence_for_order"]
        short_sequence_penalty = order_config["short_sequence_penalty"]
        
        # For sequences too short for order evaluation, adjust score based on length
        if not action_mapping or len(action_mapping) < min_sequence_for_order:
            # Calculate predicted and ground truth sequence lengths
            pred_len = len(pred_actions)
            gt_len = len(gt_actions)
            
            # If both predicted and ground truth sequences are short, order isn't important
            if pred_len < min_sequence_for_order and gt_len < min_sequence_for_order:
                return 1.0
            
            # If predicted sequence is short but ground truth is longer, reduce order score
            if pred_len < min_sequence_for_order and gt_len >= min_sequence_for_order:
                return 1.0 - short_sequence_penalty
            
            # If predicted sequence is longer but few matches, also reduce score
            if pred_len >= min_sequence_for_order:
                return 1.0 - short_sequence_penalty * 1.5
        
        # Extract matched index pairs
        matched_pairs = sorted(action_mapping.items())
        
        # Handle many-to-one mapping special case
        unique_gt_indices = {}
        for pred_idx, gt_idx in matched_pairs:
            if gt_idx not in unique_gt_indices:
                unique_gt_indices[gt_idx] = []
            unique_gt_indices[gt_idx].append(pred_idx)
        
        # If many-to-one mappings exist, special handling needed
        if any(len(pred_indices) > 1 for pred_indices in unique_gt_indices.values()):
            # For each multiply-matched ground truth action, keep only the best match
            refined_mapping = {}
            
            for gt_idx, pred_indices in unique_gt_indices.items():
                if len(pred_indices) == 1:
                    # One-to-one mapping preserved
                    refined_mapping[pred_indices[0]] = gt_idx
                else:
                    # Many-to-one mapping, select first predicted action
                    best_pred_idx = min(pred_indices)
                    refined_mapping[best_pred_idx] = gt_idx
            
            # Use refined mapping for recalculation
            matched_pairs = sorted(refined_mapping.items())
        
        if not matched_pairs:
            return 0.0  # No valid matches after refinement
        
        # Extract matched index sequences
        matched_pred_indices = [pair[0] for pair in matched_pairs]
        matched_gt_indices = [pair[1] for pair in matched_pairs]
        
        if use_kendall_tau:
            # Use Kendall's Tau to evaluate order consistency
            tau = self.calculate_kendall_tau(matched_pred_indices, matched_gt_indices)
            
            # Map tau from [-1, 1] to [0, 1]
            order_score = (tau + 1) / 2
        else:
            # Alternative: Calculate position differences and mismatch proportion
            normalized_pred_positions = [i/(len(matched_pred_indices)-1) if len(matched_pred_indices)>1 else 0.5 
                                       for i in range(len(matched_pred_indices))]
            
            # Relative positions of sorted ground truth indices
            sorted_gt_idx = sorted(range(len(matched_gt_indices)), key=lambda i: matched_gt_indices[i])
            normalized_gt_positions = [i/(len(sorted_gt_idx)-1) if len(sorted_gt_idx)>1 else 0.5 
                                     for i in range(len(sorted_gt_idx))]
            
            # Calculate position differences
            position_diffs = [abs(normalized_pred_positions[i] - normalized_gt_positions[sorted_gt_idx.index(i)]) 
                            for i in range(len(matched_pred_indices))]
            avg_position_diff = np.mean(position_diffs)
            
            # Calculate order consistency
            correct_order_count = 0
            total_pairs = 0
            
            for i in range(len(matched_pred_indices)):
                for j in range(i+1, len(matched_pred_indices)):
                    pred_order_correct = (matched_pred_indices[i] < matched_pred_indices[j] and 
                                        matched_gt_indices[i] < matched_gt_indices[j]) or \
                                        (matched_pred_indices[i] > matched_pred_indices[j] and 
                                        matched_gt_indices[i] > matched_gt_indices[j])
                    
                    if pred_order_correct:
                        correct_order_count += 1
                    
                    total_pairs += 1
            
            order_consistency = correct_order_count / total_pairs if total_pairs > 0 else 1.0
            
            # Calculate weighted score
            order_score = (1 - position_weight) * order_consistency + position_weight * (1 - avg_position_diff)
        
        # For longer sequences, reduce penalty appropriately
        if normalize_by_sequence_length and len(matched_pairs) > long_sequence_threshold:
            # Longer sequences should have smaller order error penalty
            sequence_factor = long_sequence_threshold / len(matched_pairs)
            max_order_penalty = max_order_penalty * sequence_factor + max_order_penalty * (1 - sequence_factor)
        
        # Apply maximum order penalty
        order_score = 1.0 - (1.0 - order_score) * max_order_penalty
        
        return order_score
    
    def improved_chronological_order_evaluation(self, pred_chrono, gt_chrono, method='default', similarity_threshold=None):
        """
        Enhanced chronological order evaluation with position awareness and action combination understanding
        
        Args:
            pred_chrono: Predicted chronological motion list
            gt_chrono: Ground truth chronological motion list
            method: Similarity method to use
            similarity_threshold: Similarity threshold
            
        Returns:
            (F1 score, precision, recall, order score, match mapping) tuple
        """
        # Extract action sequences
        extractor = InformationExtractor(self.config)
        pred_sequence = extractor.extract_actions_from_chronological_list(pred_chrono)
        gt_sequence = extractor.extract_actions_from_chronological_list(gt_chrono)
        
        if not pred_sequence or not gt_sequence:
            return 0, 0, 0, 0, {}
        
        # Use configured default threshold if none provided
        if similarity_threshold is None:
            similarity_threshold = self.config.get("action_matching", "similarity_threshold")
        
        # Extract action texts and subject IDs
        pred_actions = [action for action, _ in pred_sequence]
        gt_actions = [action for action, _ in gt_sequence]
        pred_subjects = [subject for _, subject in pred_sequence]
        gt_subjects = [subject for _, subject in gt_sequence]
        
        # Use enhanced action matching algorithm
        matching_score, match_precision, match_recall, action_mapping, match_scores = self.improved_action_matching(
            pred_actions, gt_actions, method, 
            similarity_threshold=similarity_threshold,
            allow_many_to_one=self.config.get("action_matching", "allow_many_to_one"),
            position_aware=self.config.get("action_matching", "position_aware"),
            clean_low_quality_matches=self.config.get("action_matching", "clean_low_quality_matches")
        )
        
        # Check subject match status
        subject_matches = []
        for i, j in action_mapping.items():
            if i < len(pred_subjects) and j < len(gt_subjects):
                subject_match = int(pred_subjects[i] == gt_subjects[j])
                subject_matches.append(subject_match)
        
        # Calculate subject match rate
        subject_match_rate = np.mean(subject_matches) if subject_matches else 0
        
        # Evaluate order correctness
        order_score = self.evaluate_action_order(action_mapping, pred_actions, gt_actions)
        
        # Integrate action match score and subject match rate (slight adjustment for subject match weight)
        final_precision = match_precision * (0.9 + 0.1 * subject_match_rate)
        final_recall = match_recall * (0.9 + 0.1 * subject_match_rate)
        
        # Calculate F1
        if final_precision + final_recall > 0:
            final_f1 = 2 * final_precision * final_recall / (final_precision + final_recall)
        else:
            final_f1 = 0
        
        return final_f1, final_precision, final_recall, order_score, action_mapping
    
    def evaluate_camera_motion(self, pred_camera_motion, gt_camera_motion, method='default'):
        """
        Evaluate camera motion description quality
        
        Args:
            pred_camera_motion: Predicted camera motion description
            gt_camera_motion: Ground truth camera motion description
            method: Similarity method to use
            
        Returns:
            (F1 score, precision, recall, order score, match mapping) tuple
        """
        # Extract camera motion segments
        extractor = InformationExtractor(self.config)
        pred_segments = extractor.extract_camera_motion_segments(pred_camera_motion)
        gt_segments = extractor.extract_camera_motion_segments(gt_camera_motion)
        
        if not pred_segments or not gt_segments:
            return 0, 0, 0, 0, {}
        
        # Use enhanced action matching algorithm to evaluate camera motion
        matching_score, match_precision, match_recall, segment_mapping, match_scores = self.improved_action_matching(
            pred_segments, gt_segments, method,
            similarity_threshold=self.config.get("action_matching", "similarity_threshold") * 0.8,  # Slightly lower threshold
            allow_many_to_one=self.config.get("action_matching", "allow_many_to_one"),
            position_aware=self.config.get("action_matching", "position_aware"),
            clean_low_quality_matches=self.config.get("action_matching", "clean_low_quality_matches")
        )
        
        # Evaluate camera motion order
        order_score = self.evaluate_action_order(segment_mapping, pred_segments, gt_segments)
        
        # Calculate final F1
        if match_precision + match_recall > 0:
            final_f1 = 2 * match_precision * match_recall / (match_precision + match_recall)
        else:
            final_f1 = 0
            
        return final_f1, match_precision, match_recall, order_score, segment_mapping


# ===== Evaluator =====
class VideoActionEvaluator:
    """Evaluate model's video action understanding and description capabilities"""
    
    def __init__(self, config_dict=None):
        """
        Initialize evaluator
        
        Args:
            config_dict: Configuration dictionary to override defaults
        """
        # Initialize configuration
        self.config_manager = ConfigManager(config_dict)
        
        # Initialize components
        self.model_manager = ModelManager(self.config_manager)
        self.model_manager.check_dependencies()
        self.text_processor = TextProcessor(self.model_manager, self.config_manager)
        self.similarity_calculator = SimilarityCalculator(self.model_manager, self.text_processor, self.config_manager)
        self.information_extractor = InformationExtractor(self.config_manager)
        self.sequence_matcher = SequenceMatcher(self.similarity_calculator, self.config_manager)
        
        # Ensure NLTK resources are available
        self.model_manager.load_nltk_resources()
        
        # Set evaluation method
        self.method = self.config_manager.get("similarity", "method")
        
        # Set weights
        self.weights = self.config_manager.get_scoring_weights()
    
    def compute_overall_score(self, gt_annotation, pred_extraction):
        """
        Calculate comprehensive score integrating multiple evaluation dimensions
        
        Args:
            gt_annotation: Ground truth annotation data
            pred_extraction: Model prediction extraction result
        
        Returns:
            scores: Dictionary with scores for each dimension
            detailed_results: Dictionary with detailed evaluation results
        """
        scores = {}
        detailed_results = {}
        
        try:
            # 1. Evaluate camera motion
            gt_camera_motion = gt_annotation.get("camera_motion", "")
            # Check if ground truth annotation includes camera motion
            if not gt_camera_motion or not gt_camera_motion.strip():
                # If no camera motion in ground truth, give full score
                camera_f1, camera_precision, camera_recall, camera_order = 1.0, 1.0, 1.0, 1.0
                camera_mapping = {}
                
                # Record that full score was given due to missing GT camera motion
                detailed_results["camera_motion"] = {
                    "pred_segments": self.information_extractor.extract_camera_motion_segments(
                        pred_extraction.get("camera_motion", "")
                    ),
                    "gt_segments": [],
                    "segment_mapping": {},
                    "f1_score": 1.0,
                    "precision_score": 1.0,
                    "recall_score": 1.0,
                    "order_score": 1.0,
                    "no_gt_camera_motion": True  # Flag indicating no GT camera motion
                }
            else:
                # Normal camera motion evaluation
                camera_f1, camera_precision, camera_recall, camera_order, camera_mapping = self.sequence_matcher.evaluate_camera_motion(
                    pred_extraction.get("camera_motion", ""),
                    gt_camera_motion,
                    self.method
                )
                
                # Record camera motion evaluation details
                detailed_results["camera_motion"] = {
                    "pred_segments": self.information_extractor.extract_camera_motion_segments(
                        pred_extraction.get("camera_motion", "")
                    ),
                    "gt_segments": self.information_extractor.extract_camera_motion_segments(
                        gt_camera_motion
                    ),
                    "segment_mapping": camera_mapping,
                    "f1_score": camera_f1,
                    "precision_score": camera_precision,
                    "recall_score": camera_recall,
                    "order_score": camera_order,
                    "no_gt_camera_motion": False
                }

            scores["camera_motion_score"] = camera_f1  # Overall score
            scores["camera_precision_score"] = camera_precision
            scores["camera_recall_score"] = camera_recall
            scores["camera_order_score"] = camera_order

            
            # 2. Extract subjects and actions
            gt_subjects, gt_actions = self.information_extractor.extract_subjects_and_actions_from_motion_list(
                gt_annotation.get("motion_list", "")
            )
            pred_subjects, pred_actions = self.information_extractor.extract_subjects_and_actions_from_motion_list(
                pred_extraction.get("motion_list", "")
            )
            
            detailed_results["extracted_info"] = {
                "gt_subjects": gt_subjects,
                "gt_actions": gt_actions,
                "pred_subjects": pred_subjects,
                "pred_actions": pred_actions
            }
            
            # 3. Match subjects
            subject_mapping = self.sequence_matcher.match_subjects(
                gt_subjects, pred_subjects, gt_actions, pred_actions, self.method
            )
            detailed_results["subject_mapping"] = subject_mapping
            
            # 4. Evaluate each subject's action recognition and order
            subject_scores = []
            subject_precision_scores = []
            subject_recall_scores = []
            subject_order_scores = []
            subject_weights = []
            subject_details = {}
            
            for pred_id, gt_id in subject_mapping.items():
                if pred_id in pred_actions and gt_id in gt_actions:
                    # Evaluate action matching
                    action_f1, action_precision, action_recall, action_mapping, match_scores = self.sequence_matcher.improved_action_matching(
                        pred_actions[pred_id], gt_actions[gt_id], self.method,
                        clean_low_quality_matches=self.config_manager.get("action_matching", "clean_low_quality_matches")
                    )
                    
                    subject_scores.append(action_f1)
                    subject_precision_scores.append(action_precision)
                    subject_recall_scores.append(action_recall)
                    
                    # Evaluate action order
                    order_score = self.sequence_matcher.evaluate_action_order(
                        action_mapping, pred_actions[pred_id], gt_actions[gt_id]
                    )
                    subject_order_scores.append(order_score)
                    
                    # Weight by ground truth action count
                    weight = len(gt_actions[gt_id])
                    subject_weights.append(weight)
                    
                    # Record detailed results
                    subject_details[f"Subject {pred_id} → {gt_id}"] = {
                        "pred_actions": pred_actions[pred_id],
                        "gt_actions": gt_actions[gt_id],
                        "action_mapping": action_mapping,
                        "match_scores": match_scores,
                        "action_f1": action_f1,
                        "action_precision": action_precision, 
                        "action_recall": action_recall,
                        "order_score": order_score,
                        "weight": weight,
                        "rejected_matches": self.sequence_matcher.rejected_matches
                    }
            
            detailed_results["subject_evaluations"] = subject_details
            
            # Calculate weighted average scores
            if subject_weights:
                weighted_score = sum(s * w for s, w in zip(subject_scores, subject_weights)) / sum(subject_weights)
                weighted_precision = sum(p * w for p, w in zip(subject_precision_scores, subject_weights)) / sum(subject_weights)
                weighted_recall = sum(r * w for r, w in zip(subject_recall_scores, subject_weights)) / sum(subject_weights)
                weighted_order_score = sum(s * w for s, w in zip(subject_order_scores, subject_weights)) / sum(subject_weights)
            else:
                weighted_score = 0
                weighted_precision = 0
                weighted_recall = 0
                weighted_order_score = 0
            
            scores["subject_action_score"] = weighted_score  # Legacy compatibility
            scores["subject_precision_score"] = weighted_precision
            scores["subject_recall_score"] = weighted_recall
            scores["subject_order_score"] = weighted_order_score
            
            # 5. Evaluate chronological order accuracy
            chrono_f1, chrono_precision, chrono_recall, chrono_order_score, chrono_mapping = self.sequence_matcher.improved_chronological_order_evaluation(
                pred_extraction.get("chronological_motion_list", ""),
                gt_annotation.get("chronological_motion_list", ""),
                self.method
            )
            
            scores["chrono_match_score"] = chrono_f1  # Legacy compatibility
            scores["chrono_precision_score"] = chrono_precision
            scores["chrono_recall_score"] = chrono_recall
            scores["chrono_order_score"] = chrono_order_score
            
            # Record chronological order detailed results
            detailed_results["chronological_evaluation"] = {
                "pred_sequence": self.information_extractor.extract_actions_from_chronological_list(
                    pred_extraction.get("chronological_motion_list", "")
                ),
                "gt_sequence": self.information_extractor.extract_actions_from_chronological_list(
                    gt_annotation.get("chronological_motion_list", "")
                ),
                "chrono_mapping": chrono_mapping,
                "f1_score": chrono_f1,
                "precision_score": chrono_precision,
                "recall_score": chrono_recall,
                "order_score": chrono_order_score,
                "rejected_matches": self.sequence_matcher.rejected_matches
            }
            
            # 6. Calculate final comprehensive score (based on precision/recall metrics)
            final_score = (
                self.weights["camera_motion"] * camera_f1 +
                self.weights["subject_precision"] * weighted_precision +
                self.weights["subject_recall"] * weighted_recall +
                self.weights["subject_order"] * weighted_order_score +
                self.weights["chrono_precision"] * chrono_precision +
                self.weights["chrono_recall"] * chrono_recall +
                self.weights["chrono_order"] * chrono_order_score
            )
            
            scores["final_score"] = final_score
        
        except Exception as e:
            if self.config_manager.get("display", "debug_level") > 0:
                print(f"Error computing score: {e}")
                import traceback
                traceback.print_exc()
            
            # Return zero scores on error
            scores = {
                "camera_motion_score": 0,
                "camera_precision_score": 0,
                "camera_recall_score": 0,
                "camera_order_score": 0,
                "subject_action_score": 0,
                "subject_precision_score": 0, 
                "subject_recall_score": 0,
                "subject_order_score": 0,
                "chrono_match_score": 0,
                "chrono_precision_score": 0,
                "chrono_recall_score": 0,
                "chrono_order_score": 0,
                "final_score": 0
            }
            detailed_results = {"error": str(e)}
        
        return scores, detailed_results
    
    def evaluate_videos(self, gt_annotations, pred_extractions):
        """
        Evaluate scores for all videos
        
        Args:
            gt_annotations: Ground truth annotations dictionary
            pred_extractions: Predicted extractions dictionary
            
        Returns:
            video_scores: Dictionary with scores for each video
            overall_scores: Dictionary with average scores across all videos
            video_details: Dictionary with detailed evaluation results for each video
        """
        video_scores = {}
        video_details = {}
        overall_scores = {
            "camera_motion_score": [],
            "camera_precision_score": [],
            "camera_recall_score": [],
            "camera_order_score": [],
            "subject_action_score": [], 
            "subject_precision_score": [],
            "subject_recall_score": [],
            "subject_order_score": [],
            "chrono_match_score": [],
            "chrono_precision_score": [],
            "chrono_recall_score": [],
            "chrono_order_score": [],
            "final_score": []
        }
        
        # Show progress bar if enabled
        display_config = self.config_manager.get_display_config()
        use_progress_bar = display_config["progress_bar"]
        
        video_ids = list(gt_annotations.keys())
        
        if use_progress_bar:
            iter_obj = tqdm(video_ids, desc="Evaluating videos")
        else:
            iter_obj = video_ids
        
        for video_id in iter_obj:
            if video_id in pred_extractions:
                try:
                    scores, details = self.compute_overall_score(
                        gt_annotations[video_id], 
                        pred_extractions[video_id]
                    )
                    video_scores[video_id] = scores
                    video_details[video_id] = details
                    
                    # Accumulate scores
                    for key in overall_scores:
                        if key in scores:
                            overall_scores[key].append(scores[key])
                except Exception as e:
                    if self.config_manager.get("display", "debug_level") > 0:
                        print(f"Error evaluating video {video_id}: {e}")
        
        # Calculate average scores
        avg_scores = {key: np.mean(values) if values else 0 for key, values in overall_scores.items()}
        
        return video_scores, avg_scores, video_details
    
    def display_action_sequence_comparison(self, pred_actions, gt_actions, action_mapping, rejected_matches=None):
        """
        Display comparison of action sequences including rejected matches
        
        Args:
            pred_actions: Predicted actions list
            gt_actions: Ground truth actions list
            action_mapping: Action mapping dictionary {pred_idx: gt_idx}
            rejected_matches: Dictionary of rejected matches and reasons
        """
        if not pred_actions and not gt_actions:
            display(Markdown("**Both sequences are empty.**"))
            return
        
        # Get display configuration
        display_config = self.config_manager.get_display_config()
        color_coding = display_config["color_coding"]
        precision = display_config["precision"]
        show_rejected = display_config["show_rejected_matches"]
        
        display(Markdown("### Action Sequence Comparison"))
        
        # Build predicted sequence table
        pred_data = []
        for i, action in enumerate(pred_actions):
            matched = i in action_mapping
            if matched:
                gt_idx = action_mapping[i]
                gt_action = gt_actions[gt_idx] if gt_idx < len(gt_actions) else "Out of range"
                # Calculate match quality
                similarity = self.similarity_calculator.compute_action_similarity(action, gt_action, self.method)
                similarity_str = f"{similarity:.{precision}f}"
                
                if color_coding:
                    # Use color coding for match quality
                    if similarity >= 0.8:
                        status = "✓ <span style='color:green'>High</span>"
                    elif similarity >= 0.6:
                        status = "✓ <span style='color:blue'>Medium</span>"
                    else:
                        status = "✓ <span style='color:orange'>Low</span>"
                else:
                    status = f"✓ ({similarity_str})"
                
                pred_data.append({
                    "Index": i+1, 
                    "Predicted Action": action, 
                    "Match Status": status, 
                    "Matched Ground Truth Action": f"#{gt_idx+1}: {gt_action}",
                    "Similarity": similarity_str
                })
            else:
                # Check if it's a rejected match
                reject_info = ""
                if rejected_matches and show_rejected and i in rejected_matches:
                    reject = rejected_matches[i]
                    gt_idx = reject["gt_idx"]
                    score = reject["score"]
                    reason = reject["reason"]
                    gt_action = gt_actions[gt_idx] if gt_idx < len(gt_actions) else "Out of range"
                    
                    if color_coding:
                        reject_info = f"<span style='color:red'>Rejected: {gt_action} (score:{score:.{precision}f}, reason:{reason})</span>"
                    else:
                        reject_info = f"Rejected: {gt_action} (score:{score:.{precision}f}, reason:{reason})"
                
                pred_data.append({
                    "Index": i+1, 
                    "Predicted Action": action, 
                    "Match Status": "✗", 
                    "Matched Ground Truth Action": "Not matched" if not reject_info else reject_info,
                    "Similarity": "-"
                })
        
        # Build ground truth sequence table
        gt_data = []
        matched_gt_indices = set(action_mapping.values())
        for i, action in enumerate(gt_actions):
            matched = i in matched_gt_indices
            if matched:
                pred_indices = [p_idx for p_idx, g_idx in action_mapping.items() if g_idx == i]
                pred_actions_matched = [pred_actions[p_idx] if p_idx < len(pred_actions) else "Out of range" for p_idx in pred_indices]
                pred_info = ", ".join([f"#{p_idx+1}: {pred_action}" for p_idx, pred_action in zip(pred_indices, pred_actions_matched)])
                
                # Calculate match quality
                similarities = [self.similarity_calculator.compute_action_similarity(pred_act, action, self.method) 
                             for pred_act in pred_actions_matched]
                avg_similarity = np.mean(similarities)
                similarity_str = f"{avg_similarity:.{precision}f}"
                
                if color_coding:
                    # Use color coding for match quality
                    if avg_similarity >= 0.8:
                        status = "✓ <span style='color:green'>High</span>"
                    elif avg_similarity >= 0.6:
                        status = "✓ <span style='color:blue'>Medium</span>"
                    else:
                        status = "✓ <span style='color:orange'>Low</span>"
                else:
                    status = f"✓ ({similarity_str})"
                
                gt_data.append({
                    "Index": i+1, 
                    "Ground Truth Action": action, 
                    "Match Status": status, 
                    "Matched Predicted Action": pred_info,
                    "Similarity": similarity_str
                })
            else:
                gt_data.append({
                    "Index": i+1, 
                    "Ground Truth Action": action, 
                    "Match Status": "✗", 
                    "Matched Predicted Action": "Not matched",
                    "Similarity": "-"
                })
        
        display(Markdown("#### Predicted Action Sequence"))
        if color_coding:
            display(HTML(pd.DataFrame(pred_data).to_html(escape=False)))
        else:
            display(pd.DataFrame(pred_data))
        
        display(Markdown("#### Ground Truth Action Sequence"))
        if color_coding:
            display(HTML(pd.DataFrame(gt_data).to_html(escape=False)))
        else:
            display(pd.DataFrame(gt_data))
    
    def display_detailed_results(self, video_id, details, scores):
        """
        Display detailed evaluation results including rejected match information
        
        Args:
            video_id: Video ID
            details: Detailed evaluation results
            scores: Scores dictionary
        """
        # Get display configuration
        display_config = self.config_manager.get_display_config()
        precision = display_config["precision"]
        show_metrics = display_config["show_metrics"]
        
        display(Markdown(f"## Detailed Evaluation Results for Video {video_id}"))
        
        # 1. Camera Motion Evaluation
        if "camera_motion" in details:
            camera_motion = details["camera_motion"]
            if show_metrics:
                display(Markdown(f"### Camera Motion Evaluation (Precision: {scores['camera_precision_score']:.{precision}f}, Recall: {scores['camera_recall_score']:.{precision}f}, Order Score: {scores['camera_order_score']:.{precision}f})"))
            else:
                display(Markdown(f"### Camera Motion Evaluation (Score: {scores['camera_motion_score']:.{precision}f}, Order Score: {scores['camera_order_score']:.{precision}f})"))
            
            # Show camera motion segment matching
            self.display_action_sequence_comparison(
                camera_motion["pred_segments"],
                camera_motion["gt_segments"],
                camera_motion["segment_mapping"]
            )
        
        # 2. Subject Matching Results
        if "subject_mapping" in details:
            subject_mapping = details["subject_mapping"]
            display(Markdown(f"### Subject Matching Results"))
            if subject_mapping:
                mapping_data = [{"Predicted Subject": k, "Ground Truth Subject": v} for k, v in subject_mapping.items()]
                display(pd.DataFrame(mapping_data))
            else:
                display(Markdown("*No valid subject matches found*"))
        
        # 3. Subject Action Evaluation
        if "subject_evaluations" in details:
            subject_evals = details["subject_evaluations"]
            
            if show_metrics:
                display(Markdown(f"### Subject Action Evaluation (Precision: {scores['subject_precision_score']:.{precision}f}, Recall: {scores['subject_recall_score']:.{precision}f}, Order Score: {scores['subject_order_score']:.{precision}f})"))
            else:
                display(Markdown(f"### Subject Action Evaluation (Overall Score: {scores['subject_action_score']:.{precision}f}, Order Score: {scores['subject_order_score']:.{precision}f})"))
            
            for subject_pair, eval_data in subject_evals.items():
                display(Markdown(f"#### {subject_pair}"))
                
                # Display action matches
                if "action_mapping" in eval_data:
                    action_mapping = eval_data["action_mapping"]
                    pred_actions = eval_data["pred_actions"]
                    gt_actions = eval_data["gt_actions"]
                    rejected_matches = eval_data.get("rejected_matches", {})
                    
                    # Use general display function
                    self.display_action_sequence_comparison(pred_actions, gt_actions, action_mapping, rejected_matches)
                    
                    # Show scores
                    if show_metrics:
                        display(Markdown(f"**Action Precision:** {eval_data['action_precision']:.{precision}f}"))
                        display(Markdown(f"**Action Recall:** {eval_data['action_recall']:.{precision}f}"))
                        display(Markdown(f"**Action F1 Score:** {eval_data['action_f1']:.{precision}f}"))
                    else:
                        display(Markdown(f"**Action Match Score:** {eval_data['action_f1']:.{precision}f}"))
                    
                    display(Markdown(f"**Action Order Score:** {eval_data['order_score']:.{precision}f}"))
                    display(Markdown(f"**Weight (based on action count):** {eval_data['weight']}"))
        
        # 4. Chronological Order Evaluation
        if "chronological_evaluation" in details:
            chrono_eval = details["chronological_evaluation"]
            
            if show_metrics:
                display(Markdown(f"### Chronological Order Evaluation (Precision: {scores['chrono_precision_score']:.{precision}f}, Recall: {scores['chrono_recall_score']:.{precision}f}, Order Score: {scores['chrono_order_score']:.{precision}f})"))
            else:
                display(Markdown(f"### Chronological Order Evaluation (Match Score: {scores['chrono_match_score']:.{precision}f}, Order Score: {scores['chrono_order_score']:.{precision}f})"))
            
            pred_sequence = chrono_eval["pred_sequence"]
            gt_sequence = chrono_eval["gt_sequence"]
            chrono_mapping = chrono_eval["chrono_mapping"]
            rejected_matches = chrono_eval.get("rejected_matches", {})
            
            # Extract action texts
            pred_actions = [action for action, _ in pred_sequence]
            gt_actions = [action for action, _ in gt_sequence]
            
            # Use action sequence comparison function
            self.display_action_sequence_comparison(pred_actions, gt_actions, chrono_mapping, rejected_matches)
        
        # 5. Final Score
        display(Markdown(f"### Final Composite Score: {scores['final_score']:.{precision}f}"))
    
    def evaluate_and_display(self, gt_annotations, pred_extractions, show_details=None):
        """
        Evaluate and display results
        
        Args:
            gt_annotations: Ground truth annotations
            pred_extractions: Predicted extractions
            show_details: Whether to show detailed results
            
        Returns:
            video_scores: Scores for each video
            avg_scores: Average scores across all videos
            video_details: Detailed evaluation results
            scores_df: DataFrame with video scores
            avg_scores_df: DataFrame with average scores
        """
        # Get default value from configuration
        if show_details is None:
            show_details = self.config_manager.get("display", "show_details")
        
        # Display configuration
        precision = self.config_manager.get("display", "precision")
        show_metrics = self.config_manager.get("display", "show_metrics")
        
        # Run evaluation
        start_time = time.time()
        if self.config_manager.get("display", "debug_level") > 0:
            print(f"Starting evaluation using {self.method} method...")
        
        video_scores, avg_scores, video_details = self.evaluate_videos(
            gt_annotations, pred_extractions
        )
        
        eval_time = time.time() - start_time
        if self.config_manager.get("display", "debug_level") > 0:
            print(f"Evaluation complete, time: {eval_time:.2f} seconds")
        
        # Display overall average scores
        display(Markdown("## Evaluation Summary Results"))
        
        # Choose display content based on metric detail preference
        if show_metrics:
            avg_scores_df = pd.DataFrame([{
                "Camera Motion Score": round(avg_scores["camera_motion_score"], precision),
                "Subject Action Precision": round(avg_scores["subject_precision_score"], precision),
                "Subject Action Recall": round(avg_scores["subject_recall_score"], precision),
                "Subject Order Score": round(avg_scores["subject_order_score"], precision),
                "Chronological Precision": round(avg_scores["chrono_precision_score"], precision),
                "Chronological Recall": round(avg_scores["chrono_recall_score"], precision),
                "Chronological Order Score": round(avg_scores["chrono_order_score"], precision),
                "Final Score": round(avg_scores["final_score"], precision)
            }])
        else:
            avg_scores_df = pd.DataFrame([{
                "Camera Motion Score": round(avg_scores["camera_motion_score"], precision),
                "Subject Action Score": round(avg_scores["subject_action_score"], precision),
                "Subject Order Score": round(avg_scores["subject_order_score"], precision),
                "Chronological Match Score": round(avg_scores["chrono_match_score"], precision),
                "Chronological Order Score": round(avg_scores["chrono_order_score"], precision),
                "Final Score": round(avg_scores["final_score"], precision)
            }])
            
        display(avg_scores_df)
        
        # Display detailed score distribution
        if show_details and len(video_scores) > 1:
            display(Markdown("### Score Distribution by Video"))
            
            # Choose table columns based on metric detail preference
            if show_metrics:
                scores_df = pd.DataFrame([
                    {
                        "Video ID": vid,
                        "Camera Motion Score": round(scores["camera_motion_score"], precision),
                        "Subject Action Precision": round(scores["subject_precision_score"], precision),
                        "Subject Action Recall": round(scores["subject_recall_score"], precision),
                        "Subject Order Score": round(scores["subject_order_score"], precision),
                        "Chronological Precision": round(scores["chrono_precision_score"], precision),
                        "Chronological Recall": round(scores["chrono_recall_score"], precision),
                        "Chronological Order Score": round(scores["chrono_order_score"], precision),
                        "Final Score": round(scores["final_score"], precision)
                    }
                    for vid, scores in video_scores.items()
                ])
            else:
                scores_df = pd.DataFrame([
                    {
                        "Video ID": vid,
                        "Camera Motion Score": round(scores["camera_motion_score"], precision),
                        "Subject Action Score": round(scores["subject_action_score"], precision),
                        "Subject Order Score": round(scores["subject_order_score"], precision),
                        "Chronological Match Score": round(scores["chrono_match_score"], precision),
                        "Chronological Order Score": round(scores["chrono_order_score"], precision),
                        "Final Score": round(scores["final_score"], precision)
                    }
                    for vid, scores in video_scores.items()
                ])
                
            scores_df = scores_df.sort_values("Final Score", ascending=False)
            display(scores_df)
        
        return video_scores, avg_scores, video_details, scores_df, avg_scores_df
    
    def set_config(self, config_dict):
        """
        Update evaluator configuration
        
        Args:
            config_dict: Configuration dictionary with updates
        """
        self.config_manager.update_config(config_dict)
        # Update method and weights
        self.method = self.config_manager.get("similarity", "method")
        self.weights = self.config_manager.get_scoring_weights()
    
    def get_config(self):
        """
        Get current configuration
        
        Returns:
            Current configuration dictionary
        """
        return self.config_manager.config
    
    def clear_model_cache(self):
        """Clear all model caches"""
        self.model_manager.clear_cache()
        self.text_processor.key_entities_cache.clear()
        self.text_processor.verb_synonyms_cache.clear()
        self.similarity_calculator._similarity_cache.clear()
        
    def get_flat_config(self):
        """
        Get flattened configuration (for UI display)
        
        Returns:
            Flattened configuration dictionary
        """
        return self.config_manager.get_flat_config()


# ===== Main Program and Configuration =====
def get_available_similarity_methods():
    """Get all available similarity calculation methods"""
    methods = [
        ('default', 'Automatically select best available model'),
        ('sbert-small', 'Use SBERT MiniLM (fastest)'),
        ('sbert-medium', 'Use SBERT RoBERTa (balanced speed/performance)'),
        ('simple', 'Use simple word overlap (no pretrained model required)')
    ]
    return methods

def main(input_path, output_path, gt_annotations_path):
    """
    Main function demonstrating the use of the evaluation tool
    
    Args:
        input_path: Path to predictions JSON file
        output_path: Path to save evaluation results CSV
        gt_annotations_path: Path to ground-truth annotations of FAVOR-Bench
    Returns:
        (evaluator, scores, avg_scores, details, scores_df, avg_scores_df) tuple
    """
    # Standard configuration
    standard_config = {
        "scoring_weights": {
            "camera_motion": 0.05,       # Camera motion score weight
            "subject_precision": 0.2,    # Subject action precision score weight
            "subject_recall": 0.2,       # Subject action recall score weight
            "subject_order": 0.05,       # Subject action order score weight
            "chrono_precision": 0.2,     # Chronological order precision score weight
            "chrono_recall": 0.2,        # Chronological order recall score weight
            "chrono_order": 0.1          # Chronological order correctness score weight
        },
        
        # Action matching parameters
        "action_matching": {
            "similarity_threshold": 0.3,  # Action matching minimum similarity threshold
            "length_penalty": 0.2,        # Sequence length difference penalty
        },
        
        # Order evaluation parameters
        "order_evaluation": {
            "min_sequence_for_order": 3,     # Minimum sequence length for order scoring
            "short_sequence_penalty": 0.2,   # Short sequence penalty coefficient
        },
    }
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            pred_extractions = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {input_path} not found")
        return None, None, None, None, None, None
    except json.JSONDecodeError:
        print(f"Error: JSON file {input_path} decoding failed. Please check the file format.")
        return None, None, None, None, None, None
    
    # Load ground truth annotations from specified path
    gt_annotations_path = gt_annotations_path
    try:
        with open(gt_annotations_path, 'r', encoding='utf-8') as f:
            loaded_gt_annotations = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {gt_annotations_path} not found")
        return None, None, None, None, None, None
    except json.JSONDecodeError:
        print(f"Error: JSON file {gt_annotations_path} decoding failed. Please check the file format.")
        return None, None, None, None, None, None
    
    # Filter ground truth annotations to match predictions
    gt_annotations = {}
    for video_id in pred_extractions.keys():
        if video_id in loaded_gt_annotations:
            gt_annotations[video_id] = loaded_gt_annotations[video_id]
        else:
            print(f"Warning: Video ID {video_id} not found in ground truth annotations. Evaluation will skip this video.")

    if not gt_annotations:
        print("Warning: No matching video IDs found in ground truth annotations. Evaluation cannot proceed.")
        return None, None, None, None, None, None

    # Create evaluator instance with standard configuration
    evaluator = VideoActionEvaluator(standard_config)
    
    # Run evaluation
    start_time = time.time()
    scores, avg_scores, details, scores_df, avg_scores_df = evaluator.evaluate_and_display(
        gt_annotations, pred_extractions, show_details=True
    )
    eval_time = time.time() - start_time
    
    print(f"\nTotal evaluation time: {eval_time:.2f} seconds")

    # Save results to CSV file
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            avg_scores_df.to_csv(f, header=True, index=True)
            f.write("\n")  # Add spacing
            scores_df.to_csv(f, header=True, index=True)
        print(f"Evaluation results saved to: {output_path}")
    else:
        print("No output_path specified, evaluation results will not be saved to file.")
    
    return evaluator, scores, avg_scores, details, scores_df, avg_scores_df

if __name__ == "__main__":
    input_path = "extract_results/LLM-free_case.json"
    output_path = "cal_scores/LLM-free_case.csv"
    gt_annotations_path = "./FAVOR_gt_annotations.json"
    os.makedirs('cal_scores', exist_ok=True)
    evaluator, scores, avg_scores, details, scores_df, avg_scores_df = main(input_path=input_path, output_path=output_path, gt_annotations_path=gt_annotations_path)
