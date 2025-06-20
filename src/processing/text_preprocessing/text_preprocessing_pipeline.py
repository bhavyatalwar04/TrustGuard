import re
import string
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
import spacy
from bs4 import BeautifulSoup
import logging
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import warnings
import glob
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", message="\[W108\]")

# Setup Enhanced Logging
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup enhanced logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers,
        force=True
    )

def get_latest_raw_data_file(data_dir: str = "data/raw_posts") -> Optional[str]:
    """
    Automatically find the most recent raw Reddit data file.
    
    Args:
        data_dir: Directory containing raw data files
        
    Returns:
        Path to the latest file or None if no files found
    """
    logger = logging.getLogger(__name__)
    
    # Create Path object for better path handling
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"Data directory '{data_dir}' does not exist")
        return None
    
    # Look for CSV files matching the pattern
    pattern = "posts_reddit_data_*.csv"
    csv_files = list(data_path.glob(pattern))
    
    if not csv_files:
        logger.warning(f"No files matching pattern '{pattern}' found in '{data_dir}'")
        return None
    
    # Sort files by modification time (most recent first)
    csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    latest_file = csv_files[0]
    logger.info(f"Found {len(csv_files)} raw data files. Using latest: {latest_file.name}")
    
    return str(latest_file)

class TextPreprocessor:
    """
    Enhanced text preprocessor with improved performance, error handling, and features.
    """
    
    def __init__(self, 
                 model_name: str = "en_core_web_sm",
                 batch_size: int = 50,
                 n_process: int = 1,
                 disable_pipes: Optional[List[str]] = None):
        """
        Initialize the TextPreprocessor with configurable options.
        
        Args:
            model_name: spaCy model name to load
            batch_size: Batch size for spaCy processing
            n_process: Number of processes for spaCy (use -1 for all CPUs)
            disable_pipes: List of spaCy pipeline components to disable for performance
        """
        self.logger = logging.getLogger(__name__)
        self.batch_size = batch_size
        self.n_process = n_process
        
        # Default pipes to disable for better performance when not needed
        if disable_pipes is None:
            disable_pipes = []
        
        self.disable_pipes = disable_pipes
        
        # Compile regex patterns once for better performance
        self._compile_regex_patterns()
        
        try:
            self.nlp = spacy.load(model_name, disable=disable_pipes)
            # Configure for better performance
            if batch_size > 1:
                self.nlp.max_length = 2000000  # Increase max length for batch processing
            if 'parser' in disable_pipes and 'sentencizer' not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")
                self.logger.info("Added 'sentencizer' to pipeline for sentence segmentation.")
            self.logger.info(f"TextPreprocessor initialized with model: {model_name}")
            self.logger.info(f"Disabled pipes: {disable_pipes}")
            
        except OSError as e:
            self.logger.error(f"Failed to load spaCy model '{model_name}': {e}")
            self.logger.error(f"Install it with: python -m spacy download {model_name}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during initialization: {e}")
            raise

    def _compile_regex_patterns(self):
        """Compile regex patterns once for better performance."""
        self.patterns = {
            'html_tags': re.compile(r'<[^>]+>'),
            'urls': re.compile(r'http\S+|www\S+|https\S+', re.MULTILINE | re.IGNORECASE),
            'reddit_mentions': re.compile(r'[ur]/\S+', re.IGNORECASE),
            'file_paths': re.compile(r'[A-Za-z]:[/\\][^\s<>:"|?*]+'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}'),
            'multiple_whitespace': re.compile(r'\s+'),
            'punctuation': re.compile(f'[{re.escape(string.punctuation)}]'),
            'digits': re.compile(r'\d+'),
            'non_ascii': re.compile(r'[^\x00-\x7F]+'),
        }

    def clean_text(self, 
                   text: str, 
                   remove_urls: bool = True,
                   remove_emails: bool = True,
                   remove_phone: bool = True,
                   remove_digits: bool = False,
                   remove_non_ascii: bool = False,
                   custom_patterns: Optional[Dict[str, str]] = None) -> str:
        """
        Enhanced text cleaning with configurable options.
        
        Args:
            text: Input text to clean
            remove_urls: Remove URLs
            remove_emails: Remove email addresses
            remove_phone: Remove phone numbers
            remove_digits: Remove all digits
            remove_non_ascii: Remove non-ASCII characters
            custom_patterns: Dictionary of custom regex patterns to apply
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # Remove HTML tags using BeautifulSoup for better handling
        try:
            soup = BeautifulSoup(text, 'html.parser')
            text = soup.get_text(separator=' ')
        except Exception as e:
            self.logger.warning(f"HTML parsing failed, using regex fallback: {e}")
            text = self.patterns['html_tags'].sub(' ', text)

        # Apply cleaning patterns
        if remove_urls:
            text = self.patterns['urls'].sub(' ', text)
        
        if remove_emails:
            text = self.patterns['email'].sub(' ', text)
            
        if remove_phone:
            text = self.patterns['phone'].sub(' ', text)

        # Reddit-specific cleaning
        text = self.patterns['reddit_mentions'].sub(' ', text)
        text = self.patterns['file_paths'].sub(' ', text)

        # Apply custom patterns if provided
        if custom_patterns:
            for pattern_name, pattern in custom_patterns.items():
                try:
                    text = re.sub(pattern, ' ', text)
                except re.error as e:
                    self.logger.warning(f"Invalid regex pattern '{pattern_name}': {e}")

        # Optional cleaning
        if remove_digits:
            text = self.patterns['digits'].sub(' ', text)
            
        if remove_non_ascii:
            text = self.patterns['non_ascii'].sub(' ', text)

        # Remove punctuation and normalize whitespace
        text = self.patterns['punctuation'].sub(' ', text)
        text = self.patterns['multiple_whitespace'].sub(' ', text).strip()

        return text

    def _process_with_spacy(self, texts: Union[str, List[str]]) -> Union[spacy.tokens.Doc, List[spacy.tokens.Doc]]:
        """Process text(s) with spaCy, handling both single strings and batches."""
        if isinstance(texts, str):
            if not texts.strip():
                return self.nlp("")
            return self.nlp(texts)
        
        # Batch processing
        valid_texts = [text if isinstance(text, str) and text.strip() else "" for text in texts]
        
        try:
            if self.n_process > 1:
                docs = list(self.nlp.pipe(valid_texts, batch_size=self.batch_size, n_process=self.n_process))
            else:
                docs = list(self.nlp.pipe(valid_texts, batch_size=self.batch_size))
            return docs
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            # Fallback to individual processing
            return [self.nlp(text) for text in valid_texts]

    def extract_entities_and_pos(self, doc: spacy.tokens.Doc) -> Dict[str, List[Dict]]:
        """Extract Named Entities and POS tags with additional linguistic features."""
        if not isinstance(doc, spacy.tokens.doc.Doc):
            return {
                "entities": [],
                "pos_tags": [],
                "noun_phrases": [],
                "dependency_relations": []
            }

        # Named entities with confidence scores if available
        entities = []
        for ent in doc.ents:
            entity_info = {
                "text": ent.text,
                "label": ent.label_,
                "description": spacy.explain(ent.label_),
                "start": ent.start_char,
                "end": ent.end_char
            }
            entities.append(entity_info)

        # POS tags with additional info
        pos_tags = []
        for token in doc:
            if not token.is_space:  # Skip whitespace tokens
                pos_info = {
                    "text": token.text,
                    "pos": token.pos_,
                    "tag": token.tag_,
                    "lemma": token.lemma_,
                    "is_alpha": token.is_alpha,
                    "is_stop": token.is_stop
                }
                pos_tags.append(pos_info)

        # Noun phrases
        noun_phrases = [{"text": chunk.text, "label": chunk.label_} for chunk in doc.noun_chunks]

        # Basic dependency relations
        dep_relations = []
        for token in doc:
            if not token.is_space:
                dep_info = {
                    "text": token.text,
                    "dep": token.dep_,
                    "head": token.head.text,
                    "children": [child.text for child in token.children]
                }
                dep_relations.append(dep_info)

        return {
            "entities": entities,
            "pos_tags": pos_tags,
            "noun_phrases": noun_phrases,
            "dependency_relations": dep_relations
        }

    def get_text_statistics(self, text: str, doc: Optional[spacy.tokens.Doc] = None) -> Dict:
        """Get comprehensive text statistics."""
        if doc is None:
            doc = self._process_with_spacy(text)
        
        sentences = list(doc.sents)
        tokens = [token for token in doc if not token.is_space]
        
        return {
            "char_count": len(text),
            "word_count": len([token for token in tokens if token.is_alpha]),
            "sentence_count": len(sentences),
            "avg_sentence_length": sum(len(sent.text.split()) for sent in sentences) / len(sentences) if sentences else 0,
            "unique_words": len(set(token.text.lower() for token in tokens if token.is_alpha)),
            "stopword_ratio": len([token for token in tokens if token.is_stop]) / len(tokens) if tokens else 0,
            "punctuation_count": len([token for token in tokens if not token.is_alpha and not token.is_space]),
        }

    def preprocess(self, 
                   text: str, 
                   include_ner_pos: bool = False,
                   include_statistics: bool = False,
                   cleaning_options: Optional[Dict] = None) -> Dict:
        """
        Enhanced preprocessing pipeline with comprehensive options.
        """
        if not isinstance(text, str) or not text.strip():
            return self._get_empty_result(include_ner_pos, include_statistics)

        # Apply cleaning with custom options
        clean_options = cleaning_options or {}
        cleaned_text = self.clean_text(text, **clean_options)
        
        if not cleaned_text:
            return self._get_empty_result(include_ner_pos, include_statistics)

        # Process with spaCy
        doc = self._process_with_spacy(cleaned_text)

        # Extract basic features
        tokens = [token.text.lower() for token in doc 
                 if token.is_alpha and not token.is_stop and len(token.text) > 1]
        
        lemmas = [token.lemma_.lower() for token in doc 
                 if token.is_alpha and not token.is_stop and len(token.lemma_) > 1]
        
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        # Build result
        result = {
            "cleaned_text": cleaned_text,
            "tokens": tokens,
            "lemmas": lemmas,
            "sentences": sentences,
            "token_count": len(tokens),
            "unique_tokens": len(set(tokens))
        }

        # Optional advanced features
        if include_ner_pos:
            result["linguistic_features"] = self.extract_entities_and_pos(doc)

        if include_statistics:
            result["statistics"] = self.get_text_statistics(cleaned_text, doc)

        return result

    def _get_empty_result(self, include_ner_pos: bool, include_statistics: bool) -> Dict:
        """Return standardized empty result."""
        result = {
            "cleaned_text": "",
            "tokens": [],
            "lemmas": [],
            "sentences": [],
            "token_count": 0,
            "unique_tokens": 0
        }
        
        if include_ner_pos:
            result["linguistic_features"] = {
                "entities": [],
                "pos_tags": [],
                "noun_phrases": [],
                "dependency_relations": []
            }
            
        if include_statistics:
            result["statistics"] = {
                "char_count": 0,
                "word_count": 0,
                "sentence_count": 0,
                "avg_sentence_length": 0,
                "unique_words": 0,
                "stopword_ratio": 0,
                "punctuation_count": 0
            }
            
        return result

    def preprocess_batch(self, 
                        texts: List[str], 
                        include_ner_pos: bool = False,
                        include_statistics: bool = False,
                        cleaning_options: Optional[Dict] = None) -> List[Dict]:
        """Process multiple texts efficiently using batch processing."""
        if not texts:
            return []

        # Clean all texts first
        clean_options = cleaning_options or {}
        cleaned_texts = [self.clean_text(text, **clean_options) for text in texts]
        
        # Filter out empty texts but keep track of original indices
        valid_texts = []
        text_indices = []
        for i, cleaned in enumerate(cleaned_texts):
            if cleaned:
                valid_texts.append(cleaned)
                text_indices.append(i)

        if not valid_texts:
            return [self._get_empty_result(include_ner_pos, include_statistics) for _ in texts]

        # Batch process with spaCy
        docs = self._process_with_spacy(valid_texts)
        
        # Initialize results array
        results = [self._get_empty_result(include_ner_pos, include_statistics) for _ in texts]
        
        # Process each valid document
        for doc, original_idx in zip(docs, text_indices):
            cleaned_text = cleaned_texts[original_idx]
            
            tokens = [token.text.lower() for token in doc 
                     if token.is_alpha and not token.is_stop and len(token.text) > 1]
            
            lemmas = [token.lemma_.lower() for token in doc 
                     if token.is_alpha and not token.is_stop and len(token.lemma_) > 1]
            
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

            result = {
                "cleaned_text": cleaned_text,
                "tokens": tokens,
                "lemmas": lemmas,
                "sentences": sentences,
                "token_count": len(tokens),
                "unique_tokens": len(set(tokens))
            }

            if include_ner_pos:
                result["linguistic_features"] = self.extract_entities_and_pos(doc)

            if include_statistics:
                result["statistics"] = self.get_text_statistics(cleaned_text, doc)

            results[original_idx] = result

        return results


def process_reddit_posts(csv_path: Optional[str] = None,
                        output_path: Optional[str] = None,
                        include_ner_pos: bool = False,
                        include_statistics: bool = False,
                        batch_size: int = 50,
                        max_workers: int = 4,
                        cleaning_options: Optional[Dict] = None,
                        auto_detect_latest: bool = True) -> None:
    """
    Enhanced CSV processing with batch processing, parallel execution, and automatic file detection.
    
    Args:
        csv_path: Path to input CSV file. If None and auto_detect_latest=True, finds latest file automatically
        output_path: Path for output CSV file. If None, generates based on input filename
        include_ner_pos: Include Named Entity Recognition and Part-of-Speech features
        include_statistics: Include text statistics
        batch_size: Size of processing batches
        max_workers: Number of worker processes
        cleaning_options: Custom text cleaning options
        auto_detect_latest: Automatically detect latest raw data file if csv_path is None
    """
    logger = logging.getLogger(__name__)
    
    # Auto-detect latest file if not specified
    if csv_path is None and auto_detect_latest:
        csv_path = get_latest_raw_data_file()
        if csv_path is None:
            raise FileNotFoundError("No raw data files found and no csv_path specified")
    
    if csv_path is None:
        raise ValueError("csv_path must be specified or auto_detect_latest must be True")
    
    # Validate input file
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")
    
    # Generate output path if not specified
    if output_path is None:
        input_path = Path(csv_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_advanced" if (include_ner_pos or include_statistics) else "_basic"
        output_path = f"data/processed_data/preprocessed_reddit_posts{suffix}_{timestamp}.csv"
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from {csv_path}")
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        raise

    # Initialize preprocessor with batch processing
    disable_pipes = ['parser', 'tagger'] if not include_ner_pos else []
    preprocessor = TextPreprocessor(
        batch_size=batch_size,
        n_process=max_workers,
        disable_pipes=disable_pipes
    )

    def extract_text_from_row(row: pd.Series) -> str:
        """Extract and combine text from various columns."""
        text_parts = []
        
        # Standard columns to check
        text_columns = ['title', 'content', 'selftext', 'body']
        
        for col in text_columns:
            if col in row and pd.notna(row[col]):
                text = str(row[col]).strip()
                if text and text.lower() not in ['', '[deleted]', '[removed]', 'nan']:
                    text_parts.append(text)
        
        return " ".join(text_parts)

    def process_batch(batch_df: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of rows."""
        texts = [extract_text_from_row(row) for _, row in batch_df.iterrows()]
        
        # Batch preprocess
        results = preprocessor.preprocess_batch(
            texts, 
            include_ner_pos=include_ner_pos,
            include_statistics=include_statistics,
            cleaning_options=cleaning_options
        )
        
        # Convert results to DataFrame columns
        processed_data = []
        for result in results:
            row_data = {
                "cleaned_text": result['cleaned_text'],
                "tokens": " ".join(result['tokens']) if result['tokens'] else "",
                "lemmas": " ".join(result['lemmas']) if result['lemmas'] else "",
                "sentences": " || ".join(result['sentences']) if result['sentences'] else "",
                "token_count": result['token_count'],
                "unique_tokens": result['unique_tokens']
            }
            
            if include_ner_pos:
                ling_features = result['linguistic_features']
                row_data.update({
                    "entities": json.dumps(ling_features['entities']),
                    "pos_tags": json.dumps(ling_features['pos_tags']),
                    "noun_phrases": json.dumps(ling_features['noun_phrases'])
                })
            
            if include_statistics:
                stats = result['statistics']
                row_data.update({
                    f"stat_{key}": value for key, value in stats.items()
                })
            
            processed_data.append(row_data)
        
        processed_df = pd.DataFrame(processed_data)
        return pd.concat([batch_df.reset_index(drop=True), processed_df], axis=1)

    # Process in batches
    logger.info(f"Processing {len(df)} rows in batches of {batch_size}")
    
    processed_batches = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size].copy()
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")
        
        try:
            processed_batch = process_batch(batch)
            processed_batches.append(processed_batch)
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
            # Add empty processed columns to maintain structure
            empty_batch = batch.copy()
            for col in ["cleaned_text", "tokens", "lemmas", "sentences", "token_count", "unique_tokens"]:
                empty_batch[col] = ""
            processed_batches.append(empty_batch)

    # Combine all batches
    final_df = pd.concat(processed_batches, ignore_index=True)
    
    # Save results
    try:
        final_df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved {len(final_df)} processed rows to {output_path}")
        logger.info(f"Output file: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise


if __name__ == "__main__":
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Process with automatic latest file detection
    configs = [
        {
            "output_file": None,  # Auto-generate filename
            "include_ner_pos": False,
            "include_statistics": False,
            "description": "Basic preprocessing with automatic file detection"
        },
        {
            "output_file": None,  # Auto-generate filename 
            "include_ner_pos": True,
            "include_statistics": True,
            "description": "Advanced preprocessing with NER/POS and statistics"
        }
    ]
    
    for config in configs:
        print(f"\n--- Running {config['description']} ---")
        try:
            process_reddit_posts(
                csv_path=None,  # Auto-detect latest file
                output_path=config['output_file'],
                include_ner_pos=config['include_ner_pos'],
                include_statistics=config['include_statistics'],
                batch_size=50,
                max_workers=4,
                auto_detect_latest=True
            )
            print(f"✅ {config['description']} complete.")
        except Exception as e:
            print(f"❌ {config['description']} failed: {e}")