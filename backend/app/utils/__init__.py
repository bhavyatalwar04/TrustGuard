"""
Utility modules for TruthGuard backend
"""

from .helpers import (
    generate_hash,
    normalize_text,
    extract_urls,
    validate_url,
    get_domain,
    clean_html,
    truncate_text,
    calculate_confidence_score,
    format_timestamp,
    parse_timestamp,
    sanitize_filename,
    chunk_text,
    merge_dicts,
    calculate_similarity,
    extract_key_phrases,
    validate_json,
    safe_divide,
    format_file_size,
    Timer,
    batch_process
)

from .text_processing import TextProcessor

__all__ = [
    'generate_hash',
    'normalize_text',
    'extract_urls',
    'validate_url',
    'get_domain',
    'clean_html',
    'truncate_text',
    'calculate_confidence_score',
    'format_timestamp',
    'parse_timestamp',
    'sanitize_filename',
    'chunk_text',
    'merge_dicts',
    'calculate_similarity',
    'extract_key_phrases',
    'validate_json',
    'safe_divide',
    'format_file_size',
    'Timer',
    'batch_process',
    'TextProcessor'
]