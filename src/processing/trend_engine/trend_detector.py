import pandas as pd
from collections import Counter
from typing import List, Dict, Tuple, Optional
import logging
from gensim import corpora
from gensim.models import LdaModel
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import glob
import os
from pathlib import Path

# Setup Logging (ensure this is consistent across your project)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TrendDetectionEngine:
    """
    A class for detecting trends and topics in preprocessed text data.
    Optimized for social media data analysis.
    """
    def __init__(self, data_dir: str = "data/processed_data"):
        self.lda_model = None
        self.dictionary = None
        self.data_dir = data_dir
        logging.info("TrendDetectionEngine initialized.")

    def find_latest_preprocessed_file(self, pattern: str = "preprocessed_reddit_posts_advanced_*.csv") -> Optional[str]:
        """
        Find the most recent preprocessed file based on timestamp in filename.
        
        Args:
            pattern (str): File pattern to search for
            
        Returns:
            Optional[str]: Path to the most recent file, or None if no files found
        """
        search_pattern = os.path.join(self.data_dir, pattern)
        files = glob.glob(search_pattern)
        
        if not files:
            # Try current directory as fallback
            files = glob.glob(pattern)
            
        if not files:
            logging.error(f"No files found matching pattern: {pattern}")
            return None
            
        # Sort files by modification time (most recent first)
        files.sort(key=os.path.getmtime, reverse=True)
        latest_file = files[0]
        
        logging.info(f"Found latest preprocessed file: {latest_file}")
        return latest_file

    def load_preprocessed_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load preprocessed data from CSV file.
        
        Args:
            file_path (Optional[str]): Specific file path, or None to auto-detect latest
            
        Returns:
            pd.DataFrame: Loaded preprocessed data
        """
        if file_path is None:
            file_path = self.find_latest_preprocessed_file()
            
        if file_path is None:
            raise FileNotFoundError("No preprocessed data file found")
            
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Loaded {len(df)} records from {file_path}")
            return df
        except Exception as e:
            logging.error(f"Error loading file {file_path}: {e}")
            raise

    def get_top_ngrams(
        self, 
        texts: List[str], 
        n: int = 1, 
        top_k: int = 20,
        min_freq: int = 2
    ) -> List[Tuple[str, int]]:
        """
        Calculates the frequency of top N-grams (words or phrases).

        Args:
            texts (List[str]): A list of strings, where each string is a space-separated sequence of tokens/lemmas.
            n (int): The 'n' for N-grams (1 for unigrams, 2 for bigrams, etc.).
            top_k (int): The number of top N-grams to return.
            min_freq (int): Minimum frequency threshold for N-grams.

        Returns:
            List[Tuple[str, int]]: A list of (N-gram, count) tuples, sorted by count.
        """
        if not texts:
            logging.warning("No texts provided for N-gram analysis.")
            return []

        all_ngrams = []
        for text in texts:
            # Ensure text is treated as a string, split into words
            words = str(text).split()
            if len(words) < n:
                continue
                
            if n == 1:
                all_ngrams.extend(words)
            else:
                # Generate N-grams
                ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
                all_ngrams.extend(ngrams)

        # Filter by minimum frequency
        ngram_counts = Counter(all_ngrams)
        filtered_counts = {k: v for k, v in ngram_counts.items() if v >= min_freq}
        
        logging.info(f"Calculated top {top_k} {n}-grams (min_freq={min_freq}).")
        return Counter(filtered_counts).most_common(top_k)

    def perform_lda_topic_modeling(
        self,
        texts: List[str],
        num_topics: int = 5,
        num_words: int = 10,
        passes: int = 10,
        min_word_count: int = 5,
        max_vocab_size: int = 10000
    ) -> List[List[Tuple[str, float]]]:
        """
        Performs Latent Dirichlet Allocation (LDA) topic modeling on a collection of texts.

        Args:
            texts (List[str]): A list of strings, where each string is a space-separated sequence of tokens/lemmas.
            num_topics (int): The number of topics to discover.
            num_words (int): The number of top words to show for each topic.
            passes (int): Number of passes through the corpus during training.
            min_word_count (int): Minimum word frequency in corpus.
            max_vocab_size (int): Maximum vocabulary size.

        Returns:
            List[List[Tuple[str, float]]]: A list of topics, where each topic is a list of (word, probability) tuples.
        """
        if not texts:
            logging.warning("No texts provided for LDA topic modeling.")
            return []

        # Convert list of strings to list of lists of words for Gensim
        documents = [str(text).split() for text in texts]
        documents = [doc for doc in documents if doc and len(doc) > 2]  # Remove empty/very short documents

        if not documents:
            logging.warning("No valid documents after splitting for LDA.")
            return []

        # Create a dictionary from the documents
        self.dictionary = corpora.Dictionary(documents)
        
        # Filter out words that occur less than min_word_count times or in more than 50% of docs
        self.dictionary.filter_extremes(
            no_below=min_word_count,
            no_above=0.5,
            keep_n=max_vocab_size
        )

        # Create a corpus (Bag of Words representation)
        corpus = [self.dictionary.doc2bow(doc) for doc in documents]
        corpus = [doc for doc in corpus if doc]  # Remove empty documents

        if not corpus:
            logging.warning("No valid corpus after filtering for LDA.")
            return []

        # Train the LDA model
        self.lda_model = LdaModel(
            corpus=corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            random_state=100,  # for reproducibility
            update_every=1,
            chunksize=100,
            passes=passes,
            alpha='auto',
            per_word_topics=True
        )

        topics = self.lda_model.print_topics(num_words=num_words)
        
        # Format the output for better readability
        formatted_topics = []
        for topic_id, topic_words in topics:
            # topic_words is like "0.050*word1 + 0.030*word2"
            words = []
            for item in topic_words.split(' + '):
                weight, word = item.split('*')
                words.append((word.strip().replace('"', ''), float(weight)))
            formatted_topics.append(words)

        logging.info(f"LDA topic modeling completed with {num_topics} topics.")
        return formatted_topics

    def get_document_topics(self, texts: List[str]) -> List[List[Tuple[int, float]]]:
        """
        Assigns topics to each document using a trained LDA model.

        Args:
            texts (List[str]): A list of strings (documents).

        Returns:
            List[List[Tuple[int, float]]]: A list where each inner list contains (topic_id, probability) tuples for a document.
        """
        if self.lda_model is None or self.dictionary is None:
            logging.error("LDA model not trained. Call perform_lda_topic_modeling first.")
            return []

        documents = [str(text).split() for text in texts]
        corpus = [self.dictionary.doc2bow(doc) for doc in documents]
        
        doc_topics = []
        for doc_bow in corpus:
            # lda_model[doc_bow] returns a list of (topic_id, probability) tuples
            main_topics = self.lda_model.get_document_topics(doc_bow, per_word_topics=False)
            doc_topics.append(main_topics)
        return doc_topics

    def analyze_trends_over_time(self, df: pd.DataFrame, time_column: str = 'created_utc', 
                                text_column: str = 'lemmas', window_days: int = 7) -> Dict:
        """
        Analyze how trends change over time.
        
        Args:
            df (pd.DataFrame): DataFrame with text and time data
            time_column (str): Column name for timestamp
            text_column (str): Column name for text data
            window_days (int): Time window for trend analysis
            
        Returns:
            Dict: Trend analysis results
        """
        try:
            # Convert timestamp to datetime
            df[time_column] = pd.to_datetime(df[time_column], unit='s')
            df = df.sort_values(time_column)
            
            # Group by time windows
            df['time_window'] = df[time_column].dt.floor(f'{window_days}D')
            
            trends_over_time = {}
            for window, group in df.groupby('time_window'):
                texts = group[text_column].dropna().tolist()
                if texts:
                    top_terms = self.get_top_ngrams(texts, n=1, top_k=10)
                    trends_over_time[window] = top_terms
                    
            logging.info(f"Analyzed trends over {len(trends_over_time)} time windows")
            return trends_over_time
            
        except Exception as e:
            logging.error(f"Error in trend analysis over time: {e}")
            return {}

    def get_subreddit_trends(self, df: pd.DataFrame, text_column: str = 'lemmas', 
                            subreddit_column: str = 'subreddit') -> Dict:
        """
        Analyze trends by subreddit.
        
        Args:
            df (pd.DataFrame): DataFrame with text and subreddit data
            text_column (str): Column name for text data
            subreddit_column (str): Column name for subreddit
            
        Returns:
            Dict: Trends by subreddit
        """
        try:
            subreddit_trends = {}
            for subreddit, group in df.groupby(subreddit_column):
                texts = group[text_column].dropna().tolist()
                if texts and len(texts) >= 5:  # Minimum posts for meaningful analysis
                    top_terms = self.get_top_ngrams(texts, n=1, top_k=15)
                    subreddit_trends[subreddit] = {
                        'post_count': len(texts),
                        'top_terms': top_terms
                    }
                    
            logging.info(f"Analyzed trends for {len(subreddit_trends)} subreddits")
            return subreddit_trends
            
        except Exception as e:
            logging.error(f"Error in subreddit trend analysis: {e}")
            return {}

    def run_full_analysis(self, file_path: Optional[str] = None) -> Dict:
        """
        Run complete trend analysis on preprocessed data.
        
        Args:
            file_path (Optional[str]): Path to preprocessed file, or None for auto-detection
            
        Returns:
            Dict: Complete analysis results
        """
        # Load data
        df = self.load_preprocessed_data(file_path)
        
        # Prepare text data
        corpus_for_analysis = df['lemmas'].dropna().tolist()
        
        if not corpus_for_analysis:
            logging.error("No valid text data found for analysis")
            return {}

        results = {
            'data_summary': {
                'total_posts': len(df),
                'valid_text_posts': len(corpus_for_analysis),
                'unique_subreddits': df['subreddit'].nunique(),
                'date_range': {
                    'start': pd.to_datetime(df['created_utc'], unit='s').min().isoformat(),
                    'end': pd.to_datetime(df['created_utc'], unit='s').max().isoformat()
                }
            }
        }

        # N-gram analysis
        logging.info("Running N-gram analysis...")
        results['ngrams'] = {
            'unigrams': self.get_top_ngrams(corpus_for_analysis, n=1, top_k=20),
            'bigrams': self.get_top_ngrams(corpus_for_analysis, n=2, top_k=15),
            'trigrams': self.get_top_ngrams(corpus_for_analysis, n=3, top_k=10)
        }

        # Topic modeling
        logging.info("Running LDA topic modeling...")
        topics = self.perform_lda_topic_modeling(corpus_for_analysis, num_topics=8, num_words=10)
        results['topics'] = topics

        # Assign topics to documents
        if self.lda_model and self.dictionary:
            doc_topics_list = []
            documents = [str(text).split() for text in corpus_for_analysis]
            corpus_bow = [self.dictionary.doc2bow(doc) for doc in documents]
            
            for doc_bow in corpus_bow:
                doc_topics = self.lda_model.get_document_topics(doc_bow, per_word_topics=False)
                if doc_topics:
                    dominant_topic = max(doc_topics, key=lambda item: item[1])
                    doc_topics_list.append(f"Topic {dominant_topic[0]} ({dominant_topic[1]:.2f})")
                else:
                    doc_topics_list.append("No Topic")

            # Add dominant topic to DataFrame
            df['dominant_topic'] = pd.Series(doc_topics_list, index=df.index[:len(doc_topics_list)])
            df['dominant_topic'] = df['dominant_topic'].fillna("No Topic")

        # Subreddit trends
        logging.info("Analyzing subreddit trends...")
        results['subreddit_trends'] = self.get_subreddit_trends(df)

        # Time-based trends
        logging.info("Analyzing trends over time...")
        results['time_trends'] = self.analyze_trends_over_time(df)

        # Save enhanced data
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{self.data_dir}/preprocessed_reddit_posts_with_topics_{timestamp}.csv"
        
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            df.to_csv(output_file, index=False)
            logging.info(f"Enhanced data saved to: {output_file}")
            results['output_file'] = output_file
        except Exception as e:
            logging.error(f"Error saving enhanced data: {e}")

        return results


def print_analysis_results(results: Dict):
    """Pretty print analysis results."""
    
    print("\n" + "="*60)
    print("SOCIAL MEDIA TREND ANALYSIS RESULTS")
    print("="*60)
    
    # Data summary
    summary = results.get('data_summary', {})
    print(f"\n📊 DATA SUMMARY:")
    print(f"   Total Posts: {summary.get('total_posts', 'N/A'):,}")
    print(f"   Valid Text Posts: {summary.get('valid_text_posts', 'N/A'):,}")
    print(f"   Unique Subreddits: {summary.get('unique_subreddits', 'N/A')}")
    date_range = summary.get('date_range', {})
    if date_range:
        print(f"   Date Range: {date_range.get('start', 'N/A')} to {date_range.get('end', 'N/A')}")

    # N-grams
    ngrams = results.get('ngrams', {})
    
    print(f"\n🔤 TOP UNIGRAMS (Single Words):")
    for word, count in ngrams.get('unigrams', [])[:10]:
        print(f"   '{word}': {count:,}")

    print(f"\n🔤 TOP BIGRAMS (Two-word Phrases):")
    for phrase, count in ngrams.get('bigrams', [])[:10]:
        print(f"   '{phrase}': {count:,}")

    print(f"\n🔤 TOP TRIGRAMS (Three-word Phrases):")
    for phrase, count in ngrams.get('trigrams', [])[:8]:
        print(f"   '{phrase}': {count:,}")

    # Topics
    topics = results.get('topics', [])
    print(f"\n🎯 DISCOVERED TOPICS:")
    for i, topic_words in enumerate(topics):
        words_str = ", ".join([f"{word}({prob:.3f})" for word, prob in topic_words[:5]])
        print(f"   Topic #{i}: {words_str}")

    # Subreddit trends
    subreddit_trends = results.get('subreddit_trends', {})
    if subreddit_trends:
        print(f"\n🌐 TOP SUBREDDIT TRENDS:")
        sorted_subreddits = sorted(subreddit_trends.items(), 
                                 key=lambda x: x[1]['post_count'], reverse=True)[:5]
        for subreddit, data in sorted_subreddits:
            print(f"   r/{subreddit} ({data['post_count']} posts):")
            top_terms = data['top_terms'][:3]
            terms_str = ", ".join([f"{term}({count})" for term, count in top_terms])
            print(f"     Top terms: {terms_str}")

    print(f"\n✅ Analysis complete!")
    if 'output_file' in results:
        print(f"📁 Enhanced data saved to: {results['output_file']}")


# --- Example Usage ---
if __name__ == "__main__":
    # Initialize trend detection engine
    trend_detector = TrendDetectionEngine(data_dir="data/processed_data")
    
    try:
        # Run full analysis
        results = trend_detector.run_full_analysis()
        
        # Print results
        print_analysis_results(results)
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        print(f"❌ Analysis failed: {e}")