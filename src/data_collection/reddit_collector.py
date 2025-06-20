# TruthGuard/data_collection/social_media_collector.py

import praw
import pandas as pd
import time
import logging
from datetime import datetime
from typing import List, Optional
import sqlite3
from dataclasses import dataclass
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reddit_collector.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class RedditPost:
    id: str
    title: str
    content: str
    author: str
    subreddit: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: float
    url: str
    permalink: str
    flair: Optional[str]
    is_self: bool
    selftext: str
    thumbnail: str
    domain: str
    collected_at: datetime

class RedditCollector:
    def __init__(self):
        self.client_id = "UiUfunumwIZ3w9C0yuz4sQ"
        self.client_secret = "4BsjMadK2fFaE-hDMxbziUOhFycujw"
        self.user_agent = "Forecast_the_False/1.0 by LameBoi1806"
        self.username = "LameBoi1806"
        self.password = "18M@rch06"
        self.redirect_uri = "http://localhost:8000/callback"

        self.reddit = None
        self.db_connection = None
        self.stats = {
            'posts_collected': 0,
            'errors': 0,
            'start_time': None
        }

        self._setup_database()
        self._connect_to_reddit()

    def _connect_to_reddit(self):
        try:
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
                redirect_uri=self.redirect_uri
            )
            logging.info(f"Connected to Reddit API. Read-only: {self.reddit.read_only}")
        except Exception as e:
            logging.error(f"Failed to connect to Reddit API: {e}")
            raise

    def _setup_database(self):
        try:
            self.db_connection = sqlite3.connect('reddit_data.db', check_same_thread=False)
            cursor = self.db_connection.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS posts (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    content TEXT,
                    author TEXT,
                    subreddit TEXT,
                    score INTEGER,
                    upvote_ratio REAL,
                    num_comments INTEGER,
                    created_utc REAL,
                    url TEXT,
                    permalink TEXT,
                    flair TEXT,
                    is_self BOOLEAN,
                    selftext TEXT,
                    thumbnail TEXT,
                    domain TEXT,
                    collected_at TIMESTAMP
                )
            ''')
            self.db_connection.commit()
            logging.info("Database setup completed")
        except Exception as e:
            logging.error(f"Database setup failed: {e}")
            raise

    def collect_subreddit_posts(self, subreddit_name: str, limit: int = 100, sort_by: str = "hot") -> List[RedditPost]:
        posts = []
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            logging.info(f"Collecting {limit} {sort_by} posts from r/{subreddit_name}")
            submissions = getattr(subreddit, sort_by)(limit=limit)
            for submission in submissions:
                try:
                    post = RedditPost(
                        id=submission.id,
                        title=submission.title,
                        content=submission.selftext if submission.is_self else "",
                        author=str(submission.author) if submission.author else "[deleted]",
                        subreddit=submission.subreddit.display_name,
                        score=submission.score,
                        upvote_ratio=submission.upvote_ratio,
                        num_comments=submission.num_comments,
                        created_utc=submission.created_utc,
                        url=submission.url,
                        permalink=submission.permalink,
                        flair=submission.link_flair_text,
                        is_self=submission.is_self,
                        selftext=submission.selftext,
                        thumbnail=submission.thumbnail,
                        domain=submission.domain,
                        collected_at=datetime.now()
                    )
                    posts.append(post)
                    self.stats['posts_collected'] += 1
                    self._store_post(post)
                    time.sleep(0.1)
                except Exception as e:
                    logging.error(f"Error processing post {submission.id}: {e}")
                    self.stats['errors'] += 1
                    continue
            logging.info(f"Collected {len(posts)} posts from r/{subreddit_name}")
        except Exception as e:
            logging.error(f"Error collecting from r/{subreddit_name}: {e}")
        return posts

    def search_posts(self, query: str, subreddit_names: List[str] = None, limit: int = 100, sort: str = "relevance") -> List[RedditPost]:
        posts = []
        try:
            if subreddit_names:
                for subreddit_name in subreddit_names:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    for submission in subreddit.search(query, sort=sort, limit=limit):
                        post = self._submission_to_post(submission)
                        posts.append(post)
                        self._store_post(post)
            else:
                for submission in self.reddit.subreddit("all").search(query, sort=sort, limit=limit):
                    post = self._submission_to_post(submission)
                    posts.append(post)
                    self._store_post(post)
            logging.info(f"Found {len(posts)} posts for query: '{query}'")
        except Exception as e:
            logging.error(f"Search error: {e}")
        return posts

    def _submission_to_post(self, submission) -> RedditPost:
        return RedditPost(
            id=submission.id,
            title=submission.title,
            content=submission.selftext if submission.is_self else "",
            author=str(submission.author) if submission.author else "[deleted]",
            subreddit=submission.subreddit.display_name,
            score=submission.score,
            upvote_ratio=submission.upvote_ratio,
            num_comments=submission.num_comments,
            created_utc=submission.created_utc,
            url=submission.url,
            permalink=submission.permalink,
            flair=submission.link_flair_text,
            is_self=submission.is_self,
            selftext=submission.selftext,
            thumbnail=submission.thumbnail,
            domain=submission.domain,
            collected_at=datetime.now()
        )

    def _store_post(self, post: RedditPost):
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO posts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                post.id, post.title, post.content, post.author, post.subreddit,
                post.score, post.upvote_ratio, post.num_comments, post.created_utc,
                post.url, post.permalink, post.flair, post.is_self, post.selftext,
                post.thumbnail, post.domain, post.collected_at
            ))
            self.db_connection.commit()
        except Exception as e:
            logging.error(f"Error storing post {post.id}: {e}")

    def export_to_csv(self, filename: str = None):
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"posts_reddit_data_{timestamp}.csv"

        output_dir = os.path.join("TruthGuard", "data", "raw_posts")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, filename)

        try:
            posts_df = pd.read_sql_query("SELECT * FROM posts", self.db_connection)
            posts_df.to_csv(output_path, index=False)
            logging.info(f"Data exported to {output_path}")
        except Exception as e:
            logging.error(f"Export error: {e}")

    def close(self):
        if self.db_connection:
            self.db_connection.close()
        logging.info("Reddit collector closed")
