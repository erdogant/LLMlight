"""Memory functionalities for LLMlight."""

from memvid import MemvidEncoder, MemvidRetriever
# from memvid.config import get_default_config as memvid_get_default_config

import logging
from typing import List, Union
import os
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)


#%%
class memvid_llm:
    """Video Memory."""

    def __init__(self, file_path: str = "llmlight_memory.mp4", config: dict = None):
        """Build QR code video and index from chunks with unified codec handling.

        Parameters
        ----------
        file_path : str
            Path to output video memory file.
        config : dict
            Dictionary containing configuration parameters.

        """
        self.file_path = None
        self.index_path = None
        # Return if file path is None
        if file_path is None:
            return

        # Get absolute path
        file_path = os.path.abspath(file_path)
        if os.path.isfile(file_path):
            logger.info(f'Initializing existing video memory: {file_path}')
        else:
            logger.info(f'Initializing new video memory: {file_path}')

        # Set memory path in self
        self._set_memory_path(file_path)
        # Initialize new encoder
        self.encoder = MemvidEncoder(config=config)
        # Update config from memvid Encoder
        self.config = self.encoder.config

    def _set_memory_path(self, file_path):
        """Set Memory paths."""
        # Get the absolute file_path
        file_path = os.path.abspath(file_path)
        # Get directory path (folder)
        directory = os.path.dirname(file_path)
        # full filename with extension
        filename = os.path.basename(file_path)
        # split name and extension
        name, extension = os.path.splitext(filename)
        # Make check
        if extension not in ['.mp4', '.avi', '.mkv']:
            logger.error(f"File path to memory should be of type: ['.mp4', '.avi', '.mkv']")
            raise ValueError(f"File path to memory should be of type: ['.mp4', '.avi', '.mkv']")

        # Store file names
        self.file_path = file_path
        self.index_path = os.path.join(directory, name) + '.json'

    def load(self):
        """Load video-memory files."""
        # Validate files exist and are readable
        if not hasattr(self, 'file_path') or not os.path.isfile(self.file_path):
            raise ValueError(f"Video file not found: {self.file_path}")
        if not os.path.isfile(self.index_path):
            raise ValueError(f"Index file not found: {self.index_path}")

        # Validate file integrity
        try:
            with open(self.index_path, 'r') as f:
                index_data = json.load(f)
            chunk_count = len(index_data.get('metadata', []))
        except Exception as e:
            raise ValueError(f"Index file corrupted: {e}")

        # Check video file size
        file_path = Path(self.file_path)
        video_size_mb = file_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Video file: {video_size_mb:.1f} MB, containing {chunk_count} chunks")
        logger.info(f"Loading memory:")
        logger.info(f"  📁 Video: {self.file_path}")
        logger.info(f"  📋 Index: {self.index_path}")

        # Loading
        self.retriever = MemvidRetriever(video_file=self.file_path, index_file=self.index_path, config=self.config)

    def add(self,
            text: Union[str, List[str]] = None,
            input_files: Union[str, List[str]] = None,
            dirpath: str = None,
            filetypes: List[str] = ['.pdf', '.txt', '.epub', '.md', '.doc', '.docx', '.rtf', '.html', '.htm'],
            chunk_size: int = 512,
            chunk_overlap: int = 100,
            overwrite=False):
        """Add chunks to memory.

        Parameters
        ----------
        input_files : (str, list)
            Path to file(s).

        """
        # Make checks
        if not hasattr(self, 'file_path') or not hasattr(self, 'encoder'):
            logger.error('Memory is not yet initialized. Use client.memory_init() first')
            raise AssertionError('Memory is not yet initialized. Use client.memory_init() first')
        if os.path.isfile(self.file_path) and not overwrite:
            logger.warning(f'Video memory already exists appending is not possible: {self.file_path}')
            return

        # Make lists
        if isinstance(text, str): text = [text]
        if isinstance(input_files, str): input_files = [input_files]

        if dirpath is not None and os.path.isdir(dirpath):
            if input_files is None: input_files = []
            for root, _, files in os.walk(dirpath):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in filetypes):
                        input_files.append(os.path.join(root, file))

        # Add text chunk to video-memory
        if text is not None:
            logger.info(f'Adding {len(text)} text chunks to memory.')
            self.encoder.add_chunks(text)

        # Run over all files
        if input_files is not None:
            logger.info(f'Adding {len(input_files)} into memory.')

            for file_path in input_files:
                if not os.path.isfile(file_path):
                    logger.warning(f"File not found: {file_path}")
                else:
                    # full filename with extension
                    filename = os.path.basename(file_path)
                    # split name and extension
                    name, ext = os.path.splitext(filename.lower())
                    # Add to encoder
                    if (ext == '.pdf') and (ext in filetypes):
                        self.encoder.add_pdf(file_path, chunk_size=chunk_size, overlap=chunk_overlap)
                    elif ext == '.epub' and (ext in filetypes):
                        self.encoder.add_epub(file_path, chunk_size=chunk_size, overlap=chunk_overlap)
                    elif ext == '.txt' and (ext in filetypes):
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                            self.encoder.add_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
                    elif ext in ['.html', '.htm']:
                        # Process HTML with BeautifulSoup
                        try:
                            from bs4 import BeautifulSoup
                        except ImportError:
                            print(f"Warning: BeautifulSoup not available for HTML processing. Skipping {file_path}")
                            continue

                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            soup = BeautifulSoup(f.read(), 'html.parser')
                            for script in soup(["script", "style"]):
                                script.decompose()
                            text = soup.get_text()
                            lines = (line.strip() for line in text.splitlines())
                            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                            clean_text = ' '.join(chunk for chunk in chunks if chunk)
                            if clean_text.strip():
                                self.encoder.add_text(clean_text, chunk_size=chunk_size, overlap=chunk_overlap)

                    elif ext in filetypes:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                            self.encoder.add_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
                    else:
                        continue
                    # Show message
                    logger.info(f'Added to memory: {filename}')

    def save(self,
             file_path: str = None,
             codec: str = 'mp4v',
             auto_build_docker: bool = True,
             allow_fallback: bool = True,
             overwrite: bool = False,
             show_progress: bool = True,
             ):
        """Build QR code video and index from chunks with unified codec handling.

        Parameters
        ----------
        file_path : str (default is the initialization memory-path)
            Path to output video memory file.
        codec : str, optional
            Video codec ('mp4v', 'h265', 'h264', etc.)
            'mp4v': Default
        auto_build_docker : bool (default: True)
            Whether to auto-build Docker if needed.
        allow_fallback : bool
            Whether to fall back to MP4V if advanced codec fails.
        show_progress : bool (default: True)

        """
        if not hasattr(self, 'file_path'):
            logger.error('Memory is not yet initialized. Use client.memory_init() first')
            raise AssertionError('Memory is not yet initialized. Use client.memory_init() first')

        self.build_stats = {}
        # Make checks
        if not hasattr(self, 'encoder') or len(self.encoder.chunks) == 0:
            logger.error('No chunks to encode. Use client.add_chunks() first')
            raise AssertionError('No chunks to encode. Use client.add_chunks() first')

        if file_path is not None:
            self._set_memory_path(file_path)

        # Check
        if os.path.isfile(self.file_path) and not overwrite:
            logger.warning(f'File already exists and not allowed to overwrite: {self.file_path}')
            return

        # Remove files when overwrite
        if overwrite:
            if os.path.isfile(self.file_path):
                logger.info(f'Video memory file is overwriten: {self.file_path}')
                os.remove(self.file_path)
            # Also remove the index file
            if os.path.isfile(self.index_path):
                os.remove(self.index_path)

        logger.info(f"🎬 Building video-memory: {self.file_path}")
        logger.info(f"📊 Total chunks to encode: {len(self.encoder.chunks)}")
        encoding_start = time.time()
        try:
            # Build the by passing all parameters to the building proces
            build_stats = self.encoder.build_video(output_file=self.file_path,
                                                   index_file=self.index_path,
                                                   codec=codec,
                                                   show_progress=show_progress,
                                                   auto_build_docker=auto_build_docker,
                                                   allow_fallback=allow_fallback,
                                                   )
        except Exception as e:
            error_str = str(e)
            if "is_trained" in error_str or "IndexIVFFlat" in error_str or "training" in error_str.lower():
                logger.warning(f"⚠️  FAISS IVF training failed: {e}")
                logger.warning(f"🔄 Auto-switching to Flat index for compatibility...")

                # Override config to use Flat index
                original_index_type = self.encoder.config["index"]["type"]
                self.encoder.config["index"]["type"] = "Flat"

                try:
                    # Recreate the index manager with Flat index
                    self.encoder._setup_index()
                    # Build the by passing all parameters to the building proces
                    build_stats = self.encoder.build_video(output_file=self.file_path,
                                                           index_file=self.index_path,
                                                           codec=codec,
                                                           show_progress=show_progress,
                                                           auto_build_docker=auto_build_docker,
                                                           allow_fallback=allow_fallback,
                                                           )

                    logger.info(f"✅ Successfully created memory using Flat index")
                    return build_stats
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback also failed: {fallback_error}")
                    raise
            else:
                raise

        # Time
        build_stats['encoding_time'] = time.time() - encoding_start

        # Clear all chunks of text from list because it will use the video memory file.
        self.encoder.chunks = []
        self.build_stats = build_stats
        logger.info('✅ Video Memory saved to disk.')

    def get_random_chunks(self, n=1000):
        if not hasattr(self, 'retriever'):
            logger.warning('No chunks to encode for null distribution. Use client.add_chunks() first')

        import random
        # Get all chunks
        chunks = list(map(lambda x: x.get('text'), self.retriever.index_manager.metadata))[0:n]

        # Step 1: Combine all words from every chunk
        combined_words = []
        for chunk in chunks:
            # Split by newline, space, or tab to handle multi-line strings properly
            combined_chunk = chunk.replace('\n', ' ').replace('\t', ' ')
            words = combined_chunk.split()
            combined_words.extend(words)

        # Step 2: Create new lists with random sets of words
        new_chunks = [[] for _ in range(len(chunks))]
        n_chunks = len(new_chunks)
        for word in combined_words:
            # Choose a random index to place the current word into one of the chunks
            chunk_index = random.randint(0, n_chunks - 1)
            new_chunks[chunk_index].append(word)

        chunk_strings = [[] for _ in range(n_chunks)]
        for i, lst in enumerate(new_chunks):
            chunk_strings[i] = ' '.join(lst)

        return chunk_strings


    def show_stats(self):
        # Enhanced statistics
        if not hasattr(self, 'build_stats'):
            logger.warning('No video-memory statistics found.')
            return

        build_stats = self.build_stats
        video_path = Path(self.file_path)
        encoding_time = build_stats.get('encoding_time')

        logger.info(f"\n🎉 Memory created successfully!")
        logger.info(f"  📁 Video: {self.file_path}")
        logger.info(f"  📋 Index: {self.index_path}")
        logger.info(f"  📊 Chunks: {build_stats.get('total_chunks', 'unknown')}")
        logger.info(f"  🎞️  Frames: {build_stats.get('total_frames', 'unknown')}")
        logger.info(f"  📏 Video size: {video_path.stat().st_size / (1024 * 1024):.1f} MB")
        logger.info(f"  ⏱️  Encoding time: {encoding_time:.2f} seconds" if encoding_time is not None else "  ⏱️  Encoding time: unknown")
