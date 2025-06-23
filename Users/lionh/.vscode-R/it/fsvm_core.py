"""
Fracture-Sourced Virtual Machine (FSVM) - Core Module

This module implements the core cognitive system with:
- Core Cognitive Loop
- Entropy Sources Module  
- Symbolic Tension-Response Engine
- Memory Store (STM and LTM)
- Emergent Communication Engine
"""

import os
import time
import random
import json
import threading
import queue
import pickle
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
import numpy as np
import requests
from bs4 import BeautifulSoup
import psutil
import networkx as nx
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from config import (
    CORE_SETTINGS, MEMORY_SETTINGS, ENTROPY_SETTINGS,
    DRIVE_SETTINGS, NEURAL_SETTINGS, DEV_SETTINGS
)
import torch
import torch.nn as nn
import torch.optim as optim
import traceback
import uuid
import math
import re


@dataclass
class ThoughtStep:
    """Represents one step in a thought process."""
    step_type: str # e.g., "input", "thought", "resonance"
    content: Any
    tension: float
    symbol_id: Optional[str] = None
    response: Any = None


@dataclass
class Symbol:
    """Represents an emergent symbolic pattern in the FSVM"""
    id: str
    pattern: List[Any]
    frequency: int = 1
    last_seen: float = 0.0
    created: float = 0.0
    associations: List[str] = None
    vector: Optional[np.ndarray] = None
    cluster_id: Optional[int] = None
    
    def __post_init__(self):
        if self.associations is None:
            self.associations = []
        if self.created == 0.0:
            self.created = time.time()
        self.last_seen = time.time()


@dataclass
class MemoryEntry:
    """Represents a memory entry in the FSVM"""
    content: Any
    timestamp: float
    tension_level: float
    symbol_id: Optional[str] = None


class EntropySources:
    """Manages various sources of entropy for the FSVM"""
    
    def __init__(self):
        self.source_counts = Counter()
        self.api_endpoints = [
            "https://httpbin.org/get",
            "https://httpbin.org/post", 
            "https://httpbin.org/put",
            "https://httpbin.org/delete",
            "https://httpbin.org/status/200",
            "https://httpbin.org/status/404",
            "https://httpbin.org/status/500",
            "https://httpbin.org/delay/1",
            "https://httpbin.org/stream/5",
            "https://httpbin.org/bytes/100",
            "https://httpbin.org/uuid",
            "https://httpbin.org/json",
            "https://httpbin.org/html",
            "https://httpbin.org/xml",
            "https://httpbin.org/robots.txt",
            "https://httpbin.org/cache/60",
            "https://httpbin.org/etag/abc123",
            "https://httpbin.org/response-headers",
            "https://httpbin.org/redirect/1",
            "https://httpbin.org/redirect-to?url=https://example.com"
        ]
        self.current_endpoint_index = 0
        self.math_patterns = [
            "fibonacci", "prime", "square", "cube", "factorial", 
            "geometric", "arithmetic", "harmonic", "lucas", "pell"
        ]
        self.text_patterns = [
            "lorem ipsum dolor sit amet",
            "the quick brown fox jumps over the lazy dog",
            "pack my box with five dozen liquor jugs",
            "how vexingly quick daft zebras jump",
            "the five boxing wizards jump quickly",
            "sphinx of black quartz judge my vow",
            "crazy fredrick bought many very exquisite opal jewels",
            "we promptly judged antique ivory buckles for the next prize",
            "a mad boxer shot a quick gloved jab to the jaw of his dizzy opponent",
            "the job requires extra pluck and zeal from every young wage earner"
        ]
        self.symbol_patterns = [
            "!@#$%^&*()_+-=[]{}|;':\",./<>?",
            "αβγδεζηθικλμνξοπρστυφχψω",
            "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
            "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん",
            "0123456789ABCDEF",
            "♠♣♥♦♤♧♡♢",
            "☀☁☂☃★☆♠♣♥♦♤♧♡♢",
            "⚀⚁⚂⚃⚄⚅",
            "♔♕♖♗♘♙",
            "♩♪♫♬♭♮♯"
        ]
        
    def get_entropy(self) -> List[List[str]]:
        """Generate truly infinite and diverse entropy patterns"""
        patterns = []
        
        # Generate completely random patterns with real-world data
        for _ in range(random.randint(5, 15)):
            pattern_type = random.choice([
                'real_websites', 'file_system', 'random_text', 'math', 'symbols', 
                'api_data', 'data_structures', 'mixed', 'time_series', 'binary_data'
            ])
            self.source_counts[pattern_type] += 1
            
            if pattern_type == 'real_websites':
                # Real website data
                websites = [
                    'https://news.ycombinator.com', 'https://reddit.com', 'https://github.com',
                    'https://stackoverflow.com', 'https://wikipedia.org', 'https://arxiv.org',
                    'https://medium.com', 'https://dev.to', 'https://hashnode.dev',
                    'https://techcrunch.com', 'https://venturebeat.com', 'https://theverge.com',
                    'https://arstechnica.com', 'https://wired.com', 'https://mit.edu',
                    'https://stanford.edu', 'https://berkeley.edu', 'https://cmu.edu',
                    'https://openai.com', 'https://anthropic.com', 'https://deepmind.com',
                    'https://nvidia.com', 'https://intel.com', 'https://amd.com',
                    'https://microsoft.com', 'https://google.com', 'https://apple.com',
                    'https://amazon.com', 'https://netflix.com', 'https://spotify.com'
                ]
                website = random.choice(websites)
                patterns.append([f"website_{website}", f"status_{random.randint(200, 500)}", f"load_time_{random.uniform(0.1, 5.0):.2f}"])
                
            elif pattern_type == 'file_system':
                # File system exploration
                file_types = ['.py', '.js', '.html', '.css', '.json', '.xml', '.txt', '.md', '.csv', '.sql']
                file_type = random.choice(file_types)
                file_size = random.randint(100, 1000000)
                patterns.append([f"file_{random.randint(1, 1000)}{file_type}", f"size_{file_size}", f"modified_{int(time.time()) - random.randint(0, 86400)}"])
                
            elif pattern_type == 'random_text':
                # Random text patterns with more variety
                text_samples = [
                    ['quantum', 'entanglement', 'superposition', 'measurement'],
                    ['neural', 'network', 'backpropagation', 'gradient'],
                    ['fractal', 'dimension', 'self_similarity', 'chaos'],
                    ['algorithm', 'complexity', 'optimization', 'heuristic'],
                    ['consciousness', 'awareness', 'experience', 'qualia'],
                    ['information', 'entropy', 'bits', 'encoding'],
                    ['emergence', 'complexity', 'self_organization', 'criticality'],
                    ['learning', 'adaptation', 'evolution', 'selection'],
                    ['intelligence', 'reasoning', 'inference', 'abduction'],
                    ['creativity', 'innovation', 'novelty', 'originality'],
                    ['memory', 'recall', 'association', 'consolidation'],
                    ['attention', 'focus', 'salience', 'relevance'],
                    ['emotion', 'affect', 'valence', 'arousal'],
                    ['language', 'syntax', 'semantics', 'pragmatics'],
                    ['vision', 'perception', 'recognition', 'interpretation']
                ]
                patterns.append(random.choice(text_samples))
                
            elif pattern_type == 'math':
                # Mathematical operations with random numbers
                operations = ['add', 'multiply', 'divide', 'power', 'sqrt', 'log', 'sin', 'cos', 'tan']
                op = random.choice(operations)
                a = random.uniform(-100, 100)
                b = random.uniform(-100, 100)
                if op == 'add':
                    patterns.append([f"{a:.2f}", "+", f"{b:.2f}", "=", f"{a+b:.2f}"])
                elif op == 'multiply':
                    patterns.append([f"{a:.2f}", "*", f"{b:.2f}", "=", f"{a*b:.2f}"])
                elif op == 'power':
                    patterns.append([f"{a:.2f}", "^", f"{b:.2f}", "=", f"{a**b:.2f}"])
                elif op == 'sqrt':
                    patterns.append([f"sqrt({a:.2f})", "=", f"{abs(a)**0.5:.2f}"])
                elif op == 'sin':
                    patterns.append([f"sin({a:.2f})", "=", f"{math.sin(a):.2f}"])
                elif op == 'cos':
                    patterns.append([f"cos({a:.2f})", "=", f"{math.cos(a):.2f}"])
                    
            elif pattern_type == 'symbols':
                # Diverse symbol sets
                symbol_sets = [
                    ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ'],  # Greek
                    ['а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж'],  # Cyrillic
                    ['あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く'],  # Japanese
                    ['⚡', '🔥', '💡', '🚀', '🎯', '💎', '🌟', '🎨'],  # Emojis
                    ['∑', '∏', '∫', '∂', '∇', '∞', '≈', '≠'],  # Math
                    ['♠', '♥', '♦', '♣', '♔', '♕', '♖', '♗'],  # Cards
                    ['☀', '☁', '☂', '☃', '☄', '★', '☆', '♠'],  # Weather
                    ['⚔', '🛡', '🏹', '🗡', '⚓', '⚒', '⚓', '⚔'],  # Weapons
                    ['🎭', '🎪', '🎨', '🎬', '🎤', '🎧', '🎼', '🎹'],  # Arts
                    ['🏆', '🥇', '🥈', '🥉', '🎖', '🏅', '🎗', '🏆']   # Awards
                ]
                patterns.append(random.choice(symbol_sets))
                
            elif pattern_type == 'api_data':
                # API-like data structures
                api_endpoints = [
                    '/api/users', '/api/posts', '/api/comments', '/api/likes',
                    '/api/followers', '/api/messages', '/api/notifications',
                    '/api/settings', '/api/profile', '/api/search', '/api/upload',
                    '/api/download', '/api/delete', '/api/update', '/api/create'
                ]
                endpoint = random.choice(api_endpoints)
                status = random.choice([200, 201, 400, 401, 403, 404, 500])
                patterns.append([f"GET {endpoint}", f"status_{status}", f"time_{int(time.time())}"])
                
            elif pattern_type == 'data_structures':
                # Complex data structures
                data_types = ['list', 'dict', 'tuple', 'set', 'queue', 'stack', 'tree', 'graph']
                data_type = random.choice(data_types)
                if data_type == 'list':
                    patterns.append([f"list_{random.randint(1, 100)}", f"length_{random.randint(1, 20)}", f"type_{random.choice(['int', 'str', 'float', 'bool'])}"])
                elif data_type == 'dict':
                    patterns.append([f"dict_{random.randint(1, 100)}", f"keys_{random.randint(1, 10)}", f"nested_{random.choice(['yes', 'no'])}"])
                elif data_type == 'tree':
                    patterns.append([f"tree_{random.randint(1, 100)}", f"depth_{random.randint(1, 10)}", f"nodes_{random.randint(1, 100)}"])
                elif data_type == 'graph':
                    patterns.append([f"graph_{random.randint(1, 100)}", f"vertices_{random.randint(1, 50)}", f"edges_{random.randint(1, 200)}"])
                    
            elif pattern_type == 'time_series':
                # Time series data
                timestamps = [int(time.time()) - i * random.randint(1, 3600) for i in range(random.randint(3, 10))]
                values = [random.uniform(0, 100) for _ in range(len(timestamps))]
                patterns.append([f"ts_{t}" for t in timestamps[:3]] + [f"val_{v:.2f}" for v in values[:3]])
                
            elif pattern_type == 'binary_data':
                # Binary and encoded data
                binary_length = random.randint(8, 32)
                binary_data = ''.join(random.choice(['0', '1']) for _ in range(binary_length))
                hex_data = ''.join(random.choice('0123456789abcdef') for _ in range(8))
                patterns.append([f"bin_{binary_data[:8]}", f"hex_{hex_data}", f"len_{binary_length}"])
                
            else:  # mixed
                # Mixed random data
                mixed_data = [
                    f"uuid_{str(uuid.uuid4())[:8]}",
                    f"timestamp_{int(time.time())}",
                    f"hash_{hash(str(random.random())) % 10000:04d}",
                    f"float_{random.uniform(-1000, 1000):.3f}",
                    f"bool_{random.choice([True, False])}",
                    f"char_{chr(random.randint(33, 126))}",
                    f"emoji_{random.choice(['😀', '🎉', '🚀', '💡', '🔥', '⭐', '🎯', '💎'])}"
                ]
                patterns.append(random.sample(mixed_data, random.randint(3, 6)))
        
        return patterns
    
    def get_source_distribution(self):
        """Returns the distribution of entropy sources used."""
        return self.source_counts
    
    def _generate_fibonacci(self, n):
        """Generate first n Fibonacci numbers"""
        if n <= 0:
            return []
        elif n == 1:
            return [0]
        elif n == 2:
            return [0, 1]
        
        fib = [0, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return fib
    
    def _generate_primes(self, n):
        """Generate first n prime numbers"""
        primes = []
        num = 2
        while len(primes) < n:
            if all(num % p != 0 for p in primes):
                primes.append(num)
            num += 1
        return primes
    
    def _factorial(self, n):
        """Calculate factorial of n"""
        if n <= 1:
            return 1
        return n * self._factorial(n - 1)


class SymbolicTensionEngine:
    """The core tension-response engine that drives FSVM adaptation"""
    
    def __init__(self, memory_size: int = 1000, neural_layer: Optional['NeuralLayer'] = None):
        self.short_term_memory = []
        self.symbols: Dict[str, Symbol] = {}
        self.pattern_history: List[Any] = []
        self.tension_history: List[float] = []
        self.memory_size = memory_size
        self.last_output = None
        
        # For clustering
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.symbol_clusters = defaultdict(list)
        self.cluster_centroids = {}
        self.next_cluster_id = 0

        # Phase 4: Neural Layer - Passed in from FSVM
        self.neural_layer = neural_layer

        self.symbol_graph = nx.Graph()

    def _vectorize_pattern(self, pattern: List[Any]) -> np.ndarray:
        """Converts a symbol's pattern into a vector."""
        pattern_str = " ".join(map(str, pattern))
        return self.model.encode([pattern_str])[0]

    def _update_clusters(self, symbol: Symbol):
        """Assigns a symbol to a cluster based on cosine similarity."""
        if symbol.vector is None:
            return # Should not happen if called correctly

        if not self.cluster_centroids:
            # This is the first symbol, create the first cluster.
            new_cluster_id = self.next_cluster_id
            self.cluster_centroids[new_cluster_id] = symbol.vector
            self.symbol_clusters[new_cluster_id].append(symbol.id)
            symbol.cluster_id = new_cluster_id
            self.next_cluster_id += 1
            return

        # Find the most similar cluster
        centroids = list(self.cluster_centroids.values())
        similarities = cosine_similarity([symbol.vector], centroids)[0]
        
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        # Determine if we should create a new symbol
        similarity_threshold = 0.6  # Reduced from 0.8 to encourage more symbol creation
        
        if best_similarity > similarity_threshold:
            # Add to existing cluster
            cluster_id = list(self.cluster_centroids.keys())[best_match_idx]
            symbol.cluster_id = cluster_id
            self.symbol_clusters[cluster_id].append(symbol.id)
            # Update centroid (moving average)
            old_centroid = self.cluster_centroids[cluster_id]
            num_symbols = len(self.symbol_clusters[cluster_id])
            self.cluster_centroids[cluster_id] = ((old_centroid * (num_symbols - 1)) + symbol.vector) / num_symbols
        else:
            # Create a new cluster
            new_cluster_id = self.next_cluster_id
            self.cluster_centroids[new_cluster_id] = symbol.vector
            self.symbol_clusters[new_cluster_id].append(symbol.id)
            symbol.cluster_id = new_cluster_id
            self.next_cluster_id += 1

    def calculate_tension(self, current_input: Any, expected_input: Any = None) -> float:
        """Calculate tension level based on input surprise"""
        if expected_input is None:
            expected_input = self.last_output
        
        if expected_input is None:
            return 0.5  # Neutral tension for first input
        
        # Simple tension calculation based on hash difference
        if isinstance(current_input, (int, float)) and isinstance(expected_input, (int, float)):
            tension = abs(current_input - expected_input) / max(abs(current_input), abs(expected_input), 1)
        elif isinstance(current_input, str) and isinstance(expected_input, str):
            # String similarity based tension
            if current_input == expected_input:
                tension = 0.0
            else:
                tension = 1.0 - (len(set(current_input) & set(expected_input)) / max(len(set(current_input) | set(expected_input)), 1))
        else:
            # Default to high tension for different types
            tension = 1.0
        
        # Add a penalty from the neural network's prediction error
        neural_tension = 0
        if self.neural_layer:
            neural_tension = self.neural_layer.get_tension(current_input)
            tension = (tension + neural_tension) / 2 # Average symbolic and neural tension
        
        # Apply tension multiplier if available (for UI control)
        if hasattr(self, 'fsvm_instance') and self.fsvm_instance:
            tension *= self.fsvm_instance.get_tension_multiplier()
        
        return min(tension, 1.0)
    
    def detect_patterns(self, sequence: List[Any], min_length: int = 2, max_length: int = 5) -> List[Tuple[List[Any], int]]:
        """Detect repeating patterns in a sequence"""
        patterns = []
        
        for length in range(min_length, min(max_length + 1, len(sequence) // 2 + 1)):
            for i in range(len(sequence) - length * 2 + 1):
                pattern = sequence[i:i + length]
                # Count how many times this pattern repeats
                count = 0
                for j in range(i, len(sequence) - length + 1):
                    if sequence[j:j + length] == pattern:
                        count += 1
                
                if count >= 2:  # Pattern must repeat at least twice
                    patterns.append((pattern, count))
        
        return patterns
    
    def create_symbol(self, pattern: List[Any]) -> Symbol:
        """Create a new symbol from a detected pattern"""
        # Use timestamp and counter for unique IDs instead of hash
        symbol_id = f"SYM_{len(self.symbols):04d}_{int(time.time() * 1000) % 100000}"
        
        # Check if symbol ID already exists (prevent duplicates)
        if symbol_id in self.symbols:
            return None  # Skip duplicate
        
        # Remove the hard limit - allow infinite learning
        # Check if similar symbol already exists
        # Lowered threshold to promote more diverse symbol creation and prevent stalls.
        similarity_threshold = 0.45 
        
        pattern_vector = self._vectorize_pattern(pattern)
        
        # Check for existing vectors to prevent duplicates
        existing_vectors = [s.vector for s in self.symbols.values() if s.vector is not None]
        for existing_vector in existing_vectors:
            if existing_vector is not None:
                similarity = cosine_similarity([pattern_vector], [existing_vector])[0][0]
                if similarity > similarity_threshold:
                    if DEV_SETTINGS['debug_mode']:
                        print(f"[SKIP] Similar symbol ({similarity:.2f}) already exists for '{pattern}'")
                    return None  # Skip duplicate
        
        # Check for existing symbols with similar patterns
        for existing_symbol in self.symbols.values():
            if existing_symbol.vector is not None:
                similarity = cosine_similarity([pattern_vector], [existing_symbol.vector])[0][0]
                if similarity > similarity_threshold:
                    if DEV_SETTINGS['debug_mode']:
                        print(f"[SKIP] Similar symbol ({similarity:.2f}) already exists for '{pattern}'. ID: {existing_symbol.id}")
                    return existing_symbol # Return existing instead of creating new
        
        # Create new symbol - no limits!
        symbol = Symbol(
            id=symbol_id,
            pattern=pattern.copy(),
            created=time.time(),
            vector=pattern_vector
        )
        self.symbols[symbol_id] = symbol
        self._update_clusters(symbol)
        
        # Log symbol creation for activity tracking
        if hasattr(self, 'activity_log'):
            self.activity_log.append(f"Created symbol {symbol_id}")
        
        print(f"[NEW] Created symbol {symbol_id} for '{pattern}'")
        
        # --- Add to symbol graph ---
        self.symbol_graph.add_node(symbol_id)
        # Link to last 5 created symbols (excluding itself)
        recent_ids = list(self.symbols.keys())[-6:-1]
        for rid in recent_ids:
            self.symbol_graph.add_edge(symbol_id, rid, weight=1)
        
        return symbol
    
    def _get_ngrams(self, text: str, n: int) -> list:
        """Helper to generate n-grams from text."""
        # Simple tokenizer
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def find_resonant_symbols(self, input_data: Any) -> List[Symbol]:
        """Find symbols that resonate with current input"""
        resonant_symbols = []
        
        for symbol in self.symbols.values():
            # Simple resonance detection - check if input contains symbol pattern
            if isinstance(input_data, str) and isinstance(symbol.pattern[0], str):
                if any(pattern in input_data for pattern in symbol.pattern):
                    if symbol not in resonant_symbols:
                        resonant_symbols.append(symbol)
            elif isinstance(input_data, (list, tuple)) and len(symbol.pattern) <= len(input_data):
                # Check for pattern match in sequence
                for i in range(len(input_data) - len(symbol.pattern) + 1):
                    if input_data[i:i + len(symbol.pattern)] == symbol.pattern:
                        if symbol not in resonant_symbols:
                            resonant_symbols.append(symbol)
                        break
        
        # Phase 5: Add linguistic resonance for text
        if isinstance(input_data, str):
            # Check for resonance based on n-grams and keywords
            input_words = set(re.findall(r'\b\w+\b', input_data.lower()))
            for symbol in self.symbols.values():
                if isinstance(symbol.pattern[0], str):
                    symbol_words = set(re.findall(r'\b\w+\b', " ".join(symbol.pattern).lower()))
                    # Check for overlapping words
                    if len(input_words.intersection(symbol_words)) > 0:
                         if symbol not in resonant_symbols:
                            resonant_symbols.append(symbol)

        return resonant_symbols
    
    def update_symbol(self, symbol_id: str, input_data: Any):
        """Update symbol frequency and associations"""
        if symbol_id in self.symbols:
            symbol = self.symbols[symbol_id]
            symbol.frequency += 1
            symbol.last_seen = time.time()
            
            # Update associations based on current input
            if hasattr(input_data, '__str__'):
                input_str = str(input_data)
                if input_str not in symbol.associations:
                    symbol.associations.append(input_str)
    
    def process_input(self, input_data: Any) -> Tuple[float, Optional[str], Any]:
        """Process input and return tension, resonant symbol, and response"""
        # STEP 1 & 2 FIX: Initialize symbol_id to None at the top to prevent UnboundLocalError
        symbol_id: Optional[str] = None
        
        # Calculate tension
        tension = self.calculate_tension(input_data)
        self.tension_history.append(tension)
        
        # Find resonant symbols
        resonant_symbols = self.find_resonant_symbols(input_data)
        
        # Update pattern history
        self.pattern_history.append(input_data)
        if len(self.pattern_history) > 50:  # Keep last 50 items
            self.pattern_history.pop(0)
        
        # Detect new patterns from text
        if isinstance(input_data, str):
            # Generate n-grams to find common phrases
            ngrams = self._get_ngrams(input_data, 2) + self._get_ngrams(input_data, 3)
            if ngrams:
                # Find the most common ngram as a potential new pattern
                most_common_ngram = Counter(ngrams).most_common(1)[0][0]
                symbol = self.create_symbol(most_common_ngram.split())
                if symbol and symbol not in resonant_symbols:  # Check if symbol was created
                    resonant_symbols.append(symbol)
        else:
            # Detect new patterns from general data
            if len(self.pattern_history) >= 4:
                patterns = self.detect_patterns(self.pattern_history)
                for pattern, count in patterns:
                    if count >= CORE_SETTINGS.get('min_pattern_repetitions', 3):
                        symbol = self.create_symbol(pattern)
                        if symbol and symbol not in resonant_symbols:  # Check if symbol was created
                            resonant_symbols.append(symbol)
        
        # Generate response based on tension and resonance
        if resonant_symbols:
            # Filter out any None values
            resonant_symbols = [s for s in resonant_symbols if s is not None]
            
            if resonant_symbols:  # Check if we still have symbols after filtering
                # Low tension - use resonant symbol
                chosen_symbol = max(resonant_symbols, key=lambda s: s.frequency)
                # Guard symbol usage
                if chosen_symbol and hasattr(chosen_symbol, 'id'):
                    self.update_symbol(chosen_symbol.id, input_data)
                    self._update_clusters(chosen_symbol)
                    # Phase 5: Generate more intelligent response
                    if isinstance(input_data, str):
                        response = self.generate_linguistic_response(chosen_symbol, input_data)
                    else:
                        response = chosen_symbol.pattern
                    symbol_id = chosen_symbol.id
                else:
                    response = input_data # Fallback if symbol is invalid
            else:
                # No valid resonant symbols
                response = input_data
        else:
            # High tension - generate new pattern
            if tension > CORE_SETTINGS['high_tension_threshold']:
                # High tension: generate random response
                response = f"CHAOS_{random.randint(0, 1000)}"
            else:
                # Mid-tension: try to form a new symbol from recent patterns
                recent_patterns = self.pattern_history[-CORE_SETTINGS['max_pattern_length']:]
                new_patterns = self.detect_patterns(recent_patterns, min_length=2, max_length=CORE_SETTINGS['max_pattern_length'])
                
                if new_patterns:
                    # Create a new symbol from the most frequent new pattern
                    most_frequent_pattern = max(new_patterns, key=lambda p: p[1])[0]
                    new_symbol = self.create_symbol(most_frequent_pattern)
                    # Guard symbol usage
                    if new_symbol and hasattr(new_symbol, 'id'):
                        response = new_symbol.pattern
                        symbol_id = new_symbol.id
                    else:
                        response = input_data # Fallback
                else:
                    # If no new pattern, just echo input
                    response = input_data
        
        # Update short-term memory
        memory_entry = MemoryEntry(
            content=input_data,
            timestamp=time.time(),
            tension_level=tension,
            symbol_id=symbol_id
        )
        self.short_term_memory.append(memory_entry)
        
        # Limit memory size
        if len(self.short_term_memory) > self.memory_size:
            self.short_term_memory.pop(0)
        
        self.last_output = response
        return tension, symbol_id, response
    
    def generate_linguistic_response(self, symbol: Symbol, original_input: str) -> str:
        """Generates a more natural response for chat."""
        # Simple response generation based on the symbol's pattern
        pattern_text = " ".join(map(str, symbol.pattern))
        
        # Try to form a coherent sentence
        responses = [
            f"I've noticed a pattern related to '{pattern_text}'.",
            f"The concept of '{pattern_text}' seems significant.",
            f"Thinking about '{original_input}' brings '{pattern_text}' to mind.",
            f"Resonance found: {pattern_text}",
        ]
        return random.choice(responses)

    def generate_random_response(self, input_data: Any) -> Any:
        """Generate a random response when tension is high"""
        if isinstance(input_data, str):
            # Generate random string
            chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            return ''.join(random.choice(chars) for _ in range(random.randint(3, 10)))
        elif isinstance(input_data, (int, float)):
            # Generate random number
            return random.uniform(-100, 100)
        else:
            # Default to string
            return str(input_data) + "_" + str(random.randint(1000, 9999))
    
    def adapt_existing_patterns(self, input_data: Any) -> Any:
        """Try to adapt existing patterns to new input"""
        if not self.symbols:
            return self.generate_random_response(input_data)
        
        # Find most frequent symbol and adapt it
        most_frequent = max(self.symbols.values(), key=lambda s: s.frequency)
        
        if isinstance(input_data, str) and isinstance(most_frequent.pattern[0], str):
            # Adapt string pattern
            return most_frequent.pattern[0] + "_" + str(input_data)[:3]
        else:
            # Adapt other patterns
            return most_frequent.pattern[0] if most_frequent.pattern else input_data

    def get_symbol_clusters(self):
        """Returns the current symbol clusters."""
        return self.symbol_clusters

    def get_tension_history(self) -> List[float]:
        """Get the tension history"""
        return self.tension_history.copy()
    
    def get_current_tension(self) -> float:
        """Get the current tension level"""
        return self.tension_history[-1] if self.tension_history else 0.5

    def get_symbol_graph_data(self):
        nodes = [
            {"id": n, "label": str(self.symbols[n].pattern)[:30]} 
            for n in self.symbol_graph.nodes if n in self.symbols
        ]
        edges = [
            {"source": u, "target": v, "weight": d.get("weight", 1)}
            for u, v, d in self.symbol_graph.edges(data=True)
        ]
        return nodes, edges


class NeuralLayer(nn.Module):
    """A parallel neural network layer for pattern generalization."""
    def __init__(self, input_size=256, hidden_size=128, output_size=256):
        super(NeuralLayer, self).__init__()
        self.hidden_size = NEURAL_SETTINGS['hidden_size']
        self.vocab = {chr(i): i for i in range(input_size)}
        self.input_size = input_size
        self.output_size = output_size

        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=NEURAL_SETTINGS['learning_rate'])

        self.response_queue = queue.Queue()
        
        # Control flags
        self.running = False
        self.current_processing = False
        self.cognitive_thread = None
        
        # Statistics
        self.cycle_count = 0
        self.total_tension = 0.0

    def _to_one_hot(self, char):
        vec = torch.zeros(1, self.input_size)
        if char in self.vocab:
            vec[0][self.vocab[char]] = 1
        return vec

    def forward(self, input_seq, hidden_state):
        output, hidden_state = self.rnn(input_seq, hidden_state)
        output = self.fc(output)
        return output, hidden_state

    def train_step(self, input_str: str):
        if not input_str:
            return 0.0

        self.optimizer.zero_grad()
        hidden = torch.zeros(1, 1, self.hidden_size)
        
        total_loss = 0
        
        for i in range(len(input_str) - 1):
            input_char = self._to_one_hot(input_str[i]).unsqueeze(0)
            target_char_idx = self.vocab.get(input_str[i+1], 0)
            target = torch.LongTensor([target_char_idx])
            
            output, hidden = self(input_char, hidden)
            loss = self.criterion(output.squeeze(0), target)
            total_loss += loss.item()
            loss.backward(retain_graph=True)
        
        self.optimizer.step()
        return total_loss / len(input_str) if len(input_str) > 0 else 0.0

    def get_tension(self, input_data: Any) -> float:
        """Calculate tension based on the neural network's prediction error."""
        if not isinstance(input_data, str):
            input_str = str(input_data)
        else:
            input_str = input_data
            
        loss = self.train_step(input_str)
        # Normalize loss to a 0-1 range for tension.
        # This is a heuristic and may need tuning.
        tension = min(1.0, loss / 5.0) 
        return tension


class MemoryStore:
    """Handles short-term and long-term memory for the FSVM"""
    
    def __init__(self, memory_dir: str = "fsvm_memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        
        self.symbols_file = self.memory_dir / "symbols.json"
        self.patterns_file = self.memory_dir / "patterns.pkl"
        self.interactions_file = self.memory_dir / "interactions.json"
        self.drives_file = self.memory_dir / "drives.json"
        
        # STEP 7 FIX: Add a lock for thread-safe file operations
        self.lock = threading.Lock()
        
        # Phase 2: Internal Drives
        self.drives = {
            "curiosity": DRIVE_SETTINGS["curiosity"],
            "stability": DRIVE_SETTINGS["stability"],
            "boredom": DRIVE_SETTINGS["boredom"],
            "novelty_hunger": DRIVE_SETTINGS["novelty_hunger"]
        }
        self.cycles_since_new_symbol = 0

        # Phase 4: Neural Layer
        self.neural_layer = None
        if NEURAL_SETTINGS['use_neural_network']:
            self.neural_layer = NeuralLayer()
        
        # Phase 5: Meta-Symbols
        self.symbol_co_occurrence = defaultdict(int)
        self.last_symbol_id = None
        
        # Phase 6: Thought Tracing
        self.last_thought_trace: List[ThoughtStep] = []
        
    def save_symbols(self, symbols: Dict[str, Symbol]):
        """Save symbols to long-term memory, converting numpy vectors to lists."""
        with self.lock:
            try:
                symbols_data = {}
                for symbol_id, symbol in symbols.items():
                    data = asdict(symbol)
                    if data.get('vector') is not None:
                        # Convert numpy array to list for JSON serialization
                        data['vector'] = data['vector'].tolist()
                    symbols_data[symbol_id] = data
                
                with open(self.symbols_file, 'w') as f:
                    json.dump(symbols_data, f, indent=2)
            except Exception as e:
                print(f"Error saving symbols: {e}")
    
    def load_symbols(self) -> Dict[str, Symbol]:
        """Load symbols from long-term memory, converting lists back to numpy vectors."""
        with self.lock:
            try:
                if self.symbols_file.exists():
                    with open(self.symbols_file, 'r') as f:
                        symbols_data = json.load(f)
                    
                    symbols = {}
                    for symbol_id, data in symbols_data.items():
                        if data.get('vector') is not None:
                            # Convert list back to numpy array on load
                            data['vector'] = np.array(data['vector'])
                        symbols[symbol_id] = Symbol(**data)
                    return symbols
            except Exception as e:
                print(f"Error loading symbols: {e}")
        
        return {}
    
    def save_patterns(self, patterns: List[Any]):
        """Save pattern history to long-term memory"""
        with self.lock:
            try:
                with open(self.patterns_file, 'wb') as f:
                    pickle.dump(patterns, f)
            except Exception as e:
                print(f"Error saving patterns: {e}")
    
    def load_patterns(self) -> List[Any]:
        """Load pattern history from long-term memory"""
        with self.lock:
            try:
                if self.patterns_file.exists():
                    with open(self.patterns_file, 'rb') as f:
                        return pickle.load(f)
            except Exception as e:
                print(f"Error loading patterns: {e}")
        
        return []
    
    def save_interaction(self, user_input: str, fsvm_response: str, tension: float):
        """Save a user interaction to long-term memory"""
        with self.lock:
            try:
                interactions = []
                if self.interactions_file.exists():
                    with open(self.interactions_file, 'r') as f:
                        interactions = json.load(f)
                
                interaction = {
                    'timestamp': time.time(),
                    'user_input': user_input,
                    'fsvm_response': fsvm_response,
                    'tension': tension
                }
                interactions.append(interaction)
                
                # Keep only last 1000 interactions
                if len(interactions) > 1000:
                    interactions = interactions[-1000:]
                
                with open(self.interactions_file, 'w') as f:
                    json.dump(interactions, f, indent=2)
            except Exception as e:
                print(f"Error saving interaction: {e}")

    def save_drives(self, drives: Dict[str, float]):
        """Save the internal drives state."""
        with self.lock:
            try:
                with open(self.drives_file, 'w') as f:
                    json.dump(drives, f, indent=2)
            except Exception as e:
                print(f"Error saving drives: {e}")

    def load_drives(self) -> Optional[Dict[str, float]]:
        """Load the internal drives state."""
        with self.lock:
            try:
                if self.drives_file.exists():
                    with open(self.drives_file, 'r') as f:
                        return json.load(f)
            except Exception as e:
                print(f"Error loading drives: {e}")
        return None

    def get_symbols_with_clusters(self) -> (Dict[str, Symbol], Dict[int, list]):
        """Get all current symbols and their clusters."""
        symbols = self.tension_engine.symbols.copy()
        clusters = self.tension_engine.get_symbol_clusters()
        return symbols, clusters

    def get_tension_history(self) -> List[float]:
        """Get tension history"""
        return self.tension_engine.tension_history.copy()


class FSVM:
    """Main Fracture-Sourced Virtual Machine class"""
    
    def __init__(self, memory_dir: str = "fsvm_memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        self.interactions_file = self.memory_dir / "interactions.json"

        self.cycle_count = 0
        self.total_tension = 0
        self.start_time = time.time()
        
        # Activity logging
        self.activity_log = []
        
        # Phase 2: Internal Drives
        self.drives = {
            "curiosity": DRIVE_SETTINGS["curiosity"],
            "stability": DRIVE_SETTINGS["stability"],
            "boredom": DRIVE_SETTINGS["boredom"],
            "novelty_hunger": DRIVE_SETTINGS["novelty_hunger"]
        }
        self.cycles_since_new_symbol = 0
        self.last_cycle_time = time.time() # For performance tracking
        
        # Tension multiplier for UI control
        self.tension_multiplier = CORE_SETTINGS['tension_multiplier']
        
        # Use the correct, diverse entropy source
        self.entropy_sources = EntropySources()
        self.chat_input_queue = queue.Queue()
        self.chat_output_queue = queue.Queue()
            
        # Phase 4: Neural Layer
        self.neural_layer = None
        if NEURAL_SETTINGS['use_neural_network']:
            self.neural_layer = NeuralLayer()
            
        # Phase 5: Meta-Symbols
        self.symbol_co_occurrence = defaultdict(int)
        self.last_symbol_id = None
        
        # Phase 6: Thought Tracing
        self.last_thought_trace: List[ThoughtStep] = []

        # Core Components
        self.memory_store = MemoryStore(memory_dir=MEMORY_SETTINGS['memory_dir'])
        self.tension_engine = SymbolicTensionEngine(
            memory_size=CORE_SETTINGS['memory_size'],
            neural_layer=self.neural_layer
        )
        # Pass FSVM instance to tension engine for tension multiplier access
        self.tension_engine.fsvm_instance = self
        
        # Load existing memory
        loaded_symbols = self.memory_store.load_symbols()
        if loaded_symbols:
            self.tension_engine.symbols = loaded_symbols
            
            # Sort symbols by creation time to rebuild graph and clusters
            sorted_symbols = sorted(loaded_symbols.values(), key=lambda s: s.created)
            
            for i, sym in enumerate(sorted_symbols):
                # Add node for the current symbol
                self.tension_engine.symbol_graph.add_node(sym.id)
                
                # Restore clusters
                if sym.vector is not None:
                    self.tension_engine._update_clusters(sym)

                # Link to previous 5 symbols to rebuild historical connections
                for prev_sym in sorted_symbols[max(0, i-5):i]:
                    self.tension_engine.symbol_graph.add_edge(sym.id, prev_sym.id)
            
            print(f"Loaded {len(loaded_symbols)} symbols and rebuilt graph.")

        self.tension_engine.pattern_history = self.memory_store.load_patterns()

        # Communication
        self.communication_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.status_update_queue = queue.Queue()
        
        # Control flags
        self.running = False
        self.ready_to_reply = False
        self.current_processing = False
        self.cognitive_thread = None

        # Statistics
        self.cycle_count = 0
        self.total_tension = 0.0
        
    def start(self):
        """Start the FSVM cognitive loop"""
        if self.running:
            return
        
        self.running = True
        self.cognitive_thread = threading.Thread(target=self._cognitive_loop, daemon=True)
        self.cognitive_thread.start()
        print("FSVM started - cognitive loop running")
    
    def stop(self):
        """Stop the FSVM cognitive loop"""
        if not self.running:
            print("FSVM is not running")
            return
            
        self.running = False
        
        # Wait for thread to finish
        if self.cognitive_thread and self.cognitive_thread.is_alive():
            self.cognitive_thread.join(timeout=2)
        
        # Save memory before stopping
        self.memory_store.save_symbols(self.tension_engine.symbols)
        self.memory_store.save_patterns(self.tension_engine.pattern_history)
        self.memory_store.save_drives(self.drives)
        print("FSVM stopped - memory saved")
    
    def is_running(self) -> bool:
        """Check if FSVM is currently running"""
        return self.running and (self.cognitive_thread and self.cognitive_thread.is_alive())
    
    def _cognitive_loop(self):
        """The main cognitive loop of the FSVM"""
        print("FSVM cognitive loop started.")
        while self.running:
            try:
                self._autonomous_cognition_cycle()
                # The single sleep at the end of the sub-cycle is enough.
                # No need for a second sleep here.
            except Exception as e:
                if DEV_SETTINGS['debug_mode']:
                    self._log_error("Cognitive loop crash", e)

    def _autonomous_cognition_cycle(self):
        """Autonomous cognitive cycle that runs in the background"""
        if not self.running:
            return
        
        # Check for and process any queued chat messages first
        try:
            user_input = self.chat_input_queue.get_nowait()
            if hasattr(self, 'activity_log'):
                self.activity_log.append(f"Processing Chat: '{user_input}'")
            
            tension, symbol_id, response = self.tension_engine.process_input(user_input)
            
            response_str = " ".join(map(str, response)) if isinstance(response, list) else str(response)
            
            self.chat_output_queue.put(response_str)
            if self.memory_store:
                self.memory_store.save_interaction(user_input, response_str, tension)
            
            # After handling a chat, we can shorten the rest of the cycle
            time.sleep(0.5) 
            self.cycle_count += 1
            self.last_cycle_time = time.time()
            return # End cycle early after chat
        except queue.Empty:
            # No message, proceed with normal entropy gathering
            pass

        # 1. Gather a fresh batch of entropy.
        entropy_data = self.entropy_sources.get_entropy()
        
        # --- DYNAMIC LEARNING FIX ---
        # Instead of processing the whole batch and getting stuck on similar items,
        # process ONE random pattern from the batch per cycle. This forces variety.
        if not entropy_data:
            time.sleep(CORE_SETTINGS['cycle_delay'])
            return # Nothing to process

        pattern_to_process = random.choice(entropy_data)
        
        # Log the entropy gathering
        if not hasattr(self, 'activity_log'):
            self.activity_log = []
        self.activity_log.append(f"Focused on pattern: {pattern_to_process}")

        # Force creation of new symbols if we haven't created any in the last 10 seconds
        if not hasattr(self, 'last_symbol_creation_time'):
            self.last_symbol_creation_time = 0
        
        current_time = time.time()
        time_since_last_symbol = current_time - self.last_symbol_creation_time
        
        if time_since_last_symbol > 10:
            print(f"[FORCE] Creating new symbol after {time_since_last_symbol:.1f}s without learning")
            # Create a forced symbol from the chosen entropy pattern
            new_symbol = self.tension_engine.create_symbol(pattern_to_process)
            self.last_symbol_creation_time = current_time
            if new_symbol:
                self.activity_log.append(f"FORCED creation of symbol {new_symbol.id}")
        
        # 2. Process the chosen entropy pattern through the symbolic tension engine.
        symbol_created_this_cycle = False
        initial_symbol_count = len(self.tension_engine.symbols)
        
        tension, symbol_id, response = self.tension_engine.process_input(pattern_to_process)
        
        # Check if a new symbol was created
        if len(self.tension_engine.symbols) > initial_symbol_count:
            symbol_created_this_cycle = True
            self.last_symbol_creation_time = current_time
            self.activity_log.append(f"Created new symbol during cycle {self.cycle_count}")
            print(f"[LEARNING] Created symbol {symbol_id} from pattern {pattern_to_process}")
        
        # 3. Update internal drives based on learning progress.
        self._update_drives(symbol_created_this_cycle)
        
        # 4. Perform recursive hallucination if boredom is high.
        if self.drives["boredom"] > 0.7:
            self._recursive_hallucination()
        
        # 5. Update meta-symbols (only after first thought).
        if self.cycle_count > 0:
            self._update_meta_symbols()
        
        # 6. Save memory periodically.
        if self.cycle_count % 100 == 0:
            self.memory_store.save_symbols(self.tension_engine.symbols)
        
        self.cycle_count += 1
        self.last_cycle_time = time.time()
        
        # Push status update to queue for live UI updates
        self._push_status_update()
        
        # Add cycle delay
        time.sleep(CORE_SETTINGS['cycle_delay'])

    def _update_drives(self, symbol_created: bool):
        """Updates the synthetic emotional drives of the FSVM."""
        
        # Update boredom and stability
        if symbol_created:
            self.cycles_since_new_symbol = 0
            self.drives["boredom"] = 0 # Finding patterns is exciting
            self.drives["stability"] = min(1.0, self.drives["stability"] + DRIVE_SETTINGS["stability_increment"])
        else:
            self.cycles_since_new_symbol += 1
            if self.cycles_since_new_symbol > DRIVE_SETTINGS["boredom_threshold"]:
                self.drives["boredom"] = min(1.0, self.drives["boredom"] + DRIVE_SETTINGS["boredom_increment"])

        # Update curiosity and novelty hunger based on tension
        if self.tension_engine.tension_history[-1] > 0.8: # High surprise
            self.drives["curiosity"] = min(1.0, self.drives["curiosity"] + DRIVE_SETTINGS["curiosity_increment"])
        elif self.tension_engine.tension_history[-1] < 0.2: # Low surprise
            self.drives["novelty_hunger"] = min(1.0, self.drives["novelty_hunger"] + DRIVE_SETTINGS["boredom_increment"])

        # Apply decay to all drives to let them return to baseline
        for drive in self.drives:
            self.drives[drive] *= DRIVE_SETTINGS["drive_decay"]

    def _recursive_hallucination(self):
        """Performs recursive hallucination to generate novel patterns."""
        # Implement recursive hallucination logic here
        pass

    def _update_meta_symbols(self):
        """Update meta-symbols based on recent symbol activity."""
        if len(self.tension_engine.symbols) < 2:
            return
        
        # Get the most recent symbol ID properly
        symbol_keys = list(self.tension_engine.symbols.keys())
        if not symbol_keys:
            return
            
        self.last_symbol_id = symbol_keys[-1] if symbol_keys else None
        
        # Create meta-symbols from recent symbol combinations
        recent_symbols = list(self.tension_engine.symbols.values())[-5:]  # Last 5 symbols
        
        for i, symbol1 in enumerate(recent_symbols):
            for j, symbol2 in enumerate(recent_symbols[i+1:], i+1):
                # Create meta-symbol from symbol combination
                meta_pattern = [symbol1.pattern, symbol2.pattern]
                meta_id = f"MS_{len(self.tension_engine.symbols):04d}_{int(time.time()) % 10000}"
                
                # Check if similar meta-symbol already exists
                similarity_threshold = 0.7
                should_create = True
                
                for existing_symbol in self.tension_engine.symbols.values():
                    if existing_symbol.id.startswith('MS_'):
                        meta_vector = self.tension_engine._vectorize_pattern(meta_pattern)
                        if existing_symbol.vector is not None:
                            similarity = cosine_similarity([meta_vector], [existing_symbol.vector])[0][0]
                            if similarity > similarity_threshold:
                                should_create = False
                                break
                
                if should_create:
                    # Create new meta-symbol
                    meta_vector = self.tension_engine._vectorize_pattern(meta_pattern)
                    meta_symbol = Symbol(
                        id=meta_id,
                        pattern=meta_pattern,
                        vector=meta_vector,
                        created=time.time(),
                        frequency=1
                    )
                    self.tension_engine.symbols[meta_id] = meta_symbol
                    print(f"[META] Created meta-symbol {meta_id} from {symbol1.id} + {symbol2.id}")

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the FSVM"""
        return {
            'running': self.is_running(),
            'cycle_count': self.cycle_count,
            'symbols_count': len(self.tension_engine.symbols),
            'current_tension': self.tension_engine.tension_history[-1] if self.tension_engine.tension_history else 0.0,
            'average_tension': self.total_tension / max(self.cycle_count, 1),
            'ready_to_reply': self.ready_to_reply,
            'current_processing': self.current_processing,
            'short_term_memory_size': len(self.tension_engine.short_term_memory),
            'drives': self.drives.copy()
        }
    
    def _log_error(self, context: str, e: Exception):
        """Logs an error with context and traceback."""
        print(f"--- ERROR in {context} ---")
        print(f"Exception: {e}")
        traceback.print_exc()
        print("--------------------------")

    def chat(self, user_input: str):
        """Adds a user message to the input queue for async processing."""
        try:
            if not self.is_running():
                self.chat_output_queue.put("FSVM is not running. Please start it first.")
                return
            
            # Validate input
            if not user_input or not user_input.strip():
                self.chat_output_queue.put("Please provide a valid input.")
                return
                
            self.chat_input_queue.put(user_input)
            
        except Exception as e:
            # Fallback response on error
            self.chat_output_queue.put("ERROR: Unprocessable input.")
            if DEV_SETTINGS['debug_mode']:
                self._log_error("Chat input error", e)

    def get_cycles_per_second(self) -> float:
        """Calculate the number of cognitive cycles running per second."""
        if self.cycle_count == 0 or not hasattr(self, 'start_time'):
            return 0.0
        
        uptime = time.time() - self.start_time
        if uptime < 1: # Avoid division by zero if uptime is very short
            return 0.0
        
        return self.cycle_count / uptime

    def _push_status_update(self):
        """Push current status to the status update queue for live UI updates"""
        try:
            status = {
                'cycle_count': self.cycle_count,
                'current_tension': self.tension_engine.get_current_tension(),
                'symbol_count': len(self.tension_engine.symbols),
                'drives': self.drives.copy(),
                'uptime': time.time() - self.start_time,
                'last_activity': self.activity_log[-1] if self.activity_log else "No activity",
                'timestamp': time.time()
            }
            self.status_update_queue.put_nowait(status)
        except queue.Full:
            # Queue is full, skip this update
            pass

    def get_status_update(self):
        """Get the latest status update from the queue"""
        try:
            return self.status_update_queue.get_nowait()
        except queue.Empty:
            return None

    def set_tension_multiplier(self, multiplier: float):
        """Update the tension multiplier for UI control"""
        self.tension_multiplier = max(0.1, min(5.0, multiplier))  # Clamp between 0.1 and 5.0
        print(f"[CONFIG] Tension multiplier set to {self.tension_multiplier}")

    def get_tension_multiplier(self) -> float:
        """Get the current tension multiplier"""
        return self.tension_multiplier 