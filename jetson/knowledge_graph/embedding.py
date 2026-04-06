"""Simple embedding and similarity for NEXUS knowledge graph."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

random.seed(42)


@dataclass
class EmbeddingVector:
    """A fixed-dimensional vector with optional metadata."""
    id: str
    vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def dimensions(self) -> int:
        return len(self.vector)

    def magnitude(self) -> float:
        return math.sqrt(sum(v * v for v in self.vector))


class SimpleEmbedder:
    """Pure-Python embedder using bag-of-words hashing with cosine similarity, k-means, and PCA."""

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._vocab: Dict[str, int] = {}
        self._vocab_size = 0
        self._deterministic_rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_text(
        self,
        text: str,
        dimensions: int = 64,
    ) -> EmbeddingVector:
        """Create a bag-of-words style embedding vector via hashing."""
        tokens = self._tokenize(text)
        vector = [0.0] * dimensions

        for token in tokens:
            idx = self._hash_to_index(token, dimensions)
            vector[idx] += 1.0

        # L2 normalize
        mag = math.sqrt(sum(v * v for v in vector)) or 1.0
        vector = [v / mag for v in vector]

        text_id = hashlib.md5(text.encode()).hexdigest()[:12]
        return EmbeddingVector(id=text_id, vector=vector, metadata={"text": text, "tokens": len(tokens)})

    def embed_batch(
        self,
        texts: List[str],
        dimensions: int = 64,
    ) -> List[EmbeddingVector]:
        return [self.embed_text(t, dimensions) for t in texts]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + lowercase tokenizer."""
        text = text.lower()
        # split on non-alphanumeric
        tokens: List[str] = []
        current: List[str] = []
        for ch in text:
            if ch.isalnum() or ch == '_':
                current.append(ch)
            else:
                if current:
                    tokens.append("".join(current))
                    current = []
        if current:
            tokens.append("".join(current))
        return tokens

    @staticmethod
    def _hash_to_index(token: str, dimensions: int) -> int:
        """Hash token to a valid index."""
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        return h % dimensions

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    @staticmethod
    def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimensionality")
        dot = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(a * a for a in vec1)) or 1e-10
        mag2 = math.sqrt(sum(b * b for b in vec2)) or 1e-10
        return dot / (mag1 * mag2)

    @staticmethod
    def compute_euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
        """Compute Euclidean distance between two vectors."""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimensionality")
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def find_similar(
        self,
        query: List[float],
        embeddings: List[EmbeddingVector],
        top_k: int = 5,
    ) -> List[Tuple[EmbeddingVector, float]]:
        """Find the top-k most similar embeddings to a query vector."""
        scored: List[Tuple[EmbeddingVector, float]] = []
        for emb in embeddings:
            sim = self.compute_cosine_similarity(query, emb.vector)
            scored.append((emb, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def find_similar_by_text(
        self,
        text: str,
        embeddings: List[EmbeddingVector],
        top_k: int = 5,
        dimensions: int = 64,
    ) -> List[Tuple[EmbeddingVector, float]]:
        """Find similar embeddings by query text."""
        query_emb = self.embed_text(text, dimensions)
        return self.find_similar(query_emb.vector, embeddings, top_k)

    # ------------------------------------------------------------------
    # Clustering (simple k-means)
    # ------------------------------------------------------------------

    def cluster(
        self,
        embeddings: List[EmbeddingVector],
        k: int = 3,
        max_iterations: int = 100,
    ) -> List[int]:
        """Simple k-means clustering.  Returns assignment list."""
        if not embeddings:
            return []
        if k <= 0:
            return [0] * len(embeddings)
        if k >= len(embeddings):
            return list(range(len(embeddings)))

        dims = embeddings[0].dimensions
        rng = random.Random(self._seed)

        # Initialize centroids by random selection
        indices = list(range(len(embeddings)))
        rng.shuffle(indices)
        centroids = [list(embeddings[indices[i]].vector) for i in range(k)]

        assignments = [0] * len(embeddings)

        for _ in range(max_iterations):
            new_assignments = [0] * len(embeddings)
            # Assign each vector to nearest centroid
            for i, emb in enumerate(embeddings):
                best_cluster = 0
                best_dist = float("inf")
                for c in range(k):
                    dist = self.compute_euclidean_distance(emb.vector, centroids[c])
                    if dist < best_dist:
                        best_dist = dist
                        best_cluster = c
                new_assignments[i] = best_cluster

            # Check convergence
            if new_assignments == assignments:
                break
            assignments = new_assignments

            # Update centroids
            for c in range(k):
                members = [embeddings[i] for i in range(len(assignments)) if assignments[i] == c]
                if members:
                    for d in range(dims):
                        centroids[c][d] = sum(m.vector[d] for m in members) / len(members)

        return assignments

    # ------------------------------------------------------------------
    # Dimensionality reduction (simple random projection)
    # ------------------------------------------------------------------

    def reduce_dimension(
        self,
        embeddings: List[EmbeddingVector],
        target_dim: int = 10,
    ) -> List[List[float]]:
        """Reduce dimensionality using random projection.

        Returns list of reduced vectors (not EmbeddingVector, for simplicity).
        """
        if not embeddings:
            return []
        original_dim = embeddings[0].dimensions
        if target_dim >= original_dim:
            return [list(e.vector[:target_dim]) + [0.0] * max(0, target_dim - len(e.vector)) for e in embeddings]

        rng = random.Random(self._seed)
        # Generate projection matrix
        projection: List[List[float]] = []
        for _ in range(target_dim):
            row = [rng.gauss(0, 1.0 / math.sqrt(original_dim)) for _ in range(original_dim)]
            projection.append(row)

        reduced: List[List[float]] = []
        for emb in embeddings:
            new_vec = []
            for row in projection:
                val = sum(row[d] * emb.vector[d] for d in range(original_dim))
                new_vec.append(val)
            reduced.append(new_vec)

        return reduced

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def vector_mean(vectors: List[List[float]]) -> List[float]:
        """Compute the mean of a list of vectors."""
        if not vectors:
            return []
        dim = len(vectors[0])
        mean = [0.0] * dim
        for vec in vectors:
            for d in range(dim):
                mean[d] += vec[d]
        return [v / len(vectors) for v in mean]

    @staticmethod
    def vector_add(v1: List[float], v2: List[float]) -> List[float]:
        return [a + b for a, b in zip(v1, v2)]

    @staticmethod
    def vector_sub(v1: List[float], v2: List[float]) -> List[float]:
        return [a - b for a, b in zip(v1, v2)]

    @staticmethod
    def vector_scale(v: List[float], scalar: float) -> List[float]:
        return [x * scalar for x in v]

    @staticmethod
    def vector_dot(v1: List[float], v2: List[float]) -> float:
        return sum(a * b for a, b in zip(v1, v2))
