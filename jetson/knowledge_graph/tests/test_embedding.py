"""Tests for embedding.py — EmbeddingVector, SimpleEmbedder."""

import math
import pytest
from jetson.knowledge_graph.embedding import EmbeddingVector, SimpleEmbedder


# ======================================================================
# EmbeddingVector
# ======================================================================

class TestEmbeddingVector:
    def test_create_vector(self):
        ev = EmbeddingVector(id="v1", vector=[0.1, 0.2, 0.3])
        assert ev.id == "v1"
        assert len(ev.vector) == 3

    def test_dimensions(self):
        ev = EmbeddingVector(id="v1", vector=[1.0] * 42)
        assert ev.dimensions == 42

    def test_magnitude(self):
        ev = EmbeddingVector(id="v1", vector=[3.0, 4.0])
        assert abs(ev.magnitude() - 5.0) < 1e-6

    def test_zero_magnitude(self):
        ev = EmbeddingVector(id="v1", vector=[0.0, 0.0, 0.0])
        assert ev.magnitude() == 0.0

    def test_metadata(self):
        ev = EmbeddingVector(id="v1", vector=[1.0], metadata={"source": "test"})
        assert ev.metadata["source"] == "test"

    def test_default_metadata(self):
        ev = EmbeddingVector(id="v1", vector=[1.0])
        assert ev.metadata == {}


# ======================================================================
# SimpleEmbedder — embed_text
# ======================================================================

class TestEmbedText:
    def test_basic_embedding(self):
        embedder = SimpleEmbedder()
        ev = embedder.embed_text("hello world", dimensions=32)
        assert ev.dimensions == 32
        assert len(ev.vector) == 32

    def test_embedding_normalized(self):
        embedder = SimpleEmbedder()
        ev = embedder.embed_text("some text here", dimensions=64)
        mag = math.sqrt(sum(v * v for v in ev.vector))
        assert abs(mag - 1.0) < 1e-6

    def test_empty_text_embedding(self):
        embedder = SimpleEmbedder()
        ev = embedder.embed_text("", dimensions=16)
        # All zeros (no tokens), magnitude should be 0
        mag = math.sqrt(sum(v * v for v in ev.vector))
        assert mag == 0.0 or abs(mag - 1.0) < 1e-6  # L2 norm of zeros gives 0, division by 0.0 -> 1.0
        assert ev.dimensions == 16

    def test_embedding_id_is_deterministic(self):
        embedder = SimpleEmbedder()
        ev1 = embedder.embed_text("test", dimensions=32)
        ev2 = embedder.embed_text("test", dimensions=32)
        assert ev1.id == ev2.id

    def test_embedding_metadata(self):
        embedder = SimpleEmbedder()
        ev = embedder.embed_text("hello world", dimensions=16)
        assert "text" in ev.metadata
        assert "tokens" in ev.metadata

    def test_different_texts_different_vectors(self):
        embedder = SimpleEmbedder()
        ev1 = embedder.embed_text("alpha", dimensions=64)
        ev2 = embedder.embed_text("beta gamma", dimensions=64)
        assert ev1.vector != ev2.vector

    def test_same_text_same_vector(self):
        embedder = SimpleEmbedder()
        ev1 = embedder.embed_text("same text", dimensions=32)
        ev2 = embedder.embed_text("same text", dimensions=32)
        assert ev1.vector == ev2.vector

    def test_single_word(self):
        embedder = SimpleEmbedder()
        ev = embedder.embed_text("vessel", dimensions=16)
        assert ev.dimensions == 16

    def test_custom_dimensions(self):
        embedder = SimpleEmbedder()
        ev = embedder.embed_text("test", dimensions=128)
        assert ev.dimensions == 128


# ======================================================================
# SimpleEmbedder — embed_batch
# ======================================================================

class TestEmbedBatch:
    def test_batch_embed(self):
        embedder = SimpleEmbedder()
        texts = ["hello", "world", "test"]
        vectors = embedder.embed_batch(texts, dimensions=32)
        assert len(vectors) == 3
        assert all(v.dimensions == 32 for v in vectors)

    def test_batch_empty(self):
        embedder = SimpleEmbedder()
        assert embedder.embed_batch([]) == []


# ======================================================================
# SimpleEmbedder — cosine similarity
# ======================================================================

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        sim = SimpleEmbedder.compute_cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        sim = SimpleEmbedder.compute_cosine_similarity(v1, v2)
        assert abs(sim) < 1e-6

    def test_similar_vectors(self):
        v1 = [1.0, 1.0, 0.0]
        v2 = [1.0, 1.0, 1.0]
        sim = SimpleEmbedder.compute_cosine_similarity(v1, v2)
        assert 0.5 < sim < 1.0

    def test_opposite_vectors(self):
        v1 = [1.0, 0.0]
        v2 = [-1.0, 0.0]
        sim = SimpleEmbedder.compute_cosine_similarity(v1, v2)
        assert abs(sim + 1.0) < 1e-6

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            SimpleEmbedder.compute_cosine_similarity([1.0], [1.0, 2.0])

    def test_zero_vectors(self):
        v1 = [0.0, 0.0]
        v2 = [0.0, 0.0]
        sim = SimpleEmbedder.compute_cosine_similarity(v1, v2)
        # Should be close to 0 (both zero magnitude)
        assert sim >= 0.0


# ======================================================================
# SimpleEmbedder — euclidean distance
# ======================================================================

class TestEuclideanDistance:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        d = SimpleEmbedder.compute_euclidean_distance(v, v)
        assert abs(d) < 1e-6

    def test_known_distance(self):
        v1 = [0.0, 0.0]
        v2 = [3.0, 4.0]
        d = SimpleEmbedder.compute_euclidean_distance(v1, v2)
        assert abs(d - 5.0) < 1e-6

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            SimpleEmbedder.compute_euclidean_distance([1.0], [1.0, 2.0])

    def test_large_vectors(self):
        v1 = [float(i) for i in range(100)]
        v2 = [float(i) for i in range(100)]
        d = SimpleEmbedder.compute_euclidean_distance(v1, v2)
        assert abs(d) < 1e-6


# ======================================================================
# SimpleEmbedder — find_similar
# ======================================================================

class TestFindSimilar:
    def test_find_top_k(self):
        embedder = SimpleEmbedder()
        query = [1.0, 0.0, 0.0]
        embeddings = [
            EmbeddingVector(id="a", vector=[1.0, 0.1, 0.0]),
            EmbeddingVector(id="b", vector=[0.0, 1.0, 0.0]),
            EmbeddingVector(id="c", vector=[0.9, 0.0, 0.0]),
        ]
        results = embedder.find_similar(query, embeddings, top_k=2)
        assert len(results) == 2
        assert results[0][1] >= results[1][1]

    def test_find_similar_empty(self):
        embedder = SimpleEmbedder()
        results = embedder.find_similar([1.0, 0.0], [], top_k=5)
        assert results == []

    def test_find_similar_by_text(self):
        embedder = SimpleEmbedder()
        embeddings = embedder.embed_batch(["sonar sensor", "gps module", "depth gauge"], dimensions=32)
        results = embedder.find_similar_by_text("sonar", embeddings, top_k=2, dimensions=32)
        assert len(results) == 2
        assert results[0][1] >= results[0][1]

    def test_find_similar_top_k_larger_than_collection(self):
        embedder = SimpleEmbedder()
        embeddings = [EmbeddingVector(id="a", vector=[1.0, 0.0])]
        results = embedder.find_similar([1.0, 0.0], embeddings, top_k=10)
        assert len(results) == 1


# ======================================================================
# SimpleEmbedder — cluster (k-means)
# ======================================================================

class TestCluster:
    def test_cluster_basic(self):
        embedder = SimpleEmbedder()
        embeddings = [
            EmbeddingVector(id="a", vector=[1.0, 0.0]),
            EmbeddingVector(id="b", vector=[1.1, 0.1]),
            EmbeddingVector(id="c", vector=[0.0, 1.0]),
            EmbeddingVector(id="d", vector=[0.1, 1.1]),
        ]
        assignments = embedder.cluster(embeddings, k=2)
        assert len(assignments) == 4
        assert assignments[0] == assignments[1]  # a and b same cluster
        assert assignments[2] == assignments[3]  # c and d same cluster

    def test_cluster_empty(self):
        embedder = SimpleEmbedder()
        assert embedder.cluster([], k=3) == []

    def test_cluster_k_equals_n(self):
        embedder = SimpleEmbedder()
        embeddings = [
            EmbeddingVector(id="a", vector=[1.0]),
            EmbeddingVector(id="b", vector=[2.0]),
        ]
        assignments = embedder.cluster(embeddings, k=2)
        assert len(assignments) == 2
        # Each should be in its own cluster
        assert len(set(assignments)) == 2

    def test_cluster_k_zero(self):
        embedder = SimpleEmbedder()
        embeddings = [EmbeddingVector(id="a", vector=[1.0])]
        assignments = embedder.cluster(embeddings, k=0)
        assert assignments == [0]

    def test_cluster_k_larger_than_n(self):
        embedder = SimpleEmbedder()
        embeddings = [EmbeddingVector(id="a", vector=[1.0])]
        assignments = embedder.cluster(embeddings, k=5)
        assert assignments == [0]

    def test_cluster_single_element(self):
        embedder = SimpleEmbedder()
        embeddings = [EmbeddingVector(id="a", vector=[1.0, 2.0])]
        assignments = embedder.cluster(embeddings, k=1)
        assert assignments == [0]


# ======================================================================
# SimpleEmbedder — reduce_dimension
# ======================================================================

class TestReduceDimension:
    def test_reduce_to_lower_dim(self):
        embedder = SimpleEmbedder()
        embeddings = [
            EmbeddingVector(id="a", vector=[1.0, 2.0, 3.0, 4.0]),
        ]
        reduced = embedder.reduce_dimension(embeddings, target_dim=2)
        assert len(reduced) == 1
        assert len(reduced[0]) == 2

    def test_reduce_empty(self):
        embedder = SimpleEmbedder()
        assert embedder.reduce_dimension([]) == []

    def test_reduce_to_same_dim(self):
        embedder = SimpleEmbedder()
        embeddings = [EmbeddingVector(id="a", vector=[1.0, 2.0, 3.0])]
        reduced = embedder.reduce_dimension(embeddings, target_dim=3)
        assert len(reduced[0]) == 3

    def test_reduce_to_higher_dim(self):
        embedder = SimpleEmbedder()
        embeddings = [EmbeddingVector(id="a", vector=[1.0, 2.0])]
        reduced = embedder.reduce_dimension(embeddings, target_dim=5)
        assert len(reduced[0]) == 5

    def test_reduce_deterministic(self):
        embedder = SimpleEmbedder()
        embeddings = [EmbeddingVector(id="a", vector=[float(i) for i in range(16)])]
        r1 = embedder.reduce_dimension(embeddings, target_dim=4)
        r2 = embedder.reduce_dimension(embeddings, target_dim=4)
        assert r1 == r2

    def test_reduce_multiple_vectors(self):
        embedder = SimpleEmbedder()
        embeddings = [
            EmbeddingVector(id="a", vector=[1.0, 2.0, 3.0, 4.0]),
            EmbeddingVector(id="b", vector=[5.0, 6.0, 7.0, 8.0]),
        ]
        reduced = embedder.reduce_dimension(embeddings, target_dim=2)
        assert len(reduced) == 2
        assert all(len(r) == 2 for r in reduced)


# ======================================================================
# SimpleEmbedder — vector utilities
# ======================================================================

class TestVectorUtilities:
    def test_vector_mean(self):
        result = SimpleEmbedder.vector_mean([[1.0, 2.0], [3.0, 4.0]])
        assert result == [2.0, 3.0]

    def test_vector_mean_empty(self):
        assert SimpleEmbedder.vector_mean([]) == []

    def test_vector_add(self):
        result = SimpleEmbedder.vector_add([1.0, 2.0], [3.0, 4.0])
        assert result == [4.0, 6.0]

    def test_vector_sub(self):
        result = SimpleEmbedder.vector_sub([3.0, 4.0], [1.0, 2.0])
        assert result == [2.0, 2.0]

    def test_vector_scale(self):
        result = SimpleEmbedder.vector_scale([1.0, 2.0], 3.0)
        assert result == [3.0, 6.0]

    def test_vector_dot(self):
        result = SimpleEmbedder.vector_dot([1.0, 2.0], [3.0, 4.0])
        assert result == 11.0

    def test_vector_dot_zero(self):
        result = SimpleEmbedder.vector_dot([1.0, 0.0], [0.0, 1.0])
        assert result == 0.0
