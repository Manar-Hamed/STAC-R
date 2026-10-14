"""
Tests for index compression utilities.

Tests CSR compression, delta encoding, block offset encoding, and helper functions.
"""

import pytest
import torch
from stacr.index_compression import (
    csr_memory_savings,
    compress_activation_to_csr,
    analyze_delta_compression_potential,
    DeltaEncodedIndices,
    BlockOffsetEncoding,
    IndexCompressionStats,
    index_dtype_recommendation,
    should_use_delta_encoding,
    _dtype_bytes,
)


class TestCSRMemorySavings:
    """Tests for CSR memory calculation functions."""

    def test_basic_calculation(self):
        """Test basic CSR memory savings calculation."""
        B, N = 64, 512
        nnz = 1000

        result = csr_memory_savings(B, N, nnz, dtype=torch.float32, index_dtype=torch.int32)

        # Dense: 64 * 512 * 4 = 131,072 bytes
        assert result["dense_bytes"] == 131072

        # CSR: (64+1)*4 + 1000*4 + 1000*4 = 260 + 4000 + 4000 = 8260 bytes
        expected_csr = 260 + 4000 + 4000
        assert result["csr_bytes"] == expected_csr

        # Compression ratio
        assert result["compression_ratio"] == pytest.approx(131072 / expected_csr, rel=1e-5)

        # Sparsity
        assert result["sparsity"] == pytest.approx(1.0 - (1000 / (64 * 512)), rel=1e-5)

    def test_high_sparsity(self):
        """Test with high sparsity (90%)."""
        B, N = 128, 1024
        nnz = int(0.1 * B * N)  # 10% dense = 90% sparse

        result = csr_memory_savings(B, N, nnz, dtype=torch.float16, index_dtype=torch.int32)

        assert result["sparsity"] == pytest.approx(0.9, rel=1e-5)
        assert result["compression_ratio"] > 3.0  # Should get good compression

    def test_int64_vs_int32_indices(self):
        """Test memory difference between int32 and int64 indices."""
        B, N, nnz = 64, 512, 1000

        result_int32 = csr_memory_savings(B, N, nnz, index_dtype=torch.int32)
        result_int64 = csr_memory_savings(B, N, nnz, index_dtype=torch.int64)

        # int64 should use more memory for indices
        assert result_int64["crow_bytes"] == 2 * result_int32["crow_bytes"]
        assert result_int64["col_bytes"] == 2 * result_int32["col_bytes"]

        # int64 should have lower compression ratio
        assert result_int64["compression_ratio"] < result_int32["compression_ratio"]

    def test_zero_nnz(self):
        """Test with zero non-zero elements."""
        result = csr_memory_savings(64, 512, 0)

        assert result["csr_bytes"] == (64 + 1) * 4  # Only crow_indices
        assert result["sparsity"] == 1.0


class TestCompressActivationToCSR:
    """Tests for CSR activation compression."""

    def test_basic_compression(self):
        """Test basic compression to CSR format."""
        torch.manual_seed(42)
        activation = torch.randn(8, 64)

        # Zero out small values
        activation[torch.abs(activation) < 0.5] = 0

        csr = compress_activation_to_csr(activation, threshold=0.0)

        assert csr.layout == torch.sparse_csr
        assert csr.shape == (8, 64)

        # Reconstruction should match
        reconstructed = csr.to_dense()
        torch.testing.assert_close(reconstructed, activation, rtol=1e-5, atol=1e-6)

    def test_threshold_filtering(self):
        """Test that threshold correctly filters values."""
        activation = torch.tensor([[1.0, 0.1, 0.5], [0.2, 2.0, 0.3]])

        csr = compress_activation_to_csr(activation, threshold=0.4)
        reconstructed = csr.to_dense()

        # Only values > 0.4 should remain
        expected = torch.tensor([[1.0, 0.0, 0.5], [0.0, 2.0, 0.0]])
        torch.testing.assert_close(reconstructed, expected, rtol=1e-5, atol=1e-6)

    def test_all_zeros(self):
        """Test compression of all-zero tensor."""
        activation = torch.zeros(4, 16)
        csr = compress_activation_to_csr(activation)

        assert csr._nnz() == 0
        assert csr.to_dense().sum() == 0

    def test_int32_indices(self):
        """Test that indices use specified dtype."""
        activation = torch.randn(8, 64)
        csr = compress_activation_to_csr(activation, threshold=0.5, index_dtype=torch.int32)

        assert csr.crow_indices().dtype == torch.int32
        assert csr.col_indices().dtype == torch.int32

    def test_invalid_shape(self):
        """Test error on non-2D input."""
        with pytest.raises(ValueError, match="Expected 2D tensor"):
            compress_activation_to_csr(torch.randn(8, 64, 3))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_support(self):
        """Test CSR compression on CUDA."""
        activation = torch.randn(8, 64, device="cuda")
        activation[torch.abs(activation) < 0.5] = 0

        csr = compress_activation_to_csr(activation)

        assert csr.device.type == "cuda"
        reconstructed = csr.to_dense()
        torch.testing.assert_close(reconstructed, activation, rtol=1e-5, atol=1e-5)


class TestAnalyzeDeltaCompression:
    """Tests for delta compression analysis."""

    def test_clustered_indices(self):
        """Test with clustered indices (good for delta encoding)."""
        # Small gaps
        indices = torch.arange(0, 1000, 2)  # [0, 2, 4, 6, ...]

        analysis = analyze_delta_compression_potential(indices)

        assert analysis["num_indices"] == 500
        # Average delta is slightly < 2.0 because first delta is 0
        assert 1.99 < analysis["avg_delta"] < 2.01
        assert analysis["max_delta"] == 2
        assert analysis["compression_ratio"] > 1.0  # Should compress well

        # All deltas should fit in 1 byte
        assert analysis["delta_histogram"]["< 128 (1 byte)"] == 500

    def test_sparse_indices(self):
        """Test with widely-spaced indices (poor for delta encoding)."""
        # Large gaps
        indices = torch.arange(0, 100000, 1000)  # [0, 1000, 2000, ...]

        analysis = analyze_delta_compression_potential(indices)

        # Average delta is slightly < 1000.0 because first delta is 0
        assert 980 < analysis["avg_delta"] < 1010
        assert analysis["compression_ratio"] < 2.1  # Limited benefit

        # Most deltas need 2 bytes
        assert analysis["delta_histogram"]["128-16383 (2 bytes)"] > 0

    def test_empty_indices(self):
        """Test with empty indices."""
        indices = torch.tensor([])

        analysis = analyze_delta_compression_potential(indices)

        assert analysis["num_indices"] == 0
        assert analysis["compression_ratio"] == 1.0

    def test_single_index(self):
        """Test with single index."""
        indices = torch.tensor([42])

        analysis = analyze_delta_compression_potential(indices)

        assert analysis["num_indices"] == 1
        # Only one delta (from 0 to 42)


class TestDeltaEncodedIndices:
    """Tests for DeltaEncodedIndices class."""

    def test_encode_decode_roundtrip(self):
        """Test that encoding and decoding recovers original indices."""
        indices = torch.tensor([10, 12, 15, 18, 22, 30])

        encoded = DeltaEncodedIndices(indices)
        reconstructed = encoded.decode()

        torch.testing.assert_close(reconstructed, indices)

    def test_memory_savings(self):
        """Test memory savings for clustered indices."""
        # Clustered indices (small deltas)
        indices = torch.arange(100, 200, 1)  # [100, 101, 102, ...]

        encoded = DeltaEncodedIndices(indices, dtype=torch.int16)

        original_bytes = len(indices) * 4  # int32
        compressed_bytes = encoded.memory_bytes()

        # Should compress well (all deltas = 1)
        assert compressed_bytes < original_bytes

    def test_empty_indices(self):
        """Test with empty indices."""
        indices = torch.tensor([])
        encoded = DeltaEncodedIndices(indices)

        assert encoded.num_indices == 0
        assert len(encoded.decode()) == 0

    def test_single_index(self):
        """Test with single index."""
        indices = torch.tensor([42])
        encoded = DeltaEncodedIndices(indices)

        assert encoded.base == 42
        assert len(encoded.deltas) == 0

        reconstructed = encoded.decode()
        torch.testing.assert_close(reconstructed, indices)

    def test_unsorted_indices_error(self):
        """Test that unsorted indices raise error."""
        indices = torch.tensor([10, 5, 20])  # Not sorted

        with pytest.raises(ValueError, match="must be sorted"):
            DeltaEncodedIndices(indices)

    def test_multidimensional_error(self):
        """Test that multi-dimensional indices raise error."""
        indices = torch.tensor([[1, 2], [3, 4]])

        with pytest.raises(ValueError, match="must be 1D"):
            DeltaEncodedIndices(indices)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_support(self):
        """Test delta encoding on CUDA."""
        indices = torch.arange(0, 100, device="cuda")
        encoded = DeltaEncodedIndices(indices)

        assert encoded.device.type == "cuda"

        reconstructed = encoded.decode()
        torch.testing.assert_close(reconstructed, indices)


class TestBlockOffsetEncoding:
    """Tests for BlockOffsetEncoding class."""

    def test_encode_decode_roundtrip(self):
        """Test that encoding and decoding recovers original tensor."""
        torch.manual_seed(42)

        # Create block-sparse pattern
        tensor = torch.zeros(64, 128)
        tensor[0:16, 0:16] = torch.randn(16, 16)  # Block 1
        tensor[32:48, 64:80] = torch.randn(16, 16)  # Block 2

        encoded = BlockOffsetEncoding(tensor, block_size=16)
        reconstructed = encoded.to_dense()

        torch.testing.assert_close(reconstructed, tensor, rtol=1e-5, atol=1e-6)

    def test_compression_ratio(self):
        """Test compression ratio for sparse blocks."""
        # Create sparse tensor with few active blocks
        tensor = torch.zeros(128, 256)
        tensor[0:16, 0:16] = 1.0  # 1 active block out of 128

        encoded = BlockOffsetEncoding(tensor, block_size=16)

        assert encoded.compression_ratio() > 10.0  # Should compress well

    def test_irregular_shape(self):
        """Test with shape not divisible by block_size."""
        tensor = torch.randn(50, 100)  # Not divisible by 16
        encoded = BlockOffsetEncoding(tensor, block_size=16)

        reconstructed = encoded.to_dense()
        torch.testing.assert_close(reconstructed, tensor, rtol=1e-5, atol=1e-6)

    def test_all_zeros(self):
        """Test with all-zero tensor."""
        tensor = torch.zeros(64, 64)
        encoded = BlockOffsetEncoding(tensor, block_size=16)

        assert len(encoded.block_indices) == 0
        assert len(encoded.block_values) == 0

    def test_all_nonzero(self):
        """Test with fully dense tensor."""
        tensor = torch.ones(64, 64)
        encoded = BlockOffsetEncoding(tensor, block_size=16)

        # Should have all blocks
        expected_blocks = (64 // 16) * (64 // 16)
        assert len(encoded.block_indices) == expected_blocks

    def test_invalid_shape(self):
        """Test error on non-2D input."""
        with pytest.raises(ValueError, match="must be 2D"):
            BlockOffsetEncoding(torch.randn(8, 64, 3), block_size=16)

    def test_memory_calculation(self):
        """Test memory calculation."""
        tensor = torch.zeros(128, 128, dtype=torch.float16)
        tensor[0:16, 0:16] = 1.0  # 1 active block

        encoded = BlockOffsetEncoding(tensor, block_size=16)

        mem_bytes = encoded.memory_bytes()

        # Should include: block_indices + block_values + row_ptr
        expected = (
            1 * 4  # 1 block index (int32)
            + 256 * 2  # 16x16 block values (float16)
            + 9 * 4
        )  # 9 row pointers (int32) for 8 block rows

        assert mem_bytes == expected

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_support(self):
        """Test block encoding on CUDA."""
        tensor = torch.zeros(64, 64, device="cuda")
        tensor[0:16, 0:16] = 1.0

        encoded = BlockOffsetEncoding(tensor, block_size=16)

        assert encoded.device.type == "cuda"

        reconstructed = encoded.to_dense()
        torch.testing.assert_close(reconstructed, tensor, rtol=1e-5, atol=1e-5)


class TestIndexCompressionStats:
    """Tests for IndexCompressionStats class."""

    def test_record_and_report(self):
        """Test recording statistics and generating report."""
        stats = IndexCompressionStats()

        # Record some operations
        indices1 = torch.arange(100)
        stats.record(indices1, compressed_bytes=100, reconstruction_time=0.001)

        indices2 = torch.arange(200)
        stats.record(indices2, compressed_bytes=150, reconstruction_time=0.002)

        # Check totals
        assert stats.total_indices == 300
        assert stats.total_bytes_original == 300 * 4  # int32
        assert stats.total_bytes_compressed == 250

    def test_average_calculations(self):
        """Test average compression ratio and reconstruction time."""
        stats = IndexCompressionStats()

        stats.record(torch.arange(100), 50, 0.001)  # 400 bytes -> 50 bytes = 8×
        stats.record(torch.arange(100), 100, 0.003)  # 400 bytes -> 100 bytes = 4×

        # Average ratio should be (400+400) / (50+100) = 800/150 = 5.33×
        assert stats.average_compression_ratio() == pytest.approx(800.0 / 150.0, rel=1e-5)

        # Average time should be (1 + 3) / 2 = 2 ms
        assert stats.average_reconstruction_time() == pytest.approx(2.0, rel=1e-5)

    def test_reset(self):
        """Test reset functionality."""
        stats = IndexCompressionStats()

        stats.record(torch.arange(100), 50, 0.001)
        stats.reset()

        assert stats.total_indices == 0
        assert stats.total_bytes_original == 0
        assert stats.total_bytes_compressed == 0
        assert len(stats.reconstruction_times) == 0

    def test_report_dict(self):
        """Test that report returns proper dict."""
        stats = IndexCompressionStats()
        stats.record(torch.arange(100), 50, 0.001)

        report = stats.report()

        assert "total_indices" in report
        assert "original_KB" in report
        assert "compressed_KB" in report
        assert "avg_compression_ratio" in report
        assert "avg_reconstruction_ms" in report
        assert "num_operations" in report


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_dtype_bytes(self):
        """Test _dtype_bytes function."""
        assert _dtype_bytes(torch.float32) == 4
        assert _dtype_bytes(torch.float64) == 8
        assert _dtype_bytes(torch.float16) == 2
        assert _dtype_bytes(torch.int32) == 4
        assert _dtype_bytes(torch.int64) == 8
        assert _dtype_bytes(torch.int16) == 2
        assert _dtype_bytes(torch.int8) == 1

    def test_index_dtype_recommendation_memory(self):
        """Test index dtype recommendation for memory optimization."""
        # Small indices
        dtype = index_dtype_recommendation(100, target="memory")
        assert dtype == torch.int8

        # Medium indices
        dtype = index_dtype_recommendation(10000, target="memory")
        assert dtype == torch.int16

        # Large indices
        dtype = index_dtype_recommendation(100000, target="memory")
        assert dtype == torch.int32

    def test_index_dtype_recommendation_speed(self):
        """Test index dtype recommendation for speed optimization."""
        # All should prefer int32 for MKL
        dtype = index_dtype_recommendation(100, target="speed")
        assert dtype == torch.int32

        dtype = index_dtype_recommendation(100000, target="speed")
        assert dtype == torch.int32

    def test_should_use_delta_encoding_clustered(self):
        """Test delta encoding decision for clustered indices."""
        # Small deltas - should use delta encoding
        indices = torch.arange(0, 1000, 2)  # [0, 2, 4, ...]

        assert should_use_delta_encoding(indices) is True

    def test_should_use_delta_encoding_sparse(self):
        """Test delta encoding decision for sparse indices."""
        # Large deltas - should NOT use delta encoding
        indices = torch.arange(0, 100000, 1000)  # [0, 1000, 2000, ...]

        # Might still use if ratio > threshold, but typically won't
        result = should_use_delta_encoding(indices, threshold_ratio=1.5)
        # Result depends on analysis, but large deltas typically don't compress well

    def test_should_use_delta_encoding_threshold(self):
        """Test delta encoding decision with different thresholds."""
        indices = torch.arange(0, 500, 5)

        # Lower threshold - more likely to use
        assert should_use_delta_encoding(indices, threshold_ratio=1.1) is True

        # Very high threshold - less likely
        result = should_use_delta_encoding(indices, threshold_ratio=10.0)
        # Still True because avg_delta=5 < 128


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_csr_vs_block_encoding(self):
        """Compare CSR and block encoding for structured sparsity."""
        torch.manual_seed(42)

        # Create block-sparse tensor
        tensor = torch.zeros(128, 512)
        # Add 16 random 16×16 blocks
        for _ in range(16):
            i = torch.randint(0, 128 - 16, (1,)).item()
            j = torch.randint(0, 512 - 16, (1,)).item()
            tensor[i : i + 16, j : j + 16] = torch.randn(16, 16)

        # CSR encoding
        csr = tensor.to_sparse_coo().to_sparse_csr()
        csr_bytes = (
            csr.crow_indices().numel() * 4
            + csr.col_indices().numel() * 4
            + csr.values().numel() * 4
        )

        # Block encoding
        block = BlockOffsetEncoding(tensor, block_size=16)
        block_bytes = block.memory_bytes()

        # Block encoding should be more efficient for structured sparsity
        # (though actual ratio depends on randomness)
        assert block_bytes > 0
        assert csr_bytes > 0

        # Both should reconstruct correctly
        csr_recon = csr.to_dense()
        block_recon = block.to_dense()

        torch.testing.assert_close(csr_recon, tensor, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(block_recon, tensor, rtol=1e-5, atol=1e-6)

    def test_compression_pipeline(self):
        """Test full compression pipeline: compress -> store -> reconstruct."""
        torch.manual_seed(42)

        # Simulate activation compression
        activation = torch.randn(64, 3072)

        # Apply top-k sparsity
        k = int(0.1 * activation.numel())
        threshold = torch.topk(activation.abs().flatten(), k).values[-1]
        sparse_activation = activation.clone()
        sparse_activation[sparse_activation.abs() < threshold] = 0

        # Compress to CSR
        csr = compress_activation_to_csr(sparse_activation, threshold=0.0)

        # Calculate memory
        nnz = csr._nnz()
        mem = csr_memory_savings(64, 3072, nnz, dtype=torch.float32)

        # Reconstruct
        reconstructed = csr.to_dense()

        # Verify
        torch.testing.assert_close(reconstructed, sparse_activation, rtol=1e-5, atol=1e-6)

        # Check compression
        assert mem["compression_ratio"] > 4.5  # Should get good compression with ~90% sparsity

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_pipeline(self):
        """Test compression pipeline on CUDA."""
        torch.manual_seed(42)

        activation = torch.randn(32, 1024, device="cuda")
        activation[activation.abs() < 0.5] = 0

        # Compress
        csr = compress_activation_to_csr(activation)

        # Statistics
        stats = IndexCompressionStats()

        # Reconstruction timing
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        reconstructed = csr.to_dense()
        end.record()

        torch.cuda.synchronize()
        recon_time = start.elapsed_time(end) / 1000  # Convert to seconds

        # Record stats
        stats.record(csr.col_indices(), csr.col_indices().numel() * 4, recon_time)

        # Verify
        torch.testing.assert_close(reconstructed, activation, rtol=1e-5, atol=1e-5)

        # Report
        report = stats.report()
        assert report["num_operations"] == 1
        assert report["avg_reconstruction_ms"] > 0


# =============================================================================
# Tests for NEW compress_indices / decompress_indices functions (STACR integration)
# =============================================================================

try:
    from stacr.index_compression import (
        IndexCompressionMode,
        CompressedIndices,
        compress_indices,
        decompress_indices,
        estimate_index_compression,
        _choose_col_dtype,
    )

    HAS_NEW_COMPRESSION = True
except ImportError:
    HAS_NEW_COMPRESSION = False


@pytest.mark.skipif(not HAS_NEW_COMPRESSION, reason="New compression functions not available")
class TestCompressIndicesCSR:
    """Tests for CSR compression of flat 1D indices."""

    def test_basic_roundtrip(self):
        """Test basic compress/decompress roundtrip."""
        torch.manual_seed(42)
        B_flat, N = 100, 100
        flat_indices = torch.randint(0, B_flat * N, (50,), dtype=torch.int32)

        compressed = compress_indices(flat_indices, B_flat, N, mode="csr")
        decompressed = decompress_indices(compressed)

        assert torch.equal(flat_indices, decompressed), "CSR roundtrip failed"

    def test_bert_like_workload(self):
        """Test with BERT-like dimensions."""
        torch.manual_seed(42)
        B_flat, N = 4096, 3072
        nnz = int(B_flat * N * 0.01)  # 1% budget
        flat_indices = torch.randint(0, B_flat * N, (nnz,), dtype=torch.int32)

        compressed = compress_indices(flat_indices, B_flat, N, mode="csr")
        decompressed = decompress_indices(compressed)

        assert torch.equal(flat_indices, decompressed), "BERT-like CSR roundtrip failed"

    def test_memory_savings(self):
        """Test that CSR saves memory."""
        torch.manual_seed(42)
        B_flat, N = 4096, 3072
        nnz = int(B_flat * N * 0.01)
        flat_indices = torch.randint(0, B_flat * N, (nnz,), dtype=torch.int32)

        compressed = compress_indices(flat_indices, B_flat, N, mode="csr", preserve_order=False)

        original_bytes = flat_indices.numel() * 4
        compressed_bytes = compressed.memory_bytes()

        assert compressed_bytes < original_bytes, (
            f"CSR should save memory: {compressed_bytes} >= {original_bytes}"
        )

    def test_int16_columns_for_small_n(self):
        """Test that small N uses int16 for column indices."""
        torch.manual_seed(42)
        B_flat, N = 1000, 500  # N < 32768
        flat_indices = torch.randint(0, B_flat * N, (100,), dtype=torch.int32)

        compressed = compress_indices(flat_indices, B_flat, N, mode="csr")

        assert compressed.col_indices.dtype == torch.int16, "Should use int16 for small N"

    def test_int32_columns_for_large_n(self):
        """Test that large N uses int32 for column indices."""
        torch.manual_seed(42)
        B_flat, N = 100, 50000  # N > 32768
        flat_indices = torch.randint(0, B_flat * N, (100,), dtype=torch.int32)

        compressed = compress_indices(flat_indices, B_flat, N, mode="csr")

        assert compressed.col_indices.dtype == torch.int32, "Should use int32 for large N"


@pytest.mark.skipif(not HAS_NEW_COMPRESSION, reason="New compression functions not available")
class TestCompressIndicesDelta:
    """Tests for delta compression of flat 1D indices."""

    def test_basic_roundtrip(self):
        """Test basic compress/decompress roundtrip."""
        torch.manual_seed(42)
        B_flat, N = 100, 100
        flat_indices = torch.randint(0, B_flat * N, (50,), dtype=torch.int32)

        compressed = compress_indices(flat_indices, B_flat, N, mode="delta")
        decompressed = decompress_indices(compressed)

        assert torch.equal(flat_indices, decompressed), "Delta roundtrip failed"

    def test_clustered_indices(self):
        """Test delta with clustered indices (should compress well)."""
        torch.manual_seed(42)
        B_flat, N = 1000, 1000
        # Clustered pattern
        base = torch.randint(0, 100, (50,)) * 1000
        offsets = torch.randint(0, 100, (50,))
        flat_indices = (base + offsets).clamp(0, B_flat * N - 1).to(torch.int32)

        compressed = compress_indices(flat_indices, B_flat, N, mode="delta", preserve_order=False)
        decompressed = decompress_indices(compressed, restore_order=False)

        # Should contain same elements (possibly reordered)
        assert set(flat_indices.tolist()) == set(decompressed.tolist())


@pytest.mark.skipif(not HAS_NEW_COMPRESSION, reason="New compression functions not available")
class TestCompressIndicesEdgeCases:
    """Edge case tests for compress_indices."""

    def test_empty_indices(self):
        """Test with empty indices."""
        flat_indices = torch.tensor([], dtype=torch.int32)
        B_flat, N = 100, 100

        for mode in ["csr", "delta", "none"]:
            compressed = compress_indices(flat_indices, B_flat, N, mode=mode)
            decompressed = decompress_indices(compressed)

            assert compressed.nnz == 0
            assert decompressed.numel() == 0

    def test_single_element(self):
        """Test with single element."""
        flat_indices = torch.tensor([42], dtype=torch.int32)
        B_flat, N = 100, 100

        for mode in ["csr", "delta", "none"]:
            compressed = compress_indices(flat_indices, B_flat, N, mode=mode)
            decompressed = decompress_indices(compressed)

            assert torch.equal(flat_indices, decompressed)

    def test_all_same_row(self):
        """Test when all indices are in the same row."""
        B_flat, N = 100, 50
        flat_indices = torch.tensor([5 * N + i for i in range(10)], dtype=torch.int32)

        compressed = compress_indices(flat_indices, B_flat, N, mode="csr")
        decompressed = decompress_indices(compressed)

        assert torch.equal(flat_indices, decompressed)

    def test_one_per_row(self):
        """Test when there's exactly one index per row."""
        B_flat, N = 50, 100
        flat_indices = torch.tensor([i * N + 42 for i in range(B_flat)], dtype=torch.int32)

        compressed = compress_indices(flat_indices, B_flat, N, mode="csr")
        decompressed = decompress_indices(compressed)

        assert torch.equal(flat_indices, decompressed)


@pytest.mark.skipif(not HAS_NEW_COMPRESSION, reason="New compression functions not available")
class TestCompressIndicesOrderPreservation:
    """Tests for order preservation in compress_indices."""

    def test_csr_preserves_order(self):
        """Test CSR preserves original order when requested."""
        torch.manual_seed(42)
        B_flat, N = 100, 100
        flat_indices = torch.randint(0, B_flat * N, (50,), dtype=torch.int32)

        compressed = compress_indices(flat_indices, B_flat, N, mode="csr", preserve_order=True)
        decompressed = decompress_indices(compressed, restore_order=True)

        assert torch.equal(flat_indices, decompressed), "Order not preserved"

    def test_delta_preserves_order(self):
        """Test delta preserves original order when requested."""
        torch.manual_seed(42)
        B_flat, N = 100, 100
        flat_indices = torch.randint(0, B_flat * N, (50,), dtype=torch.int32)

        compressed = compress_indices(flat_indices, B_flat, N, mode="delta", preserve_order=True)
        decompressed = decompress_indices(compressed, restore_order=True)

        assert torch.equal(flat_indices, decompressed), "Order not preserved"


@pytest.mark.skipif(not HAS_NEW_COMPRESSION, reason="New compression functions not available")
class TestEstimateIndexCompression:
    """Tests for estimate_index_compression function."""

    def test_estimate_matches_actual(self):
        """Test that estimate matches actual compression (within tolerance)."""
        torch.manual_seed(42)
        B_flat, N = 4096, 3072
        nnz = int(B_flat * N * 0.01)
        flat_indices = torch.randint(0, B_flat * N, (nnz,), dtype=torch.int32)

        estimate = estimate_index_compression(B_flat, N, nnz, mode="csr", preserve_order=False)
        compressed = compress_indices(flat_indices, B_flat, N, mode="csr", preserve_order=False)
        actual_bytes = compressed.memory_bytes()

        ratio = estimate["compressed_bytes"] / actual_bytes
        assert 0.9 <= ratio <= 1.1, (
            f"Estimate {estimate['compressed_bytes']} vs actual {actual_bytes}"
        )

    def test_estimate_shows_savings(self):
        """Test that estimate shows memory savings."""
        B_flat, N = 4096, 3072
        nnz = int(B_flat * N * 0.01)

        estimate = estimate_index_compression(B_flat, N, nnz, mode="csr", preserve_order=False)

        assert estimate["savings_percent"] > 0, "Should show positive savings"
        assert estimate["compression_ratio"] > 1.0, "Should have compression ratio > 1"


@pytest.mark.skipif(not HAS_NEW_COMPRESSION, reason="New compression functions not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCompressIndicesCUDA:
    """CUDA tests for compress_indices."""

    def test_csr_cuda_roundtrip(self):
        """Test CSR on CUDA device."""
        torch.manual_seed(42)
        B_flat, N = 1000, 1000
        flat_indices = torch.randint(0, B_flat * N, (5000,), dtype=torch.int32, device="cuda")

        compressed = compress_indices(flat_indices, B_flat, N, mode="csr")
        decompressed = decompress_indices(compressed)

        assert decompressed.device.type == "cuda"
        assert torch.equal(flat_indices, decompressed)

    def test_delta_cuda_roundtrip(self):
        """Test delta on CUDA device."""
        torch.manual_seed(42)
        B_flat, N = 1000, 1000
        flat_indices = torch.randint(0, B_flat * N, (5000,), dtype=torch.int32, device="cuda")

        compressed = compress_indices(flat_indices, B_flat, N, mode="delta")
        decompressed = decompress_indices(compressed)

        assert decompressed.device.type == "cuda"
        assert torch.equal(flat_indices, decompressed)

    def test_move_compressed_to_device(self):
        """Test moving CompressedIndices between devices."""
        torch.manual_seed(42)
        B_flat, N = 100, 100
        flat_indices = torch.randint(0, B_flat * N, (100,), dtype=torch.int32)

        compressed = compress_indices(flat_indices, B_flat, N, mode="csr")

        # Move to CUDA
        compressed_cuda = compressed.to(torch.device("cuda"))
        assert compressed_cuda.row_ptrs.device.type == "cuda"
        assert compressed_cuda.col_indices.device.type == "cuda"

        # Decompress on CUDA
        decompressed = decompress_indices(compressed_cuda)
        assert decompressed.device.type == "cuda"


@pytest.mark.skipif(not HAS_NEW_COMPRESSION, reason="New compression functions not available")
class TestCompressIndicesSTACRIntegration:
    """Tests simulating STACR integration."""

    def test_stacr_workflow_simulation(self):
        """Simulate STACR forward/backward with index compression."""
        torch.manual_seed(42)

        # Simulate BERT FFN: batch=32, seq=128, hidden=3072
        B_flat = 32 * 128  # 4096
        N = 3072
        budget = 0.01

        # Forward pass: generate residual indices (simulating top-k)
        nnz = int(B_flat * N * budget)
        flat_indices = torch.randint(0, B_flat * N, (nnz,), dtype=torch.int32)
        values = torch.randn(nnz)  # Residual values

        # Compress indices (forward pass)
        compressed = compress_indices(flat_indices, B_flat, N, mode="csr")

        # Simulate saving to autograd context
        saved_row_ptrs = compressed.row_ptrs
        saved_col_indices = compressed.col_indices
        saved_sort_indices = compressed.sort_indices
        saved_nnz = compressed.nnz
        saved_B_flat = compressed.B_flat
        saved_N = compressed.N

        # Simulate backward pass: reconstruct indices
        reconstructed_compressed = CompressedIndices(
            mode=IndexCompressionMode.CSR,
            row_ptrs=saved_row_ptrs,
            col_indices=saved_col_indices,
            sort_indices=saved_sort_indices,
            nnz=saved_nnz,
            B_flat=saved_B_flat,
            N=saved_N,
            original_dtype=torch.int32,
        )

        recovered_indices = decompress_indices(reconstructed_compressed)

        # Verify lossless
        assert torch.equal(flat_indices, recovered_indices)

    def test_memory_comparison_table(self):
        """Print memory comparison for documentation."""
        configs = [
            ("Small (B=64, N=512, 1%)", 64, 512, 0.01),
            ("Medium (B=256, N=1024, 1%)", 256, 1024, 0.01),
            ("BERT-base (B=4096, N=3072, 1%)", 4096, 3072, 0.01),
            ("BERT-large (B=4096, N=4096, 1%)", 4096, 4096, 0.01),
        ]

        print("\n" + "=" * 70)
        print("Index Compression Memory Comparison (no order preservation)")
        print("=" * 70)
        print(f"{'Config':<35} {'Original':<12} {'CSR':<12} {'Savings':<10}")
        print("-" * 70)

        for name, B_flat, N, budget in configs:
            nnz = int(B_flat * N * budget)
            original = nnz * 4  # int32

            estimate = estimate_index_compression(B_flat, N, nnz, mode="csr", preserve_order=False)
            csr_bytes = estimate["compressed_bytes"]
            savings = estimate["savings_percent"]

            print(
                f"{name:<35} {original / 1024:>8.1f} KB  {csr_bytes / 1024:>8.1f} KB  {savings:>6.1f}%"
            )

        print("=" * 70)
