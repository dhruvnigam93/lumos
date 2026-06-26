import pytest
import torch

from lumos.model import LUMOS, MultiHeadAttention
from lumos.model.layers import EncoderLayer, DecoderLayer, TransformerEncoder, TransformerDecoder
from lumos.model.positional import get_positional_embedding, NoEmbedding


class TestMultiHeadAttention:
    def test_output_shape(self):
        mha = MultiHeadAttention(d_q=64, d_kv=64, d_attn=64, n_heads=4)
        Q = torch.randn(2, 10, 64)
        K = V = torch.randn(2, 20, 64)
        out, weights = mha(Q, K, V)
        assert out.shape == (2, 10, 64)
        assert weights.shape == (2, 4, 10, 20)

    def test_cross_attention_dims(self):
        mha = MultiHeadAttention(d_q=128, d_kv=64, d_attn=128, n_heads=8)
        Q = torch.randn(4, 5, 128)
        K = V = torch.randn(4, 15, 64)
        out, weights = mha(Q, K, V)
        assert out.shape == (4, 5, 128)


class TestTransformerBlocks:
    def test_encoder_layer(self):
        layer = EncoderLayer(d_model=64, n_heads=4, dim_ff=128)
        x = torch.randn(2, 10, 64)
        out = layer(x)
        assert out.shape == x.shape

    def test_decoder_layer(self):
        layer = DecoderLayer(d_model=64, n_heads=4, dim_ff=128)
        x = torch.randn(2, 5, 64)
        enc_out = torch.randn(2, 10, 64)
        out = layer(x, enc_out)
        assert out.shape == x.shape

    def test_encoder_stack(self):
        encoder = TransformerEncoder(d_model=64, n_heads=4, dim_ff=128, n_layers=3)
        out = encoder(torch.randn(2, 10, 64))
        assert out.shape == (2, 10, 64)

    def test_decoder_stack(self):
        decoder = TransformerDecoder(d_model=64, n_heads=4, dim_ff=128, n_layers=2)
        out = decoder(torch.randn(2, 5, 64), torch.randn(2, 10, 64))
        assert out.shape == (2, 5, 64)


class TestPositionalEmbeddings:
    @pytest.mark.parametrize("pe_type", ["none", "sinusoidal", "learned", "tape", "abs"])
    def test_factory(self, pe_type):
        pe = get_positional_embedding(pe_type, d_model=64, max_len=100)
        assert pe is not None

    def test_sinusoidal_embedding(self):
        pe = get_positional_embedding("sinusoidal", d_model=64, max_len=100)
        emb = pe.get_embedding(0, 10)
        assert emb.shape == (1, 10, 64)

    def test_no_embedding(self):
        pe = NoEmbedding()
        x = torch.randn(2, 10, 64)
        assert torch.equal(pe(x), x)
        assert pe.get_embedding(0, 5) == 0.0


class TestLUMOS:
    @pytest.fixture
    def small_model(self):
        return LUMOS(
            dim_activity_raw=8,
            dim_static_raw=4,
            dim_event_context_raw=16,
            dim_activity_embed=32,
            dim_static_embed=16,
            dim_event_context_embed=32,
            d_model=64,
            n_heads=4,
            dim_ff=128,
            n_enc_layers=2,
            n_dec_layers=1,
            seq_len_hist=20,
            seq_len_future=5,
            prediction_mode="aggregated",
            n_targets=3,
            position_embedding_type="sinusoidal",
        )

    def test_forward_aggregated(self, small_model):
        out = small_model(
            torch.randn(2, 20, 8),
            torch.randn(2, 20, 16),
            torch.randn(2, 4),
            torch.randn(2, 5, 16),
        )
        assert out.shape == (2, 3)

    def test_forward_sequence(self):
        model = LUMOS(
            dim_activity_raw=8,
            dim_static_raw=4,
            dim_event_context_raw=16,
            dim_activity_embed=32,
            dim_static_embed=16,
            dim_event_context_embed=32,
            d_model=64,
            n_heads=4,
            dim_ff=128,
            n_enc_layers=2,
            n_dec_layers=1,
            seq_len_hist=20,
            seq_len_future=5,
            prediction_mode="sequence",
            n_targets=3,
        )
        out = model(
            torch.randn(2, 20, 8),
            torch.randn(2, 20, 16),
            torch.randn(2, 4),
            torch.randn(2, 5, 16),
        )
        assert out.shape == (2, 5, 3)

    def test_paper_config(self):
        model = LUMOS.from_paper_config()
        out = model(
            torch.randn(1, 360, 13),
            torch.randn(1, 360, 32),
            torch.randn(1, 9),
            torch.randn(1, 7, 32),
        )
        assert out.shape == (1, 5)

    def test_historical_user_embedding(self, small_model):
        emb = small_model.historical_user_embedding(
            torch.randn(2, 20, 8),
            torch.randn(2, 20, 16),
            torch.randn(2, 4),
        )
        assert emb.shape == (2, 64)

    @pytest.mark.parametrize("reduction", ["mean", "max", "expavg", "last"])
    def test_embedding_reductions(self, small_model, reduction):
        emb = small_model.historical_user_embedding(
            torch.randn(2, 20, 8),
            torch.randn(2, 20, 16),
            torch.randn(2, 4),
            reduction=reduction,
        )
        assert emb.shape == (2, 64)

    def test_event_context_embedding(self, small_model):
        emb = small_model.event_context_embedding(torch.randn(4, 16))
        assert emb.shape == (4, 32)

    def test_gradient_flow(self, small_model):
        out = small_model(
            torch.randn(2, 20, 8),
            torch.randn(2, 20, 16),
            torch.randn(2, 4),
            torch.randn(2, 5, 16),
        )
        loss = out.sum()
        loss.backward()
        for name, param in small_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
