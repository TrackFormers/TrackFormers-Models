import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
from tensorflow import keras
from einops import repeat


class TransformerSetEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, num_induce, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.num_induce = num_induce
        # Multi-head attention layer
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        # Dense projection layers
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True
        
    def build(self, input_shape):
        # Initialize inducing points as trainable weights
        self.inducing_points = self.add_weight(
            shape=(self.num_induce, self.embed_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="Inducing_Points",
            dtype=self.compute_dtype)

    def call(self, inputs):
        batch_size = inputs.shape[0]
        # Repeat inducing points for each batch
        inducing_points = repeat(self.inducing_points, "n d -> b n d", b=batch_size)
        # Apply multi-head attention
        attention_output = self.attention(query=inducing_points, value=inputs, key=inputs)
        proj_input = self.layernorm_1(inducing_points + attention_output)
        # Project through dense layers
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dense_dim)


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        # Multi-head attention layer
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        # Dense projection layers
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs):
        # Apply multi-head attention
        attention_output = self.attention(query=inputs, value=inputs, key=inputs)
        proj_input = self.layernorm_1(inputs + attention_output)
        # Project through dense layers
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "dense_dim": self.dense_dim,
                "num_heads": self.num_heads,
            }
        )
        return config


# We need positional embedding for the inputs into the decoder
class PositionalEmbedding(layers.Layer):
    def __init__(self, track_length, embed_dim, **kwargs):
        super().__init__(**kwargs)
        # Embedding layer for positional information
        self.position_embeddings = layers.Embedding(
            input_dim=track_length, output_dim=embed_dim
        )
        self.track_length = track_length
        self.embed_dim = embed_dim

    def call(self, inputs):
        # Generate positions based on input length
        length = tf.shape(inputs)[-2]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        # Add positional embeddings to inputs
        return inputs + embedded_positions

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "track_length": self.track_length,
                "embed_dim": self.embed_dim,
            }
        )
        return config

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        # Self and encoder attention layers
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        # Dense projection layers
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.add = layers.Add()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs):
        encoder_outputs = tf.cast(encoder_outputs, dtype=self.compute_dtype)
        # Create causal mask for decoder
        causal_mask = self.get_causal_attention_mask(inputs)

        # Self-attention with causal mask
        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)
                
        # Encoder-decoder attention
        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        # Project through dense layers
        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, max_size_tracks = input_shape[0], input_shape[1]
        i = tf.range(max_size_tracks)[:, tf.newaxis]
        j = tf.range(max_size_tracks)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

def create_model(embed_dim=128, dense_dim=2048, num_heads=16, num_induce=300, 
                max_size_hits=450, max_size_tracks=30, batch_size=64, num_stacks=2):
    """
    Constructs the Transformer model architecture for track decoding.

    Args:
        embed_dim (int): Dimension of the embedding space.
        dense_dim (int): Dimension of the dense layers.
        num_heads (int): Number of attention heads.
        num_induce (int): Number of inducing points for the TransformerSetEncoder.
        max_size_hits (int): Maximum number of hits per track.
        max_size_tracks (int): Maximum number of tracks.
        batch_size (int): Batch size for training.
        num_stacks (int): Number of stacked Transformer encoder and decoder layers.

    Returns:
        keras.Model: Compiled Transformer model ready for training.
    """
    # Set mixed precision policy
    mixed_precision.set_global_policy('float32')
    
    # Define encoder input
    encoder_inputs = keras.Input(
        shape=(max_size_hits, 3,), 
        batch_size=batch_size, 
        name="encoder_inputs"
    )
    encoder_inputs_masked = layers.Masking(mask_value=0.0)(encoder_inputs)
    x = layers.Dense(embed_dim, activation=None)(encoder_inputs_masked)
    encoder_outputs = TransformerSetEncoder(
        embed_dim, dense_dim, num_heads, num_induce
    )(x)
    
    # Add stacked Transformer encoders if specified
    if num_stacks > 1:
        encoder_outputs = layers.Masking(mask_value=0.0)(encoder_outputs)
        for _ in range(num_stacks - 1):
            encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(encoder_outputs)
    encoder = keras.Model(encoder_inputs, encoder_outputs)

    # Define decoder input
    decoder_inputs = keras.Input(
        shape=(max_size_tracks, 3,), 
        batch_size=batch_size, 
        name="decoder_inputs"
    )
    decoder_inputs_masked = layers.Masking(mask_value=0.0)(decoder_inputs)
    encoded_seq_inputs = keras.Input(
        shape=(None, embed_dim), 
        name="decoder_state_inputs"
    )
    x = layers.Dense(embed_dim, activation=None)(decoder_inputs_masked)
    x = PositionalEmbedding(max_size_tracks, embed_dim)(x)
    
    # Add stacked Transformer decoders if specified
    for _ in range(num_stacks - 1):
        x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoded_seq_inputs)
    decoder_outputs = layers.Dense(3, activation="sigmoid")(x)
    decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

    # Connect encoder and decoder to form the Transformer model
    decoder_outputs = decoder([decoder_inputs, encoder_outputs])
    transformer = keras.Model(
        [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
    )

    return transformer
