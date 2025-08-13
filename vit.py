import tensorflow as tf

class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_size: int, model_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.model_dim = model_dim
        self.proj = tf.keras.layers.Conv2D(
            filters=model_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid"
        )

    def call(self, x):
        x = self.proj(x)
        shape = tf.shape(x)
        B = shape[0]
        Hp = shape[1]
        Wp = shape[2]
        D = x.shape[-1]
        x = tf.reshape(x, [B, Hp * Wp, D])
        return x


# Transformer Block class
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads, mlp_dim, dropout=0.0, attn_dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=model_dim // num_heads,
            dropout=attn_dropout
        )
        self.drop1 = tf.keras.layers.Dropout(dropout)

        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_dim, activation=tf.keras.activations.gelu),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(model_dim),
        ])
        self.drop2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        # self-attention
        y = self.norm1(x)
        y = self.attn(y, y, training=training)
        y = self.drop1(y, training=training)
        x = x + y  # residual

        # mlp
        y = self.norm2(x)
        y = self.mlp(y, training=training)
        y = self.drop2(y, training=training)
        x = x + y  # residual
        return x



# Main ViT model class
class ViT(tf.keras.Model):
    def __init__(
            self,
            image_size,
            patch_size,
            num_classes,
            model_dim,
            num_layers,
            num_heads,
            mlp_dim,
            dropout=0.0,
            attn_dropout=0.0,
            **kwargs):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout

        # layers
        self.patch_embed = PatchEmbedding(patch_size=patch_size, model_dim=model_dim)
        self.blocks = [
            TransformerBlock(model_dim, num_heads, mlp_dim, dropout, attn_dropout)
            for _ in range(num_layers)
        ]
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.head = tf.keras.layers.Dense(num_classes, dtype="float32")

        # these depend on number of patches, so create them in build()
        self.cls_token = None
        self.pos_embed = None

    def build(self, input_shape):
        num_patches = (self.image_size // self.patch_size) ** 2
        d = self.model_dim


        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, d),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )

        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(1, num_patches + 1, d),   # learnable positional embeddings for [CLS] + patches: shape [1, 1+N, D]
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    
    def call(self, x, training=False):
        
        # Shape -> x: [B, H, W, C]
        x = self.patch_embed(x) # [B, N, D]
        B = tf.shape(x)[0]

        # add cls token at the start
        cls = tf.repeat(self.cls_token, repeats=B, axis=0) # [B, 1, D]
        x = tf.concat([cls, x], axis=1) # [B, 1+N, D]

        # add positional embeddings
        x = x + self.pos_embed # [B, 1+N, D]

        # N transformer blocks from num_
        for blk in self.blocks:
            x = blk(x, training=training)

        # final norm, take CLS, predict logits
        x = self.norm(x)
        cls_out = x[:, 0] # [B, D]
        logits = self.head(cls_out) # [B, num_classes]
        return logits

    def get_config(self):
        # for generating the config file
        return {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "num_classes": self.num_classes,
            "model_dim": self.model_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout,
            "attn_dropout": self.attn_dropout,
        }
