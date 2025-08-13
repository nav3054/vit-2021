import os, json
import tensorflow as tf
from argparse import ArgumentParser
from data_pipeline import create_datasets
from vit import ViT


class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, total_steps):
        super().__init__()
        # store as plain python numbers for serialization 
        self.base_lr = float(base_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        total  = tf.cast(self.total_steps, tf.float32)
        base   = tf.cast(self.base_lr, tf.float32)

        # linear warmup
        warm = base * (step / tf.maximum(1.0, warmup))
        # cosine decay
        pi = tf.constant(3.14159265, dtype=tf.float32)
        cos = 0.5 * base * (1.0 + tf.cos(pi * (step - warmup) / tf.maximum(1.0, total - warmup)))
        return tf.where(step < warmup, warm, cos)

    def get_config(self):
        # must return serializable python types
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
        }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log_dir", default="logs");
    parser.add_argument("--out_dir", default="ckpts")

    # data args
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--use_mixup", action="store_true");
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--use_cutmix", action="store_true");
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)

    # model args
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--model_dim", type=int, default=192)
    parser.add_argument("--num_heads", type=int, default=3)
    parser.add_argument("--mlp_dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attn_dropout", type=float, default=0.0)

    # training args
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--base_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--use_cosine_lr", action="store_true")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", action="store_true")

    args = parser.parse_args()

    # Device - CPU/GPU --> mixed_precision
    tf.random.set_seed(0)
    if args.mixed_precision and tf.config.list_physical_devices("GPU"):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # make dirs
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # raise errors if conditions not met
    assert args.image_size % args.patch_size == 0, "[ERROR] image_size must be divisible by patch_size"
    assert args.model_dim % args.num_heads == 0, "[ERROR] model_dim must be divisible by num_heads"


    # build datasets from data pipeline
    train_ds, val_ds, test_ds, steps_per_epoch, val_steps, num_classes, train_count, val_count, test_count = create_datasets(args)

    print(f"[INFO] Samples in Train set : {train_count}")
    print(f"[INFO] Samples in Val set :   {val_count}")
    print(f"[INFO] Samples in Test set :  {test_count}")

    # build model
    model = ViT(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=num_classes,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout
    )

    model.build((None, args.image_size, args.image_size, 3))
    print("\n[MODEL SUMMARY]")
    model.summary()

    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    # cosine warmup
    lr = WarmupCosine(args.base_lr, warmup_steps, total_steps) if args.use_cosine_lr else args.base_lr

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr, 
        weight_decay=args.weight_decay, 
        beta_1=0.9, 
        beta_2=0.999)


    # apply gradient clipping if given in args
    if args.grad_clip_norm and args.grad_clip_norm > 0:
        optimizer.clipnorm = args.grad_clip_norm

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=args.label_smoothing) # loss

    metrics = [tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"),
               tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")] # metrics

    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(os.path.join(args.out_dir, "best_model.keras"), monitor="val_categorical_accuracy", mode="max", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=args.log_dir)
    ]


    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks
    )


    # evaluation of test set 
    test_results = model.evaluate(test_ds, return_dict=True)
    print("[TEST RESULTS]")
    for metric_name, value in test_results.items():
        print(f"  {metric_name}: {value:.4f}")

    # save the model
    model.save(os.path.join(args.out_dir, "final_model.keras"))
    with open(os.path.join(args.out_dir, "config.json"), "w") as f: # generate config (args) file
        json.dump(vars(args), f, indent=2)