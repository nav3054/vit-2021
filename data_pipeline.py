import math
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE
NUM_CLASSES = 100


# BASIC AUGMENTATION FUNCTIONS

def random_crop_with_padding(image, crop_size):
    pad = 4 if crop_size <= 40 else max(4, crop_size // 8)
    image = tf.pad(image, [[pad, pad], [pad, pad], [0, 0]], mode="REFLECT")
    image = tf.image.random_crop(image, size=[crop_size, crop_size, 3])
    return image

def basic_augment(image, training, image_size):
    if training:
        image = tf.image.random_flip_left_right(image)
        image = random_crop_with_padding(image, image_size)
    return image



# FUNCTIONS FOR MIXUP AND CUTMIX

def sample_beta_distribution(alpha, shape):
    g1 = tf.random.gamma(shape, alpha, 1)
    g2 = tf.random.gamma(shape, alpha, 1)
    return g1 / (g1 + g2)

def apply_mixup(images, labels, alpha):
    idx = tf.random.shuffle(tf.range(tf.shape(images)[0]))
    images2 = tf.gather(images, idx)
    labels2 = tf.gather(labels, idx)

    lam = sample_beta_distribution(alpha, [tf.shape(images)[0], 1, 1, 1])
    lam_y = tf.reshape(lam, [tf.shape(images)[0], 1])

    mixed_images = images * lam + images2 * (1.0 - lam)
    mixed_labels = labels * lam_y + labels2 * (1.0 - lam_y)
    return mixed_images, mixed_labels

def rand_bbox(h, w, lam):
    cut_ratio = tf.sqrt(1.0 - lam)
    cut_h = tf.cast(cut_ratio * tf.cast(h, tf.float32), tf.int32)
    cut_w = tf.cast(cut_ratio * tf.cast(w, tf.float32), tf.int32)

    cy = tf.random.uniform([], 0, h, dtype=tf.int32)
    cx = tf.random.uniform([], 0, w, dtype=tf.int32)

    y1 = tf.clip_by_value(cy - cut_h // 2, 0, h)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, h)
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, w)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, w)
    return y1, y2, x1, x2

def apply_cutmix(images, labels, alpha):
    idx = tf.random.shuffle(tf.range(tf.shape(images)[0]))
    images2 = tf.gather(images, idx)
    labels2 = tf.gather(labels, idx)

    lam = tf.squeeze(sample_beta_distribution(alpha, [1]), axis=0)
    h = tf.shape(images)[1]
    w = tf.shape(images)[2]

    y1, y2, x1, x2 = rand_bbox(h, w, lam)

    # build a binary mask with a zero rectangle
    yy = tf.range(h)[:, None] # [h,1]
    xx = tf.range(w)[None, :] # [1,w]
    in_box = (yy >= y1) & (yy < y2) & (xx >= x1) & (xx < x2)  
    box_mask = tf.cast(in_box, tf.float32)[..., None]         
    mask = 1.0 - box_mask                                     
    mask = tf.expand_dims(mask, axis=0)                       
    mixed_images = images * mask + images2 * (1.0 - mask)

    # adjust lambda based on actual area removed
    box_area = tf.cast((y2 - y1) * (x2 - x1), tf.float32)
    lam_adj = 1.0 - box_area / (tf.cast(h * w, tf.float32))
    mixed_labels = labels * lam_adj + labels2 * (1.0 - lam_adj)
    return mixed_images, mixed_labels


def decide_cutmix_or_mixup(use_mixup, mixup_alpha, use_cutmix, cutmix_alpha):
    def map_fn(images, labels):
        if use_mixup and use_cutmix:
            return tf.cond(   # using tf.cond because using 'if' here could cause errors
                tf.random.uniform([]) < 0.5,
                lambda: apply_mixup(images, labels, mixup_alpha),
                lambda: apply_cutmix(images, labels, cutmix_alpha),
            )
        elif use_mixup:
            return apply_mixup(images, labels, mixup_alpha)
        elif use_cutmix:
            return apply_cutmix(images, labels, cutmix_alpha)
        else:
            return images, labels
    return map_fn


# MAIN PIPELINE FUNCTIONS

def preprocess_image(image, image_size): # to float32 [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    if tf.shape(image)[0] != image_size or tf.shape(image)[1] != image_size:
        image = tf.image.resize(image, [image_size, image_size])
    return image

def train_map(image, label, image_size):
    image = preprocess_image(image, image_size)
    image = basic_augment(image, training=True, image_size=image_size)
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

def eval_map(image, label, image_size):
    image = preprocess_image(image, image_size)
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

def create_datasets(args):
    image_size = args.image_size
    batch_size = args.batch_size
    val_split = args.val_split

    # load data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    y_train = y_train.squeeze().astype("int32")
    y_test = y_test.squeeze().astype("int32")

    # train/val split
    n_train = x_train.shape[0]
    n_val = int(n_train * val_split)
    x_val = x_train[:n_val]
    y_val = y_train[:n_val]
    x_train = x_train[n_val:]
    y_train = y_train[n_val:]

    train_count = len(x_train)
    val_count = len(x_val)
    test_count = len(x_test)

    # create datasets
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds   = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_ds  = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_ds = (
        train_ds
        .shuffle(buffer_size=min(10000, x_train.shape[0]), reshuffle_each_iteration=True)
        .map(lambda im, lb: train_map(im, lb, image_size), num_parallel_calls=AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .map(decide_cutmix_or_mixup(args.use_mixup, args.mixup_alpha, args.use_cutmix, args.cutmix_alpha), num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        val_ds
        .map(lambda im, lb: eval_map(im, lb, image_size), num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .cache()
        .prefetch(AUTOTUNE)
    )
    test_ds = (
        test_ds
        .map(lambda im, lb: eval_map(im, lb, image_size), num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .cache()
        .prefetch(AUTOTUNE)
    )

    steps_per_epoch = math.floor(len(x_train) / batch_size)
    val_steps = math.ceil(len(x_val) / batch_size)

    return train_ds, val_ds, test_ds, steps_per_epoch, val_steps, NUM_CLASSES, train_count, val_count, test_count