    import tensorflow as tf

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32


    def get_dataloaders(data_dir="data"):
        train_ds = tf.keras.utils.image_dataset_from_directory(
            f"{data_dir}/train",
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="binary"
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            f"{data_dir}/val",
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="binary"
        )

        test_ds = tf.keras.utils.image_dataset_from_directory(
            f"{data_dir}/test",
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="binary",
            shuffle=False
        )

        class_names = train_ds.class_names  # ['NORMAL', 'PNEUMONIA']

        # Data augmentation (applied only to training set)
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomZoom(0.05),
            tf.keras.layers.RandomBrightness(0.1),
        ])

        train_ds = train_ds.map(lambda x, y: (augmentation(x, training=True), y))

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

        return train_ds, val_ds, test_ds, class_names