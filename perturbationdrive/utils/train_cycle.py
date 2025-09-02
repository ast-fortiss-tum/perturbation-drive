from perturbationdrive import train_cycle_gan

train_cycle_gan.train(
    input_dir="./relative/path/to/folder",
    output_dir="./relative/path/to/folder",
    image_extension_input="jpg",
    image_extension_output="jpg",
    buffer_size=100,
    batch_size=2,
    early_stop_patience=None,
    epochs=50,
    steps_per_epoch=300,
)