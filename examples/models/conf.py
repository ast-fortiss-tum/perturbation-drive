class Conf:
    def __init__(self):
        self.training_patience = 20

        self.training_batch_size = 128

        self.training_default_epochs = 200

        self.training_default_aug_mult = 1

        self.training_default_aug_percent = 0.0

        self.image_width = 320
        self.image_height = 240
        self.image_depth = 3

        self.row = self.image_height
        self.col = self.image_width
        self.ch = self.image_depth

        # when we wish to try training for steering and throttle:
        self.num_outputs = 2

        # when steering alone:
        # num_outputs = 1

        self.throttle_out_scale = 1.0
