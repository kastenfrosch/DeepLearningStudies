class Config:
    def __init__(self):

        self.epochs = 300
        self.no_runs = 3
        self.ds_batch = 256
        self.batch_size = 50
        self.noise_dim = 100
        self.num_examples_to_generate = 16
        self.buffer_size = 60000
        self.model_dir = "..\\model_files\\"

