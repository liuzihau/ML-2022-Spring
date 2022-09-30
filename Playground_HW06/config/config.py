class Config:
    def __init__(self):
        self.args = {
            # base
            "model_type": "WGAN-GP",
            "batch_size": 64,
            "n_epoch": 500,
            "z_dim": 100,
            "continue_train": False,
            "workspace_dir": ".",  # define in the environment setting
        }
        if self.args['model_type'] == "GAN":
            self.args['lr'] = 1e-4
            self.args['n_critic'] = 1
        elif self.args['model_type'] == "WGAN-GP":
            self.args['lr'] = 2 * 1e-4
            self.args['n_critic'] = 10
            self.args['lambda_gp'] = 10



