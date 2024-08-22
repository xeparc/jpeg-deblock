import config
from builders import *



if __name__ == "__main__":

    cfg = config.default_config()

    train_loader    = build_dataloader(cfg, "train")
    val_loader      = build_dataloader(cfg, "val")
    test_loader     = build_dataloader(cfg, "test")

    fdnet           = build_spectral_net(cfg)
    print(fdnet)