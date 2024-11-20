# format: arg_name, type, default, help
cmdline_args =  [
    ["lr",                  float,  1e-4,       "learning rate"],
    ["epoch",               int,    200,        "num of epochs"],
    ["num-features",        int,    1377,       "input feature dimensions"],
    ["num-hidden",          int,    16,         "num of hidden layer dimensions"],
    ["dataset",             str,    "cora",     "learning rate"],
    ["device",              str,    "cpu",      "device that model runs on"],
]