# format: arg_name, type, default, help
cmdline_args =  [
    ["lr",                  float,  1e-4,       "learning rate"],
    ["seed",                int,    2024,       "random seed"],
    ["epoch",               int,    200,        "num of epochs"],
    ["hidden-feats",        int,    16,         "num of hidden layer dimensions"],
    ["dataset",             str,    "cora",     "learning rate"],
    ["device",              str,    "cpu",      "device that model runs on"],
    ["split-ratio",         str,    "85:5:10",    "the ratio of splitting edges into train edges, valid edges and test edges"]
]