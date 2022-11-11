import argparse
import torch

def default_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--vae_ins', type=int, default=3780)
    parser.add_argument('--vae_hids0', type=int, default=2048)
    parser.add_argument('--vae_lats', type=int, default=512)
    parser.add_argument('--vae_hids1', type=int, default=1024)
    parser.add_argument('--vae_outs', type=int, default=1890)
    parser.add_argument('--edgeconv_ins', type=int, default=7560)
    parser.add_argument('--edgeconv_hids', type=int, default=4096)
    parser.add_argument('--edgeconv_outs', type=int, default=1890)
    parser.add_argument('--gat_ins', type=int, default=3780)
    parser.add_argument('--gat_hids', type=int, default=2048)
    parser.add_argument('--gat_outs', type=int, default=1890)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epsilon_cross_atten', type=float, default=0.3)
    parser.add_argument('--t', type=int, default=4)
    parser.add_argument('--epsilon_clipped', type=float, default=0.001)
    parser.add_argument('--top_k', type=int, default=64)
    parser.add_argument('--prob', type=float, default=0.333)
    parser.add_argument('--train', type=str, default='train_data')
    parser.add_argument('--val', type=str, default='val_data')
    parser.add_argument('--test', type=str, default='test_data')
    parser.add_argument('--model_save', type=str, default='model_save')
    parser.add_argument('--num_labels', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=6400)

    return parser.parse_args()