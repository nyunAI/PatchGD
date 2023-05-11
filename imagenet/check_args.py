import argparse
import yaml


def _parse_args(parser, train_parser):
    args1 = train_parser.parse_args()
    if args1.config:
        with open(args1.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args2 = parser.parse_known_args()
    args = argparse.Namespace()
    print(args2)
    for key, value in vars(args2[0]).items():
        setattr(args, key, value)
    for key, value in vars(args1).items():
        setattr(args, key, value)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def print_5(**args):
    print(5)


if __name__ == '__main__':
    train_parser = argparse.ArgumentParser(
        description="Arguments for training baseline on ImageNet100")
    train_parser.add_argument('--config', default='./config.yaml', type=str, metavar='FILE',
                              help='YAML config file specifying default arguments')
    train_parser.add_argument('--root_dir', default='./', type=str)
    train_parser.add_argument('--epochs', default=2, type=int)
    train_parser.add_argument('--batch_size', default=32, type=int)
    train_parser.add_argument('--image_size', default=160, type=int)
    train_parser.add_argument('--seed', default=42, type=int)
    train_parser.add_argument('--num_classes', default=100, type=int)
    train_parser.add_argument('--num_workers', default=2, type=int)
    train_parser.add_argument('--lr', default=1e-3, type=float)
    train_parser.add_argument('--output_dir', default='./', type=str)
    train_parser.add_argument('--model_save_dir', default='./', type=str)
    train_parser.add_argument('--model_load_dir', default='', type=str)

    parser = argparse.ArgumentParser(
        description="Complete parser")
    args, args_text = _parse_args(parser=parser, train_parser=train_parser)
    print(args)
    print_5(a=10)
