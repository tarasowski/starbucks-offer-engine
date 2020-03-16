import argparse

def init_argparser(step):
    parser = argparse.ArgumentParser(
            description=f'CLI for the {step} starbucks')
    parser.add_argument(
            'load_path', type=str, help=f'Loading files for {step}')
    parser.add_argument(
            'save_path', type=str, help=f'Saving files from {step}')
    parser.add_argument(
            '--save', type=bool, default=False, help=f'Should the files be saved?')
    args = parser.parse_args()
    load_path = args.load_path
    save_path = args.save_path
    save = args.save
    return (load_path, save_path, save)

if __name__ == '__main__':
    init_argparser('test')
