import sys
from mg import trainer

if __name__ == '__main__':
    args = sys.argv[1:]
    if args[0] == 'multiple':
        args = args[1:]
        trainer.multi_gpu_trainer(args[0], args[1], int(args[2]))
    elif args[0] == 'multiple-full':
        args = args[1:]
        trainer.multi_gpu_trainer_full(args[0], args[1], int(args[2]))
    elif args[0] == 'single':
        trainer.single_gpu_trainer(args[1])
    elif args[0] == 'single-full':
        trainer.single_gpu_trainer_full(args[1])
    else:
        raise Exception('Unaccepted training mode')
