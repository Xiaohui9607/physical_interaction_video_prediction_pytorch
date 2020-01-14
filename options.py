import argparse
import os
import torch

# pylint: disable=C0103,C0301,R0903,W0622


class Options():
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--data_dir', default='', help='directory containing data.')
        self.parser.add_argument('--channels', type=int, default=3, help='# channel of input')
        self.parser.add_argument('--height', type=int, default=64, help='height of image')
        self.parser.add_argument('--width', type=int, default=64, help='width of image')

        # flags.DEFINE_string('output_dir', OUT_DIR, 'directory for model checkpoints.')
        self.parser.add_argument('--output_dir', default='', help='directory for model checkpoints.')
        # flags.DEFINE_string('event_log_dir', OUT_DIR, 'directory for writing summary.')
        self.parser.add_argument('--event_log_dir', default='', help='directory for writing summary.')
        # flags.DEFINE_integer('num_iterations', 100000, 'number of training iterations.')
        self.parser.add_argument('--num_iterations', type=int, default=100000, help='number of training iterations.')
        # flags.DEFINE_string('pretrained_model', '',
        self.parser.add_argument('--pretrained_model', default='', help='filepath of a pretrained model to initialize from.')
        # flags.DEFINE_integer('sequence_length', 10,
        #                      'sequence length, including context frames.')
        self.parser.add_argument('--sequence_length', type=int, default=10, help='sequence length, including context frames.')
        # flags.DEFINE_integer('context_frames', 2, '# of frames before predictions.')
        self.parser.add_argument('--context_frames', type=int, default=2, help= '# of frames before predictions.')
        # flags.DEFINE_integer('use_state', 1,
        #                      'Whether or not to give the state+action to the model')
        self.parser.add_argument('--use_state',  default=True, action='store_true', help='Whether or not to give the state+action to the model')
        # flags.DEFINE_string('model', 'CDNA',
        #                     'model architecture to use - CDNA, DNA, or STP')
        self.parser.add_argument('--model', default='CDNA', help='model architecture to use - CDNA, DNA, or STP')
        #
        # flags.DEFINE_integer('num_masks', 10,
        #                      'number of masks, usually 1 for DNA, 10 for CDNA, STN.')
        self.parser.add_argument('--num_masks', type=int, default=10, help='number of masks, usually 1 for DNA, 10 for CDNA, STN.')
        # flags.DEFINE_float('schedsamp_k', 900.0,
        #                    'The k hyperparameter for scheduled sampling,'
        #                    '-1 for no scheduled sampling.')
        self.parser.add_argument('--schedsamp_k', type=float, default=900.0, help='The k hyperparameter for scheduled sampling, -1 for no scheduled sampling.')
        # flags.DEFINE_float('train_val_split', 0.95,
        #                    'The percentage of files to use for the training set,'
        #                    ' vs. the validation set.')
        self.parser.add_argument('--train_val_split', type=float, default=0.95, help='The percentage of files to use for the training set, vs. the validation set.')
        # flags.DEFINE_integer('batch_size', 32, 'batch size for training')
        self.parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        # flags.DEFINE_float('learning_rate', 0.001,
        #                    'the base learning rate of the generator')
        self.parser.add_argument('--learning_rate', type=float, default=0.001, help='the base learning rate of the generator')
        self.opt = None

    def parse(self):
        """ Parse Arguments.
        """
        self.opt = self.parser.parse_args()

        return self.opt

if __name__ == '__main__':
    options = Options().parse()

