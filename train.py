import argparse
import os
import os.path

import numpy as np
import torch as t
from torch.optim import Adam

import pickle

from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from model.rvae import RVAE

if __name__ == "__main__":

    if not os.path.exists('data/word_embeddings.npy'):
        raise FileNotFoundError("word embeddings file was't found")

    parser = argparse.ArgumentParser(description='RVAE')
    parser.add_argument('--num-iterations', type=int, default=120000, metavar='NI',
                        help='num iterations (default: 120000)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                        help='batch size (default: 32)')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--learning-rate', type=float, default=0.00005, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DR',
                        help='dropout (default: 0.3)')
    parser.add_argument('--use-trained', type=bool, default=False, metavar='UT',
                        help='load pretrained model (default: False)')
    parser.add_argument('--ce-result', default='', metavar='CE',
                        help='ce result path (default: '')')
    parser.add_argument('--kld-result', default='', metavar='KLD',
                        help='ce result path (default: '')')

    args = parser.parse_args()

    batch_loader = BatchLoader('')
    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size)

    rvae = RVAE(parameters)
    if args.use_trained and os.path.exists('trained_RVAE'):
        rvae.load_state_dict(t.load('trained_RVAE'))
    if args.use_cuda:
        rvae = rvae.cuda()

    optimizer = Adam(rvae.learnable_parameters(), args.learning_rate)

    train_step = rvae.trainer(optimizer, batch_loader)
    validate = rvae.validater(batch_loader)

    ce_result = []
    kld_result = []
    start_iteration = 0

    if args.use_trained and os.path.exists('breakpoint.pkl'):
        print('Restoring from snapshot.pkl ...')

        # Getting back the objects:
        with open('breakpoint.pkl', 'rb') as f:
            ce_result, kld_result, start_iteration = pickle.load(f)

        start_iteration += 1

    print('Iteration starts at', start_iteration)

    for iteration in range(start_iteration, args.num_iterations):

        cross_entropy, kld, coef, unk_count, non_unk_count, train_word_sample, train_chars_sample, train_target = train_step(iteration, args.batch_size, args.use_cuda, args.dropout)

        if iteration % 5 == 0:
            print('\n')
            print('------------TRAIN-------------')
            print('----------ITERATION-----------')
            print(iteration)
            print('--------CROSS-ENTROPY---------')
            print(cross_entropy.data.cpu().numpy()[0])
            print('-------------KLD--------------')
            print(kld.data.cpu().numpy()[0])
            print('-----------KLD-coef-----------')
            print(coef)
            print('-------UNK/NON-UNK COUNT------')
            print(unk_count, non_unk_count)
            print('------------------------------')

        if iteration % 10 == 0:
            cross_entropy, kld, unk_count, non_unk_count = validate(args.batch_size, args.use_cuda)

            cross_entropy = cross_entropy.data.cpu().numpy()[0]
            kld = kld.data.cpu().numpy()[0]

            print('\n')
            print('------------VALID-------------')
            print('--------CROSS-ENTROPY---------')
            print(cross_entropy)
            print('-------------KLD--------------')
            print(kld)
            print('-------UNK/NON-UNK COUNT------')
            print(unk_count, non_unk_count)
            print('------------------------------')

            ce_result += [cross_entropy]
            kld_result += [kld]

        if iteration % 20 == 0:
            train_data_sample = train_word_sample[0].data.numpy()
            train_data_sample2 = [ x for x in np.flip(train_data_sample, 0) if x != 9999 ]
            train_data_sample_original = batch_loader.decode_words(train_data_sample)
            train_data_sample_sentence = batch_loader.decode_words(train_data_sample2)

            seed = np.random.normal(size=[1, parameters.latent_variable_size])

            # sample, sample_ce, _, _, _ = rvae.sample(batch_loader, 50, seed, args.use_cuda)
            # sample2, sample2_ce, sample2_len, _, _ = rvae.sample2(batch_loader, 50, args.use_cuda, 'please play the jazz music')
            # sample3, sample3_ce, sample3_len, _, _ = rvae.sample2(batch_loader, 50, args.use_cuda, 'i really want to hear some jazz can you play some')
            sample4, sample4_ce, sample4_len, unk_count, non_unk_count = rvae.sample3(batch_loader, 50, args.use_cuda, train_word_sample, train_chars_sample, train_target)

            print('\n')
            print('------------SAMPLE------------')
            print('------------------------------')
            # print(sample, sample_ce)
            # print(sample2, '-', sample2_ce, ',', sample2_len)
            # print(sample3, '-', sample3_ce, ',', sample3_len)
            # print('------------TRAIN DATA SAMPLE------------')
            print('>>>> INPUT ORIGINAL')
            print(train_data_sample_original)
            print('>>>> INPUT PROCESSED')
            print(train_data_sample_sentence)
            print('>>>> OUTPUT')
            print(sample4, '-', sample4_ce, ',', sample4_len)
            print('-------UNK/NON-UNK COUNT------')
            print(unk_count, non_unk_count)
            print('------------------------------')

        if iteration % 300 == 0:
            print('Saving model data and iteration as', iteration)
            t.save(rvae.state_dict(), 'trained_RVAE.temp')
            # obj0, obj1, obj2 are created here...

            # Saving the objects:
            with open('breakpoint.pkl.temp', 'wb') as f:
                pickle.dump([ce_result, kld_result, iteration], f)

            os.rename('trained_RVAE.temp', 'trained_RVAE')
            os.rename('breakpoint.pkl.temp', 'breakpoint.pkl')

    t.save(rvae.state_dict(), 'trained_RVAE')
    print('Removing breakpoint.pkl ...')
    os.remove('trained_RVAE')
    os.remove('snapshot.pkl')

    np.save('ce_result_{}.npy'.format(args.ce_result), np.array(ce_result))
    np.save('kld_result_npy_{}'.format(args.kld_result), np.array(kld_result))
