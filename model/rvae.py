import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .decoder import Decoder
from .encoder import Encoder

from selfModules.embedding import Embedding
from utils.functional import kld_coef, parameters_allocation_check, fold

import matplotlib.pyplot as plt

class RVAE(nn.Module):
    def __init__(self, params):
        super(RVAE, self).__init__()

        self.params = params

        self.embedding = Embedding(self.params, '')

        self.encoder = Encoder(self.params)

        self.context_to_mu = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)
        self.context_to_logvar = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)

        self.decoder = Decoder(self.params)

    def forward(self, drop_prob,
                encoder_word_input=None, encoder_character_input=None,
                decoder_word_input=None, decoder_character_input=None,
                z=None, initial_state=None):
        """
        :param encoder_word_input: An tensor with shape of [batch_size, seq_len] of Long type
        :param encoder_character_input: An tensor with shape of [batch_size, seq_len, max_word_len] of Long type
        :param decoder_word_input: An tensor with shape of [batch_size, max_seq_len + 1] of Long type
        :param initial_state: initial state of decoder rnn in order to perform sampling

        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout

        :param z: context if sampling is performing

        :return: unnormalized logits of sentence words distribution probabilities
                    with shape of [batch_size, seq_len, word_vocab_size]
                 final rnn state with shape of [num_layers, batch_size, decoder_rnn_size]
        """

        # assert parameters_allocation_check(self), \
        #     'Invalid CUDA options. Parameters should be allocated in the same memory'
        use_cuda = self.embedding.word_embed.weight.is_cuda

        # assert z is None and fold(lambda acc, parameter: acc and parameter is not None,
        #                           [encoder_word_input, encoder_character_input, decoder_word_input],
        #                           True) \
        #     or (z is not None and decoder_word_input is not None), \
        #     "Invalid input. If z is None then encoder and decoder inputs should be passed as arguments"

        if z is None:
            ''' Get context from encoder and sample z ~ N(mu, std)
            '''
            [batch_size, _] = encoder_word_input.size()

            encoder_input = self.embedding(encoder_word_input, encoder_character_input)

            context = self.encoder(encoder_input)

            mu = self.context_to_mu(context)
            logvar = self.context_to_logvar(context)
            std = t.exp(0.5 * logvar)

            z = Variable(t.randn([batch_size, self.params.latent_variable_size]))
            if use_cuda:
                z = z.cuda()

            z = z * std + mu

            kld = (-0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1)).mean().squeeze()
        else:
            kld = None

        decoder_input = self.embedding.word_embed(decoder_word_input)
        out, final_state = self.decoder(decoder_input, z, drop_prob, initial_state)

        return out, final_state, kld

    def learnable_parameters(self):

        # word_embedding is constant parameter thus it must be dropped from list of parameters for optimizer
        return [p for p in self.parameters() if p.requires_grad]

    def trainer(self, optimizer, batch_loader):
        def train(i, batch_size, use_cuda, dropout):
            input = batch_loader.next_batch(batch_size, 'train')
            input = [Variable(t.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]

            [encoder_word_input, encoder_character_input, decoder_word_input, decoder_character_input, target] = input
            target_original = target.clone()

            logits, _, kld = self(dropout,
                                  encoder_word_input, encoder_character_input,
                                  decoder_word_input, decoder_character_input,
                                  z=None)

            logits = logits.view(-1, self.params.word_vocab_size)
            target = target.view(-1)

            # # ------------
            # inspect_word_idx = target[5].data.numpy()[0]
            # target_len = target.size()[0]
            # logits_target_distributions = logits.clone()
            # target_distributions = np.zeros((target_len, self.params.word_vocab_size))
            # for i in range(target_len):
            #     target_distributions[i, target[i].data.numpy()[0]] = 1.0
            # np.reshape(target_distributions, (-1))
            #
            # # softmax distribution for 5th word in the full list
            # fifth = t.nn.functional.softmax(logits_target_distributions).data.numpy()[5]
            #
            # # plot
            # f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            # f.suptitle('softmax vs target: softmax on 5th word ' + str(np.sum(fifth)) + ', ' + batch_loader.decode_word(inspect_word_idx))
            #
            # ax1.plot(fifth)
            # ax1.set_title('softmax')
            # ax2.plot(target_distributions[5])
            # ax2.set_title('target')
            # plt.show()
            # # ------------

            cross_entropy = F.cross_entropy(logits, target)
            loss = 79 * cross_entropy + kld_coef(i) * kld

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return cross_entropy, kld, kld_coef(i), encoder_word_input[:1], encoder_character_input[:1], target_original[:1]

        return train

    def validater(self, batch_loader):
        def validate(batch_size, use_cuda):
            input = batch_loader.next_batch(batch_size, 'valid')
            input = [Variable(t.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if use_cuda else var for var in input]

            [encoder_word_input, encoder_character_input, decoder_word_input, decoder_character_input, target] = input

            logits, _, kld = self(0.,
                                  encoder_word_input, encoder_character_input,
                                  decoder_word_input, decoder_character_input,
                                  z=None)

            logits = logits.view(-1, self.params.word_vocab_size)
            target = target.view(-1)

            cross_entropy = F.cross_entropy(logits, target)

            return cross_entropy, kld

        return validate

    def sample(self, batch_loader, seq_len, seed, use_cuda, target_word_tensor=None, use_max=False, train_target=None):
        if type(seed) is not t.autograd.variable.Variable:
            seed = Variable(t.from_numpy(seed).float())
        if use_cuda:
            seed = seed.cuda()

        decoder_word_input_np, decoder_character_input_np = batch_loader.go_input(1)

        decoder_word_input = Variable(t.from_numpy(decoder_word_input_np).long())
        decoder_character_input = Variable(t.from_numpy(decoder_character_input_np).long())

        if use_cuda:
            decoder_word_input, decoder_character_input = decoder_word_input.cuda(), decoder_character_input.cuda()

        result = ''

        initial_state = None
        logits_all = None
        cross_entropy = 0
        effective_len = 0
        found_end = False
        total_iteration_steps = seq_len if train_target is None else train_target.size()[1]

        for i in range(total_iteration_steps):
            logits, initial_state, _ = self(0., None, None,
                                            decoder_word_input, decoder_character_input,
                                            seed, initial_state)

            logits = logits.view(-1, self.params.word_vocab_size)
            logits_all = t.cat((logits_all, logits), 0) if logits_all is not None else logits

            prediction = F.softmax(logits)

            if use_max:
                word = batch_loader.sample_word(prediction.data.cpu().numpy()[-1])
            else:
                word = batch_loader.sample_word_from_distribution(prediction.data.cpu().numpy()[-1])

            if word == batch_loader.end_token:
                found_end = True

            if train_target is None and found_end:
                break

            if not found_end:
                result += ' ' + word
                effective_len += 1

            decoder_word_input_np = np.array([[batch_loader.word_to_idx[word]]])
            decoder_character_input_np = np.array([[batch_loader.encode_characters(word)]])

            decoder_word_input = Variable(t.from_numpy(decoder_word_input_np).long())
            decoder_character_input = Variable(t.from_numpy(decoder_character_input_np).long())

            if use_cuda:
                decoder_word_input, decoder_character_input = decoder_word_input.cuda(), decoder_character_input.cuda()

        if train_target is not None:
            # print("***********", seq_len, total_iteration_steps, logits_all.size(), train_target.size())
            cross_entropy = F.cross_entropy(logits_all, train_target[0])
            cross_entropy = cross_entropy.data.numpy()[0]

        return result, cross_entropy, effective_len

    def sample2(self, batch_loader, seq_len, use_cuda, source_sentence):

        sample2_word_tensor = np.array(list(map(batch_loader.word_to_idx.get, source_sentence.split())))
        sample2_character_tensor = np.array(list(map(batch_loader.encode_characters, source_sentence.split())))

        sample2_word_tensor = np.flip(sample2_word_tensor, 0)
        sample2_character_tensor = np.flip(sample2_character_tensor, 1)

        to_add = seq_len - len(sample2_word_tensor)
        to_add_words = np.array([batch_loader.word_to_idx[batch_loader.pad_token]] * to_add)
        to_add_chars = np.array([batch_loader.encode_characters(batch_loader.pad_token)] * to_add)

        sample2_word_tensor = [np.concatenate((to_add_words, sample2_word_tensor))]
        sample2_character_tensor = [np.concatenate((to_add_chars, sample2_character_tensor), 0)]

        # first sentence
        input = [np.array(sample2_word_tensor), np.array(sample2_character_tensor)]
        input = [Variable(t.from_numpy(var)) for var in input]
        input = [var.long() for var in input]
        input = [var.cuda() if use_cuda else var for var in input]

        sample2_word_tensor, sample2_character_tensor = input
        encoder_input = self.embedding(sample2_word_tensor, sample2_character_tensor)

        context = self.encoder(encoder_input)

        mu = self.context_to_mu(context)
        logvar = self.context_to_logvar(context)
        std = t.exp(0.5 * logvar)

        z = Variable(t.randn([1, self.params.latent_variable_size]))

        if use_cuda:
            z = z.cuda()

        seed = z * std + mu

        if use_cuda:
            seed = seed.cuda()

        return self.sample(batch_loader, seq_len, seed, use_cuda, sample2_word_tensor)

    def sample3(self, batch_loader, seq_len, use_cuda, train_word_sample, train_chars_sample, train_target):

        sample2_word_tensor = train_word_sample
        sample2_character_tensor = train_chars_sample
        encoder_input = self.embedding(sample2_word_tensor, sample2_character_tensor)

        context = self.encoder(encoder_input)

        mu = self.context_to_mu(context)
        logvar = self.context_to_logvar(context)
        std = t.exp(0.5 * logvar)

        z = Variable(t.randn([1, self.params.latent_variable_size]))

        if use_cuda:
            z = z.cuda()

        seed = z * std + mu

        if use_cuda:
            seed = seed.cuda()

        return self.sample(batch_loader, seq_len, seed, use_cuda, sample2_word_tensor, False, train_target)
