import io
import sys
import re
import numpy as np
import random
import tensorflow as tf
import os
import nltk
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

skip_step=100
THRESHOLD = 1
ENC_VOCAB = 41303
DEC_VOCAB = 18778
NUM_SAMPLES = 18777
NUM_LAYERS = 3
HIDDEN_SIZE = 256
BATCH_SIZE = 256
PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3
LR = 0.5
MAX_GRAD_NORM = 5.0
BUCKETS = [(19, 19), (28, 28), (33, 33), (40, 43), (50, 53), (60, 63)]
MAX_ITERATION = 20001
PROCESSED_PATH = 'processed'



def loadWords():
    encode_file = open(os.path.join(PROCESSED_PATH, 'tst2012.en.txt'), 'r', encoding='utf-8')
    decode_file = open(os.path.join(PROCESSED_PATH, 'tst2012.vi.txt'), 'r', encoding='utf-8')
    return encode_file,decode_file;


#--------------------Function to tokenize the given sentence into tokens---------------------------
def basic_tokenizer(line, normalize_digits=True):
    """ A basic tokenizer to tokenize text into tokens.
    Feel free to change this to suit your need. """
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    words = []
    _WORD_SPLIT = re.compile("([.,!?\"'-<>:;)(])")
    _DIGIT_RE = re.compile(r"\d")
    """ strip is to remove the strip off the extra whitespaces at the end of the
    line and split is to split the  given sentence into tokens and stored in list """
    for fragment in line.strip().lower().split():
        for token in re.split(_WORD_SPLIT, fragment):
            if not token:
                continue
            if normalize_digits:
                token = re.sub(_DIGIT_RE, '#', token)
            words.append(token)
    return words

#-------------------Function to build the vocabulary-------------------------------------------
def build_vocab(filename, normalize_digits=True):
    """Function to build the vocabulary """
    in_path = os.path.join(PROCESSED_PATH, filename)
    out_path = os.path.join(PROCESSED_PATH, 'vocab.{}'.format(filename[-2:]))

    vocab = {}
    with open(in_path, 'r',encoding='utf-8') as f:
        for line in f.readlines():
            for token in basic_tokenizer(line):
                if not token in vocab:
                    vocab[token] = 0
                vocab[token] += 1
    #f.write will write the strings to an output file
    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    with open(out_path, 'w',encoding='utf-8') as f:
        f.write('<pad>' + '\n')
        f.write('<unk>' + '\n')
        f.write('<s>' + '\n')
        f.write('<\s>' + '\n')
        index = 4
        for word in sorted_vocab:
            if vocab[word] < THRESHOLD:
                break
            f.write(word + '\n',)
            index += 1

#------------------function to load the vocabulary---------------------
def load_vocab(vocab_path):
    """ Function to load the path of the vocabulary"""
    with open(vocab_path, 'r',encoding='utf-8') as f:
        words = f.read().splitlines()
    return words, {words[i]: i for i in range(len(words))}


#-----------------Function to assign a sentence to a particular id--------------
def sentence2id(vocab, line):
    """ Function to convert the sentence to id"""
    return [vocab.get(token, vocab['<unk>']) for token in basic_tokenizer(line)]

#----------------Function to assign a token to a id-----------------------------
def token2id(data, mode):
    """ Convert all the tokens in the data into their corresponding
    unique-id in the vocabulary. """
    vocab_path = 'vocab.' + mode
    in_path = data + '.' + mode
    out_path = data + '_ids.' + mode

    _, vocab = load_vocab(os.path.join(PROCESSED_PATH, vocab_path))
    in_file = open(os.path.join(PROCESSED_PATH, in_path), 'r', encoding='utf-8')
    out_file = open(os.path.join(PROCESSED_PATH, out_path), 'w', encoding='utf-8')

    lines = in_file.read().splitlines()
    for line in lines:
        if mode == 'vi':  # we only care about '<s>' and </s> in encoder
            ids = [vocab['<s>']]
        else:
            ids = []
        ids.extend(sentence2id(vocab, line))
        # ids.extend([vocab.get(token, vocab['<unk>']) for token in basic_tokenizer(line)])
        if mode == 'vi':
            ids.append(vocab['<\s>'])
        out_file.write(' '.join(str(id_) for id_ in ids) + '\n')

#--------------------Function to load the data----------------------------------------
def load_data(enc_filename, dec_filename, max_training_size=None):
    encode_file = open(os.path.join(PROCESSED_PATH, enc_filename), 'r',encoding='utf-8')
    decode_file = open(os.path.join(PROCESSED_PATH, dec_filename), 'r',encoding='utf-8')
    encode, decode = encode_file.readline(), decode_file.readline()
    data_buckets = [[] for _ in BUCKETS]
    i = 0
    while encode and decode:
        if (i + 1) % 20000 == 0:
            print("Bucketing conversation number", i)
        encode_ids = [(id_) for id_ in encode.split()]
        decode_ids = [(id_) for id_ in decode.split()]
        for bucket_id, (encode_max_size, decode_max_size) in enumerate(BUCKETS):
            if len(encode_ids) <= encode_max_size and len(decode_ids) <= decode_max_size:
                data_buckets[bucket_id].append([encode_ids, decode_ids])
                break;
        encode, decode = encode_file.readline(), decode_file.readline()
        i += 1
    return data_buckets


#---------------------Function to get the appropriate buckets--------------------
def _get_buckets():
    """ Load the dataset into buckets based on their lengths.
    train_buckets_scale is the inverval that'll help us
    choose a random bucket later on.
    """
    # test_buckets = load_data('test_ids.enc.txt', 'test_ids.dec.txt')
    data_buckets = load_data('train_ids.en', 'train_ids.vi')
    train_bucket_sizes = [len(data_buckets[b]) for b in range(len(BUCKETS))]
    print("Number of samples in each bucket:\n", train_bucket_sizes)
    train_total_size = sum(train_bucket_sizes)
    # list of increasing numbers from 0 to 1 that we'll use to select a bucket.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]
    print("Bucket scale:\n", train_buckets_scale)
    return data_buckets, train_buckets_scale

#-------------------Function to assign a random bucket initially-------------------
def _get_random_bucket(train_buckets_scale):
    """ Get a random bucket from which to choose a training sample """
    rand = random.random()
    return min([i for i in range(len(train_buckets_scale))
                if train_buckets_scale[i] > rand])

#----------------Function to pad the input if the size of the input is small-------------
def _pad_input(input_, size):
    """ Function to provide the input with a padding"""
    return input_ + [PAD_ID] * (size - len(input_))

#----------------Function to reshape the given batch----------------------------------
def _reshape_batch(inputs, size, batch_size):
    """ Create batch-major inputs. Batch inputs are just re-indexed inputs
    """
    batch_inputs = []
    for length_id in range(size):
        batch_inputs.append(np.array([inputs[batch_id][length_id]
                                      for batch_id in range(batch_size)], dtype=np.int32))
    return batch_inputs

#------------------Function to get a batch--------------------------------
def get_batch(data_bucket, bucket_id, batch_size=1):
    """ Return one batch to feed into the model """
    # only pad to the max length of the bucket
    encoder_size, decoder_size = BUCKETS[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    for _ in range(batch_size):
        encoder_input, decoder_input = random.choice(data_bucket)
        # pad both encoder and decoder, reverse the encoder
        encoder_inputs.append(list(reversed(_pad_input(encoder_input, encoder_size))))
        decoder_inputs.append(_pad_input(decoder_input, decoder_size))

    # now we create batch-major vectors from the data selected above.
    batch_encoder_inputs = _reshape_batch(encoder_inputs, encoder_size, batch_size)
    batch_decoder_inputs = _reshape_batch(decoder_inputs, decoder_size, batch_size)

    # create decoder_masks to be 0 for decoders that are padding.
    batch_masks = []
    for length_id in range(decoder_size):
        batch_mask = np.ones(batch_size, dtype=np.float32)
        for batch_id in range(batch_size):
            # we set mask to 0 if the corresponding target is a PAD symbol.
            # the corresponding decoder is decoder_input shifted by 1 forward.
            if length_id < decoder_size - 1:
                target = decoder_inputs[batch_id][length_id + 1]
            if length_id == decoder_size - 1 or target == PAD_ID:
                batch_mask[batch_id] = 0.0
        batch_masks.append(batch_mask)
    return batch_encoder_inputs, batch_decoder_inputs, batch_masks


#------------------------Function to check the length of the encoder and the decoder sequence-----------
def _assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks):
    """ Assert that the encoder inputs, decoder inputs, and decoder masks are
    of the expected lengths """
    if len(encoder_inputs) != encoder_size:
        raise ValueError("Encoder length must be equal to the one in bucket,"
                         " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
        raise ValueError("Decoder length must be equal to the one in bucket,"
                         " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(decoder_masks) != decoder_size:
        raise ValueError("Weights length must be equal to the one in bucket,"
                         " %d != %d." % (len(decoder_masks), decoder_size))

#---------------------------Function to find the right bucket-----------------------
def _find_right_bucket(length):
    """ Find the proper bucket for an encoder input based on its length """
    return min([b for b in range(len(BUCKETS))
                if BUCKETS[b][0] >= length])

#-----------------------Function to construct the response correctly------------------
def _construct_response(output_logits, inv_dec_vocab):
    """ Construct a response to the user's encoder input.
    @output_logits: the outputs from sequence to sequence wrapper.
    output_logits is decoder_size np array, each of dim 1 x DEC_VOCAB

    This is a greedy decoder - outputs are just argmaxes of output_logits.
    """

    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    # If there is an EOS symbol in outputs, cut them at that point.
    if EOS_ID in outputs:
        outputs = outputs[:outputs.index(EOS_ID)]
    # Print out sentence corresponding to outputs.
    return " ".join([tf.compat.as_str(inv_dec_vocab[output]) for output in outputs])


class ChatBotModel(object):
    def __init__(self, fw_only):

        self.fw_only = fw_only

        #print('Create placeholders')
        self.encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='encoder{}'.format(i))
                               for i in range(BUCKETS[-1][0])]
        self.decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='decoder{}'.format(i))
                               for i in range(BUCKETS[-1][1] + 1)]
        self.decoder_masks = [tf.placeholder(tf.float32, shape=[None], name='mask{}'.format(i))
                              for i in range(BUCKETS[-1][1] + 1)]
        self.targets = self.decoder_inputs[1:]

        if NUM_SAMPLES > 0 and NUM_SAMPLES < DEC_VOCAB:
            w = tf.get_variable('proj_w', [HIDDEN_SIZE,
                                           DEC_VOCAB])  # as the weights are shared between different cells it is used instead of tf.Variable
            b = tf.get_variable('proj_b', [DEC_VOCAB])
            self.output_projection = (w, b)

        def sampled_loss(logits, labels):
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(weights=tf.transpose(w),
                                              biases=b,
                                              inputs=logits,
                                              labels=labels,
                                              num_sampled=NUM_SAMPLES,
                                              num_classes=DEC_VOCAB)

        self.softmax_loss_function = sampled_loss

        single_cell = tf.contrib.rnn.GRUCell(HIDDEN_SIZE)
        self.cell = tf.contrib.rnn.MultiRNNCell([single_cell for _ in range(NUM_LAYERS)])



        def _seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
            setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs, decoder_inputs, self.cell,
                num_encoder_symbols=ENC_VOCAB,
                num_decoder_symbols=DEC_VOCAB,
                embedding_size=HIDDEN_SIZE,
                output_projection=self.output_projection,
                feed_previous=do_decode)

        if self.fw_only:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs,
                self.decoder_inputs,
                self.targets,
                self.decoder_masks,
                BUCKETS,
                lambda x, y: _seq2seq_f(x, y, True),
                softmax_loss_function=self.softmax_loss_function)
            # If we use output projection, we need to project outputs for decoding.
            if self.output_projection:
                for bucket in range(len(BUCKETS)):
                    self.outputs[bucket] = [tf.matmul(output,
                                                      self.output_projection[0]) + self.output_projection[1]
                                            for output in self.outputs[bucket]]
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs,
                self.decoder_inputs,
                self.targets,
                self.decoder_masks,
                BUCKETS,
                lambda x, y: _seq2seq_f(x, y, False),
                softmax_loss_function=self.softmax_loss_function)
        with tf.variable_scope('training') as scope:
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

            if not self.fw_only:
                self.optimizer = tf.train.GradientDescentOptimizer(LR)
                trainables = tf.trainable_variables()
                self.gradient_norms = []
                self.train_ops = []

                for bucket in range(len(BUCKETS)):
                    clipped_grads, norm = tf.clip_by_global_norm(tf.gradients(self.losses[bucket],
                                                                              trainables),
                                                                 MAX_GRAD_NORM)
                    self.gradient_norms.append(norm)
                    self.train_ops.append(self.optimizer.apply_gradients(zip(clipped_grads, trainables),
                                                                         global_step=self.global_step))

#---------------------Fucntion to check the parameters from the trained model----------------------
def _check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('model' + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading Model ... ")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Initializing fresh parameters of Translator")

#------------------Function to run the model----------------------------------
def run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, forward_only, batch_size):
    """ Run one step in training.
    @forward_only: boolean value to decide whether a backward path should be created
    forward_only is set to True when you just want to evaluate on the test set,
    or when you want to the bot to be in chat mode. """
    encoder_size, decoder_size = BUCKETS[bucket_id]
    _assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks)

    # input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for step in range(encoder_size):
        input_feed[model.encoder_inputs[step].name] = encoder_inputs[step]
    for step in range(decoder_size):
        input_feed[model.decoder_inputs[step].name] = decoder_inputs[step]
        input_feed[model.decoder_masks[step].name] = decoder_masks[step]

    last_target = model.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([batch_size], dtype=np.int32) # end of sentence indication

    # output feed: depends on whether we do a backward step or not.
    if not forward_only:
        output_feed = [model.train_ops[bucket_id],  # update op that does SGD.
                       model.gradient_norms[bucket_id],  # gradient norm.
                       model.losses[bucket_id]]  # loss for this batch.
    else:
        output_feed = [model.losses[bucket_id]]  # loss for this batch.
        for step in range(decoder_size):  # output logits.
            output_feed.append(model.outputs[bucket_id][step])

    outputs = sess.run(output_feed, input_feed)
    if not forward_only:
        return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
        return None, outputs[0], outputs[1:]

#-------------------------Function to load the data set------------------------
def loaddataset():
    build_vocab('train_ids.en')
    build_vocab('train_ids.vi')
    token2id('train_ids', 'vi')
    token2id('train_ids', 'en')



#---------Function to train the network-----------------------------
def train(data_buckets, train_buckets_scale):
    print('Started Training')
    model = ChatBotModel(False)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        print('Running session')
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)
        iteration = model.global_step.eval()
        total_loss = 0
        while True:
            bucket_id = _get_random_bucket(train_buckets_scale)
            encoder_inputs, decoder_inputs, decoder_masks = get_batch(data_buckets[bucket_id],
                                                                           bucket_id,
                                                                           batch_size=BATCH_SIZE)
            _, step_loss, _ = run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, False)
            total_loss += step_loss
            iteration += 1
            if iteration % 1000 == 0:
                print('Iter {}: loss {}'.format(iteration, total_loss/skip_step))
                total_loss = 0
                saver.save(sess, os.path.join('model', 'translate'), global_step=model.global_step)
        saver = tf.train.Saver()
        save_path = saver.save(sess, "model/model.ckpt")
        print('**************Model successfully saved****************')

#--------------------Function to test the model---------------------
def chat():
    model = ChatBotModel(True)
    encode_file,decode_file=loadWords();
    saver = tf.train.Saver()  # to restore parameters value
    with tf.Session() as sess:
        print('Loading data for testing the NMT model')
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)
        encode, decode = encode_file.readline(), decode_file.readline()
        i = 0
        bleu_score = []
        while encode and decode:
            output_split = basic_tokenizer(decode)
            response = translate_line(sess, model, encode)
            response_split = basic_tokenizer(response)
            i = i + 1
            if i < 20:
                print(response)
            if (response != ""):
                bleu_score.append(nltk.translate.bleu_score.sentence_bleu([output_split], response_split,
                                                                          smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1))
            encode, decode = encode_file.readline(), decode_file.readline()
        print("Average BLEU: {0}".format(np.mean(bleu_score)*100))

#------------------Function to translate each line-------------------
def translate_line(sess,model,line):
    _, en_vocab = load_vocab(os.path.join(PROCESSED_PATH, 'vocab.en'))
    inv_vi_vocab, _ = load_vocab(os.path.join(PROCESSED_PATH, 'vocab.vi'))
    max_length = BUCKETS[-1][0]
    token_ids = sentence2id(en_vocab, str(line))
    if (len(token_ids) > max_length):
        return ""
    bucket_id = _find_right_bucket(len(token_ids))
    encoder_inputs, decoder_inputs, decoder_masks = get_batch([(token_ids, [])],
                                                                    bucket_id,
                                                                    batch_size=1)

    _, _, output_logits = run_step(sess, model, encoder_inputs, decoder_inputs,
                                   decoder_masks, bucket_id, True,1)
    response = _construct_response(output_logits, inv_vi_vocab)
    return response

#----------------------Function to receive the user input----------------------
def _get_user_input():
    """ Get user's input, which will be transformed into encoder input later """
    print("> ", end="")
    sys.stdout.flush()
    return sys.stdin.readline()

#-----------------Function to translate the given sentence------------------------

def translate():
    model = ChatBotModel(True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)
        print('Sentence in English :');
        while (1):
            l = _get_user_input();
            if len(l) > 0 and l[-1] == '\n':
                l = l[:-1]
            if l == '':
                break
            translatedSentence = translate_line(sess, model, l)
            print('Sentence in Vietnameese \n')
            print(translatedSentence)
            print('----------------------------------------------');


if sys.argv[1] == "train":
    loaddataset();
    data_buckets, train_buckets_scale = _get_buckets()
    train(data_buckets, train_buckets_scale);
elif sys.argv[1] == "test":
    chat();
elif sys.argv[1] == "translate":
    translate();
