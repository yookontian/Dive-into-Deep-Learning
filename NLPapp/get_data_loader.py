import os
import torch
from d2l import torch as d2l


def read_imdb(data_dir, is_train):
    """Read the IMDb review dataset text sequences and labels."""
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test', label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels


# the original d2l.truncate_pad() will discard the content when len(line) > max_length, here we want an overlapping method to include the remain content. 
# this will increase num of training dataset

def all_data_features(train_tokens, vocab, num_steps, train_data):
    def truncating(line_o, max_length, pad_token, label):
        line = vocab[line_o]
        num_parts = (len(line) // max_length) + 1
        if num_parts == 1:
            return [line + [pad_token] * (max_length - len(line))], list([label])  # Pad
        else:
            new_lines = []
            for i in range(num_parts):
                if i * max_length + max_length > len(line):
                    new_line = line[i * max_length :]
                    new_line = new_line + [pad_token] * (max_length - len(new_line))
                else:
                    new_line = line[i * max_length : i * max_length + max_length]
                assert len(new_line) == max_length, (line[0:4], len(line), i * max_length + max_length - 1, len(new_line), line)
                new_lines.append(new_line)
            return new_lines, [label] * num_parts
    train_features = []
    train_labels = []
    for n, line in enumerate(train_tokens):
        new_lines, new_labels = truncating(line, num_steps, vocab['<pad>'], train_data[1][n])
        train_features.extend(new_lines)
        train_labels.extend(new_labels)
    
    return torch.tensor(train_features), train_labels


# all together
def load_data_imdb(batch_size, num_steps=500):
    """Return data iterators and the vocabulary of the IMDb review dataset."""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    # train_features = torch.tensor([d2l.truncate_pad(
        # vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    train_features, train_labels = all_data_features(train_tokens, vocab, num_steps, train_data)
    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    # train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                # batch_size)
    train_iter = d2l.load_array((train_features, torch.tensor(train_labels)),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab