
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader

from tqdm import tqdm
import random
import logging
import os
logger = logging.getLogger()

# from transformers import BertTokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
from src.config import get_params
params = get_params()
from transformers import AutoTokenizer
auto_tokenizer = AutoTokenizer.from_pretrained(params.model_name)
pad_token_label_id = nn.CrossEntropyLoss().ignore_index

# politics_labels = ['O', 'B-country', 'B-politician', 'I-politician', 'B-election', 'I-election', 'B-person', 'I-person', 'B-organisation', 'I-organisation', 'B-location', 'B-misc', 'I-location', 'I-country', 'I-misc', 'B-politicalparty', 'I-politicalparty', 'B-event', 'I-event']
# science_labels = ['O', 'B-scientist', 'I-scientist', 'B-person', 'I-person', 'B-university', 'I-university', 'B-organisation', 'I-organisation', 'B-country', 'I-country', 'B-location', 'I-location', 'B-discipline', 'I-discipline', 'B-enzyme', 'I-enzyme', 'B-protein', 'I-protein', 'B-chemicalelement', 'I-chemicalelement', 'B-chemicalcompound', 'I-chemicalcompound', 'B-astronomicalobject', 'I-astronomicalobject', 'B-academicjournal', 'I-academicjournal', 'B-event', 'I-event', 'B-theory', 'I-theory', 'B-award', 'I-award', 'B-misc', 'I-misc']
# music_labels = ['O', 'B-musicgenre', 'I-musicgenre', 'B-song', 'I-song', 'B-band', 'I-band', 'B-album', 'I-album', 'B-musicalartist', 'I-musicalartist', 'B-musicalinstrument', 'I-musicalinstrument', 'B-award', 'I-award', 'B-event', 'I-event', 'B-country', 'I-country', 'B-location', 'I-location', 'B-organisation', 'I-organisation', 'B-person', 'I-person', 'B-misc', 'I-misc']
# literature_labels = ["O", "B-book", "I-book", "B-writer", "I-writer", "B-award", "I-award", "B-poem", "I-poem", "B-event", "I-event", "B-magazine", "I-magazine", "B-literarygenre", "I-literarygenre", 'B-country', 'I-country', "B-person", "I-person", "B-location", "I-location", 'B-organisation', 'I-organisation', 'B-misc', 'I-misc']
# ai_labels = ["O", "B-field", "I-field", "B-task", "I-task", "B-product", "I-product", "B-algorithm", "I-algorithm", "B-researcher", "I-researcher", "B-metrics", "I-metrics", "B-programlang", "I-programlang", "B-conference", "I-conference", "B-university", "I-university", "B-country", "I-country", "B-person", "I-person", "B-organisation", "I-organisation", "B-location", "I-location", "B-misc", "I-misc"]
only_index_labels = ["B","I","O"]
source_labels = ["教育","地点","软件","事件","文化","法律法规","时间与日历","品牌","人物","自然地理","工作","网站","组织","车辆","诊断与治疗","疾病和症状","奖项","生物","食物","游戏","星座","虚拟事物","药物"]
target_labels = ["地市","方位","行政村或社区","开发区","乡镇街道","子兴趣点","房屋编号","省份","路名","路口","路号","兴趣点","单元号","村子组别","区县","楼层号","距离","others"]
domain2labels = {"source":source_labels,"target":target_labels}

def set_type_token_embedding(model,tgt_dm):
    model.bert.resize_token_embeddings(len(auto_tokenizer))
    token_embeddings = model.bert.get_input_embeddings()
    source_type_tokens = ["[source_type_%d]" % (i) for i in range(len(source_labels))]
    target_type_tokens = ["[target_type_%d]" % (i) for i in range(len(domain2labels[tgt_dm]))]
    domain_type_words = domain2labels[tgt_dm]
    source_type_token_ids = []
    source_special_token_ids = []
    target_type_token_ids = []
    target_special_token_ids = []
    
    for i in range(len(source_type_tokens)):
        source_special_token_ids.append(auto_tokenizer.convert_tokens_to_ids(source_type_tokens[i]))
        source_type_token_ids.append(auto_tokenizer.convert_tokens_to_ids(auto_tokenizer.tokenize(source_labels[i])))
    for i in range(len(target_type_tokens)):
        target_special_token_ids.append(auto_tokenizer.convert_tokens_to_ids(target_type_tokens[i]))
        target_type_token_ids.append(auto_tokenizer.convert_tokens_to_ids(auto_tokenizer.tokenize(domain_type_words[i])))
    with torch.no_grad():
        for i in range(len(source_type_tokens)):
            token_embeddings.weight[source_special_token_ids[i]] = token_embeddings(torch.LongTensor(source_type_token_ids[i])).mean(dim=0)
        for i in range(len(target_type_tokens)):
            token_embeddings.weight[target_special_token_ids[i]] = token_embeddings(torch.LongTensor(target_type_token_ids[i])).mean(dim=0)
    model.bert.set_input_embeddings(token_embeddings)
    
def get_type_inputs(params):
    tgt_dm = params.tgt_dm
    source_type_tokens = ["[source_type_%d]" % (i) for i in range(len(source_labels))]
    target_type_tokens = ["[target_type_%d]" % (i) for i in range(len(domain2labels[tgt_dm]))]
    type_tokens = source_type_tokens + target_type_tokens
    auto_tokenizer.add_special_tokens(
        {
            "additional_special_tokens":type_tokens
        }
    )
    source_type_inputs = ["entity type "+token for token in source_type_tokens]
    target_type_inputs = ["entity type "+token for token in target_type_tokens]
    source_type_inputs = auto_tokenizer(
        source_type_inputs,
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt"
    )["input_ids"]
    target_type_inputs = auto_tokenizer(
        target_type_inputs,
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt"
    )["input_ids"]
    return source_type_inputs,target_type_inputs

def read_ner(datapath, tgt_dm,only_index = False):
    inputs, index_labels, type_labels = [], [],[]
    with open(datapath, "r") as fr:
        token_list, index_label_list,type_label_list = [], [], []
        for i, line in enumerate(fr):
            line = line.strip()
            if line == "":
                if len(token_list) > 0:
                    assert len(token_list) == len(index_label_list)
                    assert len(token_list) == len(type_label_list)
                    inputs.append([auto_tokenizer.cls_token_id] + token_list + [auto_tokenizer.sep_token_id])
                    index_labels.append([pad_token_label_id] + index_label_list + [pad_token_label_id])
                    type_labels.append([pad_token_label_id] + type_label_list + [pad_token_label_id])
                token_list, index_label_list,type_label_list = [], [], []
                continue
            
            splits = line.split(" ")
            token = splits[0]
            label = splits[1]
            if only_index:
                label = label.split("-")[0]
            else:
                index_label = label.split("-")[0]
                if index_label == "O":
                    type_label = ""
                else:
                    type_label = label.split("-")[1]
            subs_ = auto_tokenizer.tokenize(token)
            if len(subs_) > 0:
                index_label_list.extend([only_index_labels.index(index_label)] + [pad_token_label_id] * (len(subs_) - 1))
                if type_label == "":
                    type_label_list.extend([pad_token_label_id]* len(subs_))
                else:
                    type_label_list.extend([domain2labels[tgt_dm if not only_index else "only_index"].index(type_label)] + [pad_token_label_id] * (len(subs_) - 1))
                token_list.extend(auto_tokenizer.convert_tokens_to_ids(subs_))
            else:
                print("length of subwords for %s is zero; its label is %s" % (token, label))

    return inputs, index_labels,type_labels


def read_ner_for_bilstm(datapath, tgt_dm, vocab):
    inputs, labels = [], []
    with open(datapath, "r") as fr:
        token_list, label_list = [], []
        for i, line in enumerate(fr):
            line = line.strip()
            if line == "":
                if len(token_list) > 0:
                    assert len(token_list) == len(label_list)
                    inputs.append(token_list)
                    labels.append(label_list)
                
                token_list, label_list = [], []
                continue
            
            splits = line.split("\t")
            token = splits[0]
            label = splits[1]
            
            token_list.append(vocab.word2index[token])
            label_list.append(domain2labels[tgt_dm].index(label))

    return inputs, labels



class Dataset(data.Dataset):
    def __init__(self, inputs, index_labels,type_labels):
        self.X = inputs
        self.index = index_labels
        self.type = type_labels  
    def __getitem__(self, index):
        return self.X[index], self.index[index],self.type[index]

    def __len__(self):
        return len(self.X)


PAD_INDEX = 0
class Vocab():
    def __init__(self):
        self.word2index = {"PAD": PAD_INDEX}
        self.index2word = {PAD_INDEX: "PAD"}
        self.n_words = 1

    def index_words(self, word_list):
        for word in word_list:
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1

def get_vocab(path):
    vocabulary = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            vocabulary.append(line)
    return vocabulary


def collate_fn(data):
    X, index_labels,type_labels = zip(*data)
    lengths = [len(bs_x) for bs_x in X]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(auto_tokenizer.pad_token_id)
    padded_indexs = torch.LongTensor(len(X), max_lengths).fill_(pad_token_label_id)
    padded_types = torch.LongTensor(len(X), max_lengths).fill_(pad_token_label_id)
    for i, (seq, indexs,types) in enumerate(zip(X, index_labels,type_labels)):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)
        padded_indexs[i, :length] = torch.LongTensor(indexs)
        padded_types[i,:length] = torch.LongTensor(types)

    return padded_seqs,lengths, padded_indexs,padded_types


def collate_fn_for_bilstm(data):
    X, y = zip(*data)
    lengths = [len(bs_x) for bs_x in X]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(PAD_INDEX)
    for i, seq in enumerate(X):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)

    lengths = torch.LongTensor(lengths)
    return padded_seqs, lengths, y


def get_dataloader_for_bilstmtagger(params):
    vocab_src = get_vocab("ner_data/conll2003/vocab.txt")
    vocab_tgt = get_vocab("ner_data/%s/vocab.txt" % params.tgt_dm)
    vocab = Vocab()
    vocab.index_words(vocab_src)
    vocab.index_words(vocab_tgt)

    logger.info("Load training set data ...")
    conll_inputs_train, conll_labels_train = read_ner_for_bilstm("ner_data/conll2003/train", params.tgt_dm, vocab)
    inputs_train, labels_train = read_ner_for_bilstm("ner_data/%s/train" % params.tgt_dm, params.tgt_dm, vocab)
    inputs_train = inputs_train * 10 + conll_inputs_train
    labels_train = labels_train * 10 + conll_labels_train

    logger.info("Load dev set data ...")
    inputs_dev, labels_dev = read_ner_for_bilstm("ner_data/%s/dev" % params.tgt_dm, params.tgt_dm, vocab)

    logger.info("Load test set data ...")
    inputs_test, labels_test = read_ner_for_bilstm("ner_data/%s/test" % params.tgt_dm, params.tgt_dm, vocab)

    logger.info("train size: %d; dev size %d; test size: %d;" % (len(inputs_train), len(inputs_dev), len(inputs_test)))

    dataset_train = Dataset(inputs_train, labels_train)
    dataset_dev = Dataset(inputs_dev, labels_dev)
    dataset_test = Dataset(inputs_test, labels_test)
    
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn_for_bilstm)
    dataloader_dev = DataLoader(dataset=dataset_dev, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn_for_bilstm)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn_for_bilstm)

    return dataloader_train, dataloader_dev, dataloader_test, vocab


def load_corpus(tgt_dm):
    print("Loading corpus ...")
    data_path = "enwiki_corpus/%s_removebracket.tok" % tgt_dm
    sent_list = []
    with open(data_path, "r") as fr:
        for i, line in tqdm(enumerate(fr)):
            line = line.strip()
            sent_list.append(line)
    return sent_list


def get_dataloader(params):
    logger.info("Load training set data")
    inputs_train, index_labels_train,type_labels_train = read_ner("ner_data/%s/train" % params.tgt_dm, params.tgt_dm,params.only_index)
    dataset_train = Dataset(inputs_train, index_labels_train,type_labels_train)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn)

    logger.info("Load development set data")
    if os.path.exists("ner_data/%s/dev" % params.tgt_dm):
        inputs_dev, index_labels_dev,type_labels_dev = read_ner("ner_data/%s/dev" % params.tgt_dm, params.tgt_dm,params.only_index)
        dataset_dev = Dataset(inputs_dev, index_labels_dev,type_labels_dev)
        dataloader_dev = DataLoader(dataset=dataset_dev, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        dataloader_dev = None
    
    logger.info("Load test set data")
    if os.path.exists("ner_data/%s/test" % params.tgt_dm):
        inputs_test, index_labels_test,type_labels_test = read_ner("ner_data/%s/test" % params.tgt_dm, params.tgt_dm,params.only_index)
        dataset_test = Dataset(inputs_test, index_labels_test,type_labels_test)
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        dataloader_test = None
    logger.info("train size: %d; dev size %d; test size: %d;" % (len(inputs_train), 0 if dataloader_dev is None else len(inputs_dev), 0 if dataloader_test is None else len(inputs_test)))
    
    return dataloader_train, dataloader_dev, dataloader_test


def get_source_dataloader(batch_size,domain_name):

    inputs_train, index_labels_train,type_labels_train = read_ner("ner_data/{}/train".format(domain_name),domain_name)
    dataset_train = Dataset(inputs_train, index_labels_train,type_labels_train)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    if os.path.exists("ner_data/%s/dev" % domain_name):
        inputs_dev, index_labels_dev,type_labels_dev = read_ner("ner_data/{}/dev".format(domain_name), domain_name)
        dataset_dev = Dataset(inputs_dev, index_labels_dev,type_labels_dev)
        dataloader_dev = DataLoader(dataset=dataset_dev, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        dataloader_dev = None

    if os.path.exists("ner_data/%s/test" % domain_name):
        inputs_test, index_labels_test,type_labels_test = read_ner("ner_data/{}/test".format(domain_name), domain_name)
        dataset_test = Dataset(inputs_test, index_labels_test,type_labels_test)
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        dataloader_test = None
    
    logger.info("train size: %d; dev size %d; test size: %d;" % (len(inputs_train), 0 if dataloader_dev is None else len(inputs_dev), 0 if dataloader_test is None else len(inputs_test)))

    return dataloader_train, dataloader_dev, dataloader_test


if __name__ == "__main__":
    read_ner("../ner_data/final_politics/politics.txt", "politics")
