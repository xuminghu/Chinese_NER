
import torch
import torch.nn as nn

from src.conll2002_metrics import *
from src.dataloader import domain2labels, pad_token_label_id,only_index_labels
from transformers import AutoTokenizer
import os
import numpy as np
from tqdm import tqdm
import logging
import torch.nn.functional as F
from src.dataloader import pad_token_label_id

logger = logging.getLogger()

class BaseTrainer(object):
    def __init__(self, params, model):
        self.params = params
        self.model = model
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr)
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = nn.BCELoss(reduction="none")
        self.early_stop = params.early_stop
        self.no_improvement_num = 0
        self.best_acc = 0

    def train_step(self, X, indexs,types):
        self.model.train()

        index_preds,type_preds = self.model(X)
        indexs = indexs.view(indexs.size(0)*indexs.size(1))
        types = types.view(types.size(0)*types.size(1))
        index_preds = index_preds.view(index_preds.size(0)*index_preds.size(1), index_preds.size(2))
        type_preds = type_preds.view(type_preds.size(0)*type_preds.size(1),type_preds.size(2))
        self.optimizer.zero_grad()
        # index_mask = ~(indexs==pad_token_label_id).view(-1)
        # type_mask = ~(types==pad_token_label_id).view(-1)
        # indexs[indexs==pad_token_label_id] = 0
        # types[types==pad_token_label_id] = 0
        # index_one_hot = F.one_hot(indexs.long(),num_classes = 3).float()
        # type_one_hot = F.one_hot(types.long(),num_classes = self.params.num_tag).float()
        # index_loss = self.loss_fn(index_preds, index_one_hot).sum(dim=-1)
        # type_loss = self.loss_fn(type_preds,type_one_hot).sum(dim=-1)
        # index_loss = (index_loss.view(-1) * index_mask).sum() /torch.sum(index_mask)
        # type_loss = (type_loss.view(-1) * type_mask).sum() /torch.sum(type_mask)
        index_loss = self.loss_fn(index_preds,indexs)
        type_loss = self.loss_fn(type_preds,types)
        loss = self.params.alpha * index_loss + (1-self.params.alpha) * type_loss
        loss.backward()
        self.optimizer.step()
        
        return index_loss.item(),type_loss.item(),loss.item()
    
    def train_step_for_bilstm(self, X, lengths, y):
        self.model.train()
        preds = self.model(X)
        loss = self.model.crf_loss(preds, lengths, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, dataloader, tgt_dm, use_bilstm=False,case_study=False,type_study=None):
        self.model.eval()

        index_pred_list = []
        type_pred_list = []
        index_list = []
        type_list = []
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        sentences = []
        if use_bilstm:
            for i, (X, lengths, y) in pbar:
                y_list.extend(y)
                X, lengths = X.cuda(), lengths.cuda()
                preds = self.model(X)
                preds = self.model.crf_decode(preds, lengths)
                pred_list.extend(preds)
        else:
            auto_tokenizer = AutoTokenizer.from_pretrained(self.params.model_name)
            for i, (X,lengths,indexs,types) in pbar:
                index_list.extend(indexs.data.numpy()) # y is a list
                type_list.extend(types.data.numpy())
                X = X.cuda()
                index_preds,type_preds = self.model(X)
                index_pred_list.extend(index_preds.data.cpu().numpy())
                type_pred_list.extend(type_preds.data.cpu().numpy())
                if case_study:
                    X = X.data.cpu()
                    for i in range(X.shape[0]):
                        sentences.append(X[i].numpy().tolist())
        # concatenation
        index_pred_list = np.concatenate(index_pred_list, axis=0)   # (length, num_tag)
        type_pred_list = np.concatenate(type_pred_list, axis=0)   # (length, num_tag)

        if not use_bilstm:
            index_pred_list = np.argmax(index_pred_list, axis=1)
            type_pred_list = np.argmax(type_pred_list,axis=1)
        index_list = np.concatenate(index_list, axis=0)
        type_list = np.concatenate(type_list, axis=0)
        # calcuate f1 score
        index_pred_list = list(index_pred_list)
        type_pred_list = list(type_pred_list)
        index_list = list(index_list)
        type_list = list(type_list)
        lines = []

        if case_study:
            num_sentence = 0
            num_tokens = 0
            flags = []
            flag = []
        for pred_index, pred_type,gold_index,gold_type in zip(index_pred_list, type_pred_list,index_list,type_list):
            gold_index = int(gold_index)
            if gold_index != pad_token_label_id:
                pred_index = only_index_labels[pred_index]
                if gold_type != pad_token_label_id:
                    pred_type = domain2labels[tgt_dm if not self.params.only_index else "only_index"][pred_type]
                    gold_type = domain2labels[tgt_dm if not self.params.only_index else "only_index"][gold_type]
                else:
                    pred_type = domain2labels[tgt_dm if not self.params.only_index else "only_index"][pred_type]
                    gold_type = ""
                pred_token = pred_index + "-" + pred_type
                gold_index = only_index_labels[gold_index]
                gold_token = gold_index + "-" + gold_type
                lines.append("w" + " " + pred_token + " " + gold_token)
                if case_study:
                    flag.append(True)
            else:
                if case_study:
                    flag.append(False)
            if case_study:
                num_tokens +=1
                if (num_tokens >=len(sentences[num_sentence])):
                    num_tokens = 0
                    num_sentence += 1
                    flags.append(flag)
                    flag = []
        if case_study:
            auto_tokenizer = AutoTokenizer.from_pretrained(self.params.model_name)
            results = conll2002_measure(lines,sentences = sentences,flags = flags,tokenizer = auto_tokenizer,type_study=type_study)
        else:
            results = conll2002_measure(lines,type_study=type_study)
        if self.params.only_index:
            return results["fb1"]
        else:
            f1 = results["fb1"]
            precision = results["precision"]
            recall = results["recall"]
            f1_index = results["fb1_index"]
            by_type = results["by_type"]
            by_index = results["by_index"]
            
            if case_study:
                cases = results["cases"]
                return f1,precision,recall,f1_index,by_type,by_index,cases
            elif type_study is not None:
                studied_type_preds = results["studied_type_preds"]
                return f1,precision,recall,f1_index,by_type,by_index,studied_type_preds
            else:
                return f1,precision,recall,f1_index,by_type,by_index    
    def train_source(self, dataloader_train, dataloader_dev, dataloader_test, domain_name):
        logger.info("Pretraining on Source NER dataset ...")
        no_improvement_num = 0
        best_f1 = 0
        for e in range(self.params.epoch):
            logger.info("============== epoch %d ==============" % e)
            loss_list = []
        
            pbar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
            for i, (X, lengths,indexs,types) in pbar:
                X,indexs,types = X.cuda(), indexs.cuda(),types.cuda()

                index_loss,type_loss,loss = self.train_step(X, indexs,types)
                loss_list.append(loss)
                pbar.set_description("(Epoch {}) LOSS:{:.4f}".format(e, np.mean(loss_list)))

            logger.info("Finish training epoch %d. loss: %.4f" % (e, np.mean(loss_list)))
            if dataloader_dev is not None:
                logger.info("============== Evaluate epoch %d on Dev Set ==============" % e)
                f1_dev,prec_dev,recall_dev,f1_dev_index,dev_by_type,dev_by_index = self.evaluate(dataloader_dev, domain_name)
                logger.info("Evaluate on Dev Set. Prec: %.4f Recall: %.4f F1: %.4f F1_index: %.4f . " % (prec_dev,recall_dev,f1_dev,f1_dev_index))
                for t in dev_by_type.keys():
                    logger.info("Entity Type %s. Prec: %.4f Recall: %.4f F1: %.4f" %(t,dev_by_type[t]["precision"],dev_by_type[t]["recall"],dev_by_type[t]["fb1"]))
                for index in dev_by_index.keys():
                    logger.info("Entity Index %s. Prec: %.4f Recall: %.4f F1: %.4f" %(index,dev_by_index[index]["precision"],dev_by_index[index]["recall"],dev_by_index[index]["fb1"]))

                if f1_dev > best_f1:
                    logger.info("Found better model!!")
                    best_f1 = f1_dev
                    no_improvement_num = 0
                else:
                    no_improvement_num += 1
                    logger.info("No better model found (%d/%d)" % (no_improvement_num, 1))

            # if no_improvement_num >= 1:
            #     break
            if e >= 1:
                break
        
        logger.info("============== Evaluate on Test Set ==============")
        if dataloader_test is not None:
            f1_test,prec_test,recall_test,f1_test_index,test_by_type,test_by_index = self.evaluate(dataloader_test, domain_name)
            logger.info("Evaluate on Test Set. Prec: %.4f Recall: %.4f F1: %.4f F1_index: %.4f . " % (prec_test,recall_test,f1_test,f1_test_index))
            for t in test_by_type.keys():
                logger.info("Entity Type %s. Prec: %.4f Recall: %.4f F1: %.4f" %(t,test_by_type[t]["precision"],test_by_type[t]["recall"],test_by_type[t]["fb1"]))
            for index in test_by_index.keys():
                logger.info("Entity Index %s. Prec: %.4f Recall: %.4f F1: %.4f" %(index,test_by_index[index]["precision"],test_by_index[index]["recall"],test_by_index[index]["fb1"]))
    
    def update_classifier(self,domain_name):
        with torch.no_grad():
            self.model.type_classifier = nn.Linear(self.params.hidden_dim,len(domain2labels[domain_name]))
            self.model = self.model.cuda()

    def save_model(self):
        """
        save the best model
        """
        saved_path = os.path.join(self.params.dump_path, "best_finetune_model.pth")
        torch.save({
            "model": self.model,
        }, saved_path)
        logger.info("Best model has been saved to %s" % saved_path)
    

