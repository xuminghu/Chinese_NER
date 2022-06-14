
from src.config import get_params
from src.utils import init_experiment
from src.dataloader import get_dataloader, get_source_dataloader, get_dataloader_for_bilstmtagger
from src.trainer import BaseTrainer
from src.model import BertTagger, BiLSTMTagger
from src.coach.dataloader import get_dataloader_for_coach
from src.coach.model import EntityPredictor
from src.coach.trainer import CoachTrainer

import torch
import numpy as np
from tqdm import tqdm
import random


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(params):
    # initialize experiment
    logger = init_experiment(params, logger_filename=params.logger_filename)
    if params.only_index:
        params.num_tags = 3
    if params.bilstm:
        # dataloader
        dataloader_train, dataloader_dev, dataloader_test, vocab = get_dataloader_for_bilstmtagger(params)
        # bilstm-crf model
        model = BiLSTMTagger(params, vocab)
        model.cuda()
        # trainer
        trainer = BaseTrainer(params, model)
    elif params.coach:
        # dataloader
        dataloader_train, dataloader_dev, dataloader_test, vocab = get_dataloader_for_coach(params)
        # coach model
        binary_tagger = BiLSTMTagger(params, vocab)
        entity_predictor = EntityPredictor(params)
        binary_tagger.cuda()
        entity_predictor.cuda()
        # trainer
        trainer = CoachTrainer(params, binary_tagger, entity_predictor)
    else:
        # dataloader
        dataloader_train, dataloader_dev, dataloader_test = get_dataloader(params)
        # BERT-based NER Tagger
        model = BertTagger(params)
        model.cuda()
        # trainer
        trainer = BaseTrainer(params, model)
    if params.source:
        source_trainloader, source_devloader, source_testloader = get_source_dataloader(params.batch_size, params.src_dm)
        trainer.train_source(source_trainloader, source_devloader, source_testloader, params.src_dm)
    
    trainer.update_classifier(params.tgt_dm)
    no_improvement_num = 0
    best_f1 = 0
    logger.info("Training on target domain ...")
    for e in range(params.epoch):
        logger.info("============== epoch %d ==============" % e)
        
        pbar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
        if params.bilstm:
            loss_list = []
            for i, (X, lengths, y) in pbar:
                X, lengths = X.cuda(), lengths.cuda()
                loss = trainer.train_step_for_bilstm(X, lengths, y)
                loss_list.append(loss)
                pbar.set_description("(Epoch {}) LOSS:{:.4f}".format(e, np.mean(loss_list)))

            logger.info("Finish training epoch %d. loss: %.4f" % (e, np.mean(loss_list)))

        elif params.coach:
            loss_bin_list, loss_entity_list = [], []
            for i, (X, lengths, y_bin, y_final) in pbar:
                X, lengths = X.cuda(), lengths.cuda()
                loss_bin, loss_entityname = trainer.train_step(X, lengths, y_bin, y_final)
                loss_bin_list.append(loss_bin)
                loss_entity_list.append(loss_entityname)
                pbar.set_description("(Epoch {}) LOSS BIN:{:.4f}; LOSS ENTITY:{:.4f}".format(e, np.mean(loss_bin_list), np.mean(loss_entity_list)))
            
            logger.info("Finish training epoch %d. loss_bin: %.4f. loss_entity: %.4f" % (e, np.mean(loss_bin_list), np.mean(loss_entity_list)))

        else:
            loss_list = []
            for i, (X,lengths,indexs,types) in pbar:
                X, indexs,types = X.cuda(), indexs.cuda(),types.cuda()
                index_loss,type_loss,loss = trainer.train_step(X, indexs,types)
                loss_list.append(loss)
                pbar.set_description("(Epoch {}) LOSS:{:.4f}".format(e, np.mean(loss_list)))

            logger.info("Finish training epoch %d. loss: %.4f" % (e, np.mean(loss_list)))

        logger.info("============== Evaluate epoch %d on Train Set ==============" % e)       
        
        f1_train,prec_train,recall_train,f1_train_index,train_by_type,train_by_index = trainer.evaluate(dataloader_train, params.tgt_dm, use_bilstm=params.bilstm)
        logger.info("Evaluate on Train Set. Prec: %.4f Recall: %.4f F1: %.4f F1_index: %.4f . " % (prec_train,recall_train,f1_train,f1_train_index))
        for t in train_by_type.keys():
            logger.info("Entity Type %s. Prec: %.4f Recall: %.4f F1: %.4f" %(t,train_by_type[t]["precision"],train_by_type[t]["recall"],train_by_type[t]["fb1"]))
        for index in train_by_index.keys():
            logger.info("Entity Index %s. Prec: %.4f Recall: %.4f F1: %.4f" %(index,train_by_index[index]["precision"],train_by_index[index]["recall"],train_by_index[index]["fb1"]))

        logger.info("============== Evaluate epoch %d on Dev Set ==============" % e)
        if dataloader_dev is not None:       
            f1_dev,prec_dev,recall_dev,f1_dev_index,dev_by_type,dev_by_index= trainer.evaluate(dataloader_dev, params.tgt_dm, use_bilstm=params.bilstm)
            logger.info("Evaluate on Dev Set. Prec: %.4f Recall: %.4f F1: %.4f F1_index: %.4f . " % (prec_dev,recall_dev,f1_dev,f1_dev_index))
            for t in dev_by_type.keys():
                logger.info("Entity Type %s. Prec: %.4f Recall: %.4f F1: %.4f" %(t,dev_by_type[t]["precision"],dev_by_type[t]["recall"],dev_by_type[t]["fb1"]))
            for index in dev_by_index.keys():
                logger.info("Entity Index %s. Prec: %.4f Recall: %.4f F1: %.4f" %(index,dev_by_index[index]["precision"],dev_by_index[index]["recall"],dev_by_index[index]["fb1"]))

        logger.info("============== Evaluate epoch %d on Test Set ==============" % e)
        if params.only_index:
            f1_test = trainer.evaluate(dataloader_test, params.tgt_dm, use_bilstm=params.bilstm)
            logger.info("Evaluate on Test Set. F1: %.4f." % (f1_test))
        if dataloader_test is not None:
            f1_test,prec_test,recall_test,f1_test_index,test_by_type,test_by_index = trainer.evaluate(dataloader_test, params.tgt_dm, use_bilstm=params.bilstm)
            logger.info("Evaluate on Test Set. Prec: %.4f Recall: %.4f F1: %.4f F1_index: %.4f . " % (prec_test,recall_test,f1_test,f1_test_index))
            for t in test_by_type.keys():
                logger.info("Entity Type %s. Prec: %.4f Recall: %.4f F1: %.4f" %(t,test_by_type[t]["precision"],test_by_type[t]["recall"],test_by_type[t]["fb1"]))
            for index in test_by_index.keys():
                logger.info("Entity Index %s. Prec: %.4f Recall: %.4f F1: %.4f" %(index,test_by_index[index]["precision"],test_by_index[index]["recall"],test_by_index[index]["fb1"]))

            if f1_test > best_f1:
                logger.info("Found better model!!")
                best_f1 = f1_test
                no_improvement_num = 0
                # trainer.save_model()
            else:
                no_improvement_num += 1
                logger.info("No better model found (%d/%d)" % (no_improvement_num, params.early_stop))

            if no_improvement_num >= params.early_stop:
                if params.case_study:
                    f1_test,f1_test_index,cases = trainer.evaluate(dataloader_test, params.tgt_dm, use_bilstm=params.bilstm,case_study=True)
                    with open(params.tgt_dm+"_casestudy.log",'w') as f:
                        for (raw,pred,gold) in cases:
                            f.write("raw:{}\npred:{}\ngold:{}\n\n".format(raw,pred,gold))
                break


if __name__ == "__main__":
    params = get_params()

    random_seed(params.seed)
    train(params)
