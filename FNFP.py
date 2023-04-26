import sys
sys.path.append('..')
import util
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class JOProto(util.framework.FewShotNERModel):
    
    def __init__(self,word_encoder, dot=False, ignore_index=-1):
        util.framework.FewShotNERModel.__init__(self, word_encoder, ignore_index=ignore_index)
        self.drop = nn.Dropout()
        self.dot = dot

    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q, q_mask):
        # S [class, embed_dim], Q [num_of_sent, num_of_tokens, embed_dim]
        assert Q.size()[:2] == q_mask.size()
        Q = Q[q_mask==1].view(-1, Q.size(-1)) # [num_of_all_text_tokens, embed_dim]
        return self.__dist__(S.unsqueeze(0), Q.unsqueeze(1), 2)

    def __get_proto__(self, embedding, tag, mask):
        proto = []
        embedding = embedding[mask==1].view(-1, embedding.size(-1))
        tag = torch.cat(tag, 0)
        assert tag.size(0) == embedding.size(1)
        oproto = embedding[tag==0]
        for label in range(1,torch.max(tag)+1):
            proto.append(torch.mean(embedding[tag==label], 0))
        proto = torch.stack(proto)
        Loss = self.__get_jloss__(embedding, tag)
        return proto,oproto,Loss
    def __get_jloss__(self, embedding, tag):
        Oloss = 0
        sloss = 0
        N = torch.max(tag)
        for label in range(N + 1):
            old = embedding[tag == label]
            n = old.size(0)
            new = torch.mean(old, 0)
            if n!=1 :
                if label == 0:
                    loss = (torch.pow(new.unsqueeze(0) - old, 2)).sum(1)
                    loss = loss.mean() / embedding.size(-1)
                    Oloss = loss/n
                else:
                    tloss = 0
                    for i in range(n - 1):
                        a = old[i]
                        for j in range(i + 1, n):
                            b = old[j]
                            loss = torch.pow(a - b, 2).sum(0)
                            loss = loss.mean() / embedding.size(-1)
                            tloss = tloss + loss
                    tloss = (2*tloss / (n * n-n))
                    sloss = sloss + tloss
        sloss = sloss/ (N-1)
        Loss = sloss + Oloss
        return Loss
    def __get_jlo__(self, embedding, tag):
        Loss = 0
        for label in range(torch.max(tag)+1):
            old = embedding[tag == label]
            new = torch.mean(old, 0)
            # if label!=0 :
            #     loss = (torch.pow(new.unsqueeze(0)-old, 2)).sum(1)
            #     loss = loss.mean()/embedding.size(-1)
            #     loss = loss/tag.size(0)
            #     Loss = Loss + loss

            loss = (torch.pow(new.unsqueeze(0)-old, 2)).sum(1)
            loss = loss.mean()/embedding.size(-1)
            loss = loss/old.size(0)
            # print(loss)
            Loss = Loss + loss
        return Loss


    def forward(self, support, query):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        support: word:单词的位置     mask：01来判断是否存在单词        label：单词的类别     sentence_num：每个batch句子数量    text_mask：mask的基础上去除头和尾
        word和mask包含头101和102 label和text_mask不包含 text_mask头和尾用0表示
        '''
        support_emb = self.word_encoder(support['word'], support['mask']) # [num_sent, number_of_tokens, 768]
        query_emb = self.word_encoder(query['word'], query['mask']) # [num_sent, number_of_tokens, 768]
        support_emb = self.drop(support_emb)
        query_emb = self.drop(query_emb)

        logits = []
        ologits = torch.tensor([]).cuda()
        Jloss = 0

        current_support_num = 0
        current_query_num = 0
        assert support_emb.size()[:2] == support['mask'].size()
        assert query_emb.size()[:2] == query['mask'].size()

        for i, sent_support_num in enumerate(support['sentence_num']):
            sent_query_num = query['sentence_num'][i]
            # Calculate prototype for each class
            support_proto,oproto,sloss = self.__get_proto__(
                support_emb[current_support_num:current_support_num+sent_support_num], 
                support['label'][current_support_num:current_support_num+sent_support_num], 
                support['text_mask'][current_support_num: current_support_num+sent_support_num])

            ologit = self.__batch_dist__(
                oproto,
                query_emb[current_query_num:current_query_num + sent_query_num],
                query['text_mask'][
                current_query_num: current_query_num + sent_query_num])  # [num_of_query_tokens, class_num]
            ologit, opred = torch.max(ologit, 1)  # (810)
            ologits = torch.cat((ologits,ologit))
            logits.append(self.__batch_dist__(
                support_proto, 
                query_emb[current_query_num:current_query_num+sent_query_num],
                query['text_mask'][current_query_num: current_query_num+sent_query_num])) # [num_of_query_tokens, class_num]
            # print(logits.size())

            current_query_num += sent_query_num
            current_support_num += sent_support_num
            Jloss = Jloss + sloss

        ologits = ologits.view(-1,1)
        logits = torch.cat((ologits,logits),2)
        _, pred = torch.max(logits, 1)              #(810)
        return logits, pred
