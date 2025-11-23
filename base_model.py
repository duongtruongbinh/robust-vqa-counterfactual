import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier, Classifier
from fc import FCNet
import numpy as np
import math
from torch.autograd import Variable

def mask_softmax(x,mask):
    mask=mask.unsqueeze(2).float()
    x2=torch.exp(x-torch.max(x))
    x3=x2*mask
    epsilon=1e-5
    x3_sum=torch.sum(x3,dim=1,keepdim=True)+epsilon
    x4=x3/x3_sum.expand_as(x3)
    return x4

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(GeLU, self).__init__()

    def forward(self, x):
        return gelu(x)

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, dataset_name='cpv2'):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.debias_loss_fn = None
        # self.bias_scale = torch.nn.Parameter(torch.from_numpy(np.ones((1, ), dtype=np.float32)*1.2))
        self.bias_lin = torch.nn.Linear(1024, 1)
        self.is_contras = True
        
        embedding_files = {
            'cpv2': "data/answer_embs_ac.npy",
            'v2': "data/answer_embs_ac_v2.npy"
        }
        embedding_file = embedding_files[dataset_name]
        print(f"Loading answer embeddings from: {embedding_file}")
        
        ans_embed = np.load(embedding_file) + 1e-8
        ans_embed = torch.from_numpy(ans_embed).cuda()  # [num_answers, 300]
        self.ans_embed = nn.functional.normalize(ans_embed, dim=1)
        
        hid_dim = 1024
        # self.type_fc = nn.Sequential(
        #     nn.Linear(hid_dim, hid_dim * 2),
        #     GeLU(),
        #     BertLayerNorm(hid_dim * 2, eps=1e-12),
        #     nn.Linear(hid_dim * 2, 4)
        # )
        self.logit_fc_emb = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, 300)
        )
        
        self.emb_proj = nn.Sequential(
            nn.Linear(300, hid_dim),
            GeLU(),
            BertLayerNorm(hid_dim, eps=1e-12),
            nn.Linear(hid_dim, 300)
        )

    def forward(self, v, q, labels, bias, v_mask):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)

        att = self.v_att(v, q_emb)
        if v_mask is None:
            att = nn.functional.softmax(att, 1)
        else:
            att = mask_softmax(att, v_mask)
            
        v_emb = (att * v).sum(1)  # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr

        logits = self.classifier(joint_repr)
        
        if labels is not None:
            loss = self.debias_loss_fn(joint_repr, logits, bias, labels)
        else:
            loss = None
            
        gen_embs = torch.nn.functional.normalize(self.logit_fc_emb(joint_repr), dim=1)
        all_ans_embs = self.emb_proj(Variable(self.ans_embed))
            
        # type_logit = self.type_fc(joint_repr)
            
        if (self.is_contras is True) and (self.training is True):
            return logits, loss, w_emb, joint_repr, gen_embs, all_ans_embs#, type_logit
        else:
            return logits, loss, w_emb


def build_baseline0(dataset, num_hid, dataset_name='cpv2'):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier, dataset_name)


def build_baseline0_newatt(dataset, num_hid, dataset_name='cpv2'):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier, dataset_name)