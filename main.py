import argparse
import json
from collections import defaultdict, Counter
import os
import torch
import numpy as np
import wandb
import click

from dataset import Dictionary, VQAFeatureDataset 
import base_model
from train import train
import utils
from vqa_debias_loss_functions import *

def parse_args():
    parser = argparse.ArgumentParser("Train the BottomUpTopDown model with a de-biasing method")

    # Data arguments
    parser.add_argument(
        '--cache_features', default=False, action='store_true',
        help="Cache image features in RAM. Requires > 48GB RAM.")
    parser.add_argument(
        '--dataset', default='cpv2',
        choices=["v2", "cpv2", "cpv1"],
        help="Run on VQA-2.0 instead of VQA-CP 2.0"
    )    
    
    # Model & Loss arguments
    parser.add_argument(
        '-p', "--entropy_penalty", default=0.36, type=float,
        help="Entropy regularizer weight for the learned_mixin model")
    parser.add_argument(
        '--mode', default="debias",
        choices=["updn", "debias"],
        help="Kind of ensemble loss to use")
    parser.add_argument(
        '--debias', default="learned_mixin",
        choices=["learned_mixin", "reweight", "bias_product", "none",'focal'],
        help="Kind of ensemble loss to use")
    
    # CSS arguments
    parser.add_argument('--topq', type=int, default=1, choices=[1,2,3], help="num of words to be masked in question")
    parser.add_argument('--keep_qtype', default=True, help="keep qtype or not")
    parser.add_argument('--topv', type=int, default=1, choices=[1,3,5,-1], help="num of object bbox to be masked in image")
    parser.add_argument('--top_hint', type=int, default=9, choices=[9,18,27,36], help="num of hint")
    parser.add_argument('--qvp', type=int, default=5, choices=range(11), help="ratio of q_bias and v_bias")
    
    # Training arguments
    parser.add_argument('--eval_each_epoch', default=True, help="Evaluate every epoch")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='q_v_debias_contras')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    # Contrastive Loss arguments
    parser.add_argument('--margin', type=float, default=0.3, help="Margin of the original contrastive loss")
    parser.add_argument('--contras_loss_weight', type=float, default=2.0)
    parser.add_argument('--contras_loss_type', type=str, default='new',
        choices=['old', 'new'],
        help="Type of contrastive loss: 'old' (cosine) or 'new' (Batch-Contrastive)")
    parser.add_argument('--contras_temperature', type=float, default=0.05,
        help="Temperature parameter for the new contrastive loss")

    # Regularization Loss arguments
    parser.add_argument('--use_ac_loss', action='store_true', default=False, help="Enable answer contrastive loss")
    parser.add_argument('--ac_loss_weight', type=float, default=2.0, help="Answer contrastive loss weight")
    
    parser.add_argument('--use_gd_loss', action='store_true', default=False, help="Enable gradient discrepancy loss")
    parser.add_argument('--gd_loss_weight', type=float, default=6.0, help="Gradient discrepancy loss weight")

    # Optimization arguments
    parser.add_argument('--initial_lr', type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay")
    parser.add_argument('--schedule_method', type=str, default='warm_up', choices=['warm_up', 'no_warm_up'])
    parser.add_argument("--min_lr", type=float, default=3e-6)
    parser.add_argument("--gamma", type=float, default=3.0)
    parser.add_argument('--lr_decay_start', type=int, default=10)
    parser.add_argument('--lr_decay_end', type=int, default=20)
    
    # Curriculum phases
    parser.add_argument('--phase1_epochs', type=int, default=5)
    parser.add_argument('--phase2_epochs', type=int, default=15)
    
    # Wandb
    parser.add_argument('--use_wandb', action='store_true', help="Use Weights & Biases for logging")
    parser.add_argument('--wandb_project', type=str, default='CL-VQA')
    parser.add_argument('--wandb_run_name', type=str, default=None)

    args = parser.parse_args()
    return args

def get_bias(train_dset, eval_dset):
    answer_voc_size = train_dset.num_ans_candidates
    question_type_to_probs = defaultdict(Counter)
    question_type_to_count = Counter()
    
    for ex in train_dset.entries:
        ans = ex["answer"]
        q_type = ans["question_type"]
        question_type_to_count[q_type] += 1
        if ans["labels"] is not None:
            for label, score in zip(ans["labels"], ans["scores"]):
                question_type_to_probs[q_type][label] += score
    
    question_type_to_prob_array = {}
    for q_type, count in question_type_to_count.items():
        prob_array = np.zeros(answer_voc_size, np.float32)
        for label, total_score in question_type_to_probs[q_type].items():
            prob_array[label] += total_score
        prob_array /= count
        question_type_to_prob_array[q_type] = prob_array

    for ds in [train_dset, eval_dset]:
        for ex in ds.entries:
            q_type = ex["answer"]["question_type"]
            ex["bias"] = question_type_to_prob_array[q_type]

def main():
    args = parse_args()
    dataset = args.dataset        
    args.output = os.path.join('logs', args.output)
    
    if not os.path.isdir(args.output):
        utils.create_dir(args.output)
    else:
        if click.confirm(f'Exp directory already exists in {args.output}. Erase?'):
            os.system('rm -r ' + args.output)
            utils.create_dir(args.output)
        else:
            os._exit(1)

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=os.path.basename(args.output),
            config=vars(args),
            dir=args.output
        )

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')

    print("Building train dataset...")
    train_dset = VQAFeatureDataset('train', args, dictionary, dataset=dataset,
                                cache_image_features=args.cache_features)

    print("Building test dataset...")
    eval_dset = VQAFeatureDataset('val', args, dictionary, dataset=dataset,
                                cache_image_features=args.cache_features)

    get_bias(train_dset, eval_dset) 

    # Build the model
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
    
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    
    # Configure debias loss function
    if args.debias == "bias_product":
        model.debias_loss_fn = BiasProduct()
    elif args.debias == "none":
        model.debias_loss_fn = Plain()
    elif args.debias == "reweight":
        model.debias_loss_fn = ReweightByInvBias()
    elif args.debias == "learned_mixin":
        model.debias_loss_fn = LearnedMixin(args.entropy_penalty)
    elif args.debias == 'focal':
        model.debias_loss_fn = Focal()
    else:
        raise RuntimeError(args.mode)

    if args.mode == 'debias':
        model.is_contras = True
    
    with open('util/qid2type_%s.json' % args.dataset, 'r') as f:
        qid2type = json.load(f)

    with open(args.output + "/debias_objective.json", "w") as f:
        js = model.debias_loss_fn.to_json()
        json.dump(js, f, indent=2)

    model = model.cuda()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    train_loader = torch.utils.data.DataLoader(train_dset, args.batch_size, shuffle=True, num_workers=12)
    eval_loader = torch.utils.data.DataLoader(eval_dset, 512, shuffle=False, num_workers=12)

    print("Starting training...")
    args.gradual_warmup_steps = [0.5 * args.initial_lr, 1.0 * args.initial_lr, 1.5 * args.initial_lr, 2.0 * args.initial_lr]
    
    train(model, train_loader, eval_loader, args, qid2type)
    
    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()