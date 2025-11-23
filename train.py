import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
from tqdm import tqdm
import random
import copy
import math

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def compute_score_with_logits(logits, labels):
    logits = torch.argmax(logits, 1)
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def cosine_lr_schedule(epoch, start_epoch, end_epoch, start_lr, min_lr, gamma=3.0):
    if epoch < start_epoch:
        return start_lr
    if epoch >= end_epoch:
        return min_lr

    total = (end_epoch - start_epoch + 1)
    p = (epoch - start_epoch + 1) / float(total)
    p_ease = 1.0 - pow(max(0.0, 1.0 - p), gamma)

    lr = min_lr + 0.5 * (start_lr - min_lr) * (1.0 + math.cos(math.pi * p_ease))
    return max(lr, min_lr)

def compute_contrastive_loss(feature_ori, feature_pos, feature_neg, loss_type='new', temperature=0.05):
    """
    Compute contrastive loss.
    'new': Batch-Contrastive (SimCLR style) - recommended.
    'old': Simple Cosine Similarity.
    """
    if loss_type == 'old':
        pos = torch.cosine_similarity(feature_ori, feature_pos, dim=1)
        neg = torch.cosine_similarity(feature_ori, feature_neg, dim=1)              
        logit = torch.stack([pos, neg], dim=1) # [b, 2]                   
        softmax_logit = nn.functional.softmax(logit, 1) #[b, 2]
        contras_loss = - torch.log(softmax_logit[:, 0])
        contras_loss = contras_loss.mean()
        
    elif loss_type == 'new':
        z_ori = nn.functional.normalize(feature_ori, dim=1)
        z_pos = nn.functional.normalize(feature_pos, dim=1)
        z_neg = nn.functional.normalize(feature_neg, dim=1)
        
        l_pos = (z_ori * z_pos).sum(dim=1).unsqueeze(1)
        l_neg = z_ori @ z_neg.T
        logits = torch.cat([l_pos, l_neg], dim=1) / temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        contras_loss = nn.CrossEntropyLoss()(logits, labels)
        
        softmax_logit = nn.functional.softmax(logits, dim=1)
        
    else:
        raise ValueError(f"Unknown contrastive loss type: {loss_type}")
    
    return contras_loss, softmax_logit

def train(model, train_loader, eval_loader, args, qid2type): 
    dataset = args.dataset
    num_epochs = args.epochs
    mode = args.mode
    run_eval = args.eval_each_epoch
    output = args.output
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0
    
    optim = torch.optim.Adamax(model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
    last_lr = args.gradual_warmup_steps[-1]
    
    s_epoch = 0

    if mode == 'debias':
        print(f"Using contrastive loss type: {args.contras_loss_type}")
        if args.contras_loss_type == 'new':
            print(f"Temperature parameter: {args.contras_temperature}")
    
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.log(vars(args), step=0)
    
    for epoch in range(s_epoch, num_epochs):       
        total_loss = 0
        contras_loss_total = 0
        gd_loss_v_total = 0 
        train_score = 0
        ac_loss_total = 0
        
        t = time.time()        
        
        # Learning Rate Schedule
        if epoch < len(args.gradual_warmup_steps) and args.schedule_method == 'warm_up':
            utils.set_lr(optim, args.gradual_warmup_steps[epoch])   
        elif args.schedule_method == 'warm_up':
            new_lr = cosine_lr_schedule(epoch, args.lr_decay_start, args.lr_decay_end, last_lr, args.min_lr, args.gamma)
            utils.set_lr(optim, new_lr)      

        for i, (v, q, a, b, hintscore, type_mask, notype_mask, q_mask, orig_top_ans_emb) \
                in tqdm(enumerate(train_loader), ncols=100, desc=f"Epoch {epoch+1}", total=len(train_loader)):

            v = Variable(v).cuda().requires_grad_()
            q = Variable(q).cuda()
            q_mask = Variable(q_mask).cuda()
            a = Variable(a).cuda()
            b = Variable(b).cuda()
            type_mask = Variable(type_mask).float().cuda()
            notype_mask = Variable(notype_mask).float().cuda()
            orig_top_ans_emb = Variable(orig_top_ans_emb).cuda()
            hintscore = Variable(hintscore).cuda()

            random_num = random.randint(1, 10)
            sen_mask = type_mask if args.keep_qtype else notype_mask
                
            if random_num <= args.qvp: 
                # --- Question Debiasing Branch ---
                padding_idx = 18456
                q_orig_len = (q != padding_idx).sum(dim=1)
                q_lens_orig = sorted([(q_orig_len[i].item(), i) for i in range(q_orig_len.shape[0])], key=lambda x: x[0], reverse=True)
                
                v_orig, q_orig, a_orig, b_orig, sen_mask_sorted, q_mask_sorted, orig_top_ans_emb_sorted = \
                    sort_sample_orig(q_lens_orig, v, q, a.data, b, sen_mask, q_mask, orig_top_ans_emb)
                v_orig = v_orig.requires_grad_()
                
                # 1. First Pass: Original
                pred, loss, word_emb, feature_ori, gen_embs_ori, all_ans_embs = model(v_orig, q_orig, a_orig, b_orig, None)                 
                
                if torch.isnan(loss).any(): raise ValueError("NaN loss")
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()
                
                total_loss += loss.item() * q.size(0)
                train_score += compute_score_with_logits(pred, a_orig.data).sum()

                # Phase 2: Contrastive Learning
                if epoch + 1 > args.phase1_epochs:
                    # Generate Positive/Negative Samples via Gradient Attribution (CAM)
                    word_grad = torch.autograd.grad((pred * (a_orig > 0).float()).sum(), word_emb, create_graph=True)[0]
                    word_grad_cam = word_grad.sum(2).masked_fill(sen_mask_sorted == 0, -1e9)
                    word_grad_cam_softmax = torch.softmax(word_grad_cam, dim=1)
                    w_ind = word_grad_cam_softmax.sort(1, descending=True)[1][:, :args.topq]                    
                    
                    # Create Counterfactual Question
                    q2 = copy.deepcopy(q_mask_sorted)
                    m1 = copy.deepcopy(sen_mask_sorted)
                    m1.scatter_(1, w_ind, 0)
                    m2 = 1 - m1
                    m3 = m1 * 18455 # Mask token
                    q2 = q2 * m2.long() + m3.long()
                    
                    # 2. Second Pass: Positive (V, Q+)
                    pred_pos, _, _, feature_pos, gen_embs_pos, _ = model(v_orig, q2, None, b_orig, None) 
                        
                    # Create Negative Answer for Negative Pass
                    pred_ind = torch.argsort(pred_pos, 1, descending=True)[:, :5]
                    false_ans = torch.ones_like(pred_pos).cuda()
                    false_ans.scatter_(1, pred_ind, 0)
                    a2 = a_orig * false_ans
                    
                    q3 = copy.deepcopy(q_orig)
                    q3.scatter_(1, w_ind, 18455) # Mask critical words

                    # 3. Third Pass: Negative (V, Q-)
                    pred_neg, loss_neg, _, feature_neg, gen_embs_neg, _ = model(v_orig, q3, a2, b_orig, None)

                    # Contrastive Loss
                    contras_loss, softmax_logit = compute_contrastive_loss(
                        feature_ori, feature_pos, feature_neg, 
                        loss_type=args.contras_loss_type, 
                        temperature=args.contras_temperature
                    )
                    contras_loss_total += contras_loss.item() * q.size(0)                     
                    loss = loss_neg + args.contras_loss_weight * contras_loss
                    
                    # Phase 3: AC Loss
                    if epoch + 1 > args.phase2_epochs and args.use_ac_loss:
                        if all_ans_embs.dim() == 2: all_ans_embs = all_ans_embs.unsqueeze(0).expand(gen_embs_ori.size(0), -1, -1)
                        
                        z = torch.cosine_similarity(gen_embs_ori.unsqueeze(1), all_ans_embs, dim=2)
                        z_pos = torch.cosine_similarity(gen_embs_pos.unsqueeze(1), all_ans_embs, dim=2)
                        z_neg = torch.cosine_similarity(gen_embs_neg.unsqueeze(1), all_ans_embs, dim=2)
                        z_all = torch.stack([z, z_pos, z_neg], dim=1) # [B, 3, D]
                        
                        prob_ori = nn.functional.softmax(z_all, dim=2)[:, 0, :]
                        pos_mask = (a_orig > 0).float()
                        ac_loss = -torch.log((prob_ori * pos_mask).sum(dim=1) + 1e-8).mean()
                        
                        ac_loss_total += ac_loss.item() * q.size(0)                              
                        loss += ac_loss * args.ac_loss_weight

                    if torch.isnan(loss).any(): raise ValueError("NaN loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                    optim.step()
                    optim.zero_grad()
                    total_loss += loss.item() * q.size(0)
            
            else: 
                # --- Vision Debiasing Branch ---
                padding_idx = 18456
                q_orig_len = (q != padding_idx).sum(dim=1)
                q_lens_orig = sorted([(q_orig_len[i].item(), i) for i in range(q_orig_len.shape[0])], key=lambda x: x[0], reverse=True)
                
                v_orig, q_orig, a_orig, b_orig, hintscore_sorted, orig_top_ans_emb_sorted = \
                    sort_sample_orig(q_lens_orig, v, q, a.data, b, hintscore, orig_top_ans_emb)
                v_orig = v_orig.requires_grad_()

                # 1. First Pass: Original
                pred, loss, _, feature_ori, gen_embs_ori, all_ans_embs = model(v_orig, q_orig, a_orig, b_orig, None)
                                       
                if torch.isnan(loss).any(): raise ValueError("NaN loss")
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()               

                total_loss += loss.item() * q.size(0)
                train_score += compute_score_with_logits(pred, a_orig.data).sum()

                # Phase 2: Contrastive Learning
                if epoch + 1 > args.phase1_epochs:
                    visual_grad = torch.autograd.grad((pred * (a_orig > 0).float()).sum(), v_orig, create_graph=True)[0]  
                    visual_grad_cam = visual_grad.sum(2)
                    
                    # Determine critical visual regions
                    v_mask = torch.zeros(v.shape[0], 36).cuda()
                    _, hint_ind = hintscore_sorted.sort(1, descending=True)
                    v_ind = hint_ind[:, :args.top_hint]
                    v_grad = visual_grad_cam.gather(1, v_ind)

                    if args.topv == -1:
                        v_grad_score, _ = v_grad.sort(1, descending=True)
                        v_grad_score = nn.functional.softmax(v_grad_score*10, dim=1)
                        v_grad_mask = (torch.cumsum(v_grad_score, dim=1) <= 0.65).long()
                        v_grad_mask[:, 0] = 1
                        v_mask_ind = v_grad_mask * v_ind
                        for x in range(a.shape[0]):
                            num = len(torch.nonzero(v_grad_mask[x]))
                            v_mask[x].scatter_(0, v_mask_ind[x, :num], 1)
                    else:
                        v_grad_ind = v_grad.sort(1, descending=True)[1][:, :args.topv]
                        v_star = v_ind.gather(1, v_grad_ind)
                        v_mask.scatter_(1, v_star, 1)
                                                
                    # 2. Second Pass: Positive (V+, Q)
                    pred_pos, _, _, feature_pos, gen_embs_pos, _ = model(v_orig, q_orig, None, b_orig, v_mask)
                    v_mask_neg = 1 - v_mask

                    pred_ind = torch.argsort(pred_pos, 1, descending=True)[:, :5]
                    false_ans = torch.ones_like(pred_pos).cuda()
                    false_ans.scatter_(1, pred_ind, 0)
                    a2 = a_orig * false_ans

                    # 3. Third Pass: Negative (V-, Q)
                    pred_neg, loss_neg, _, feature_neg, gen_embs_neg, _ = model(v_orig, q_orig, a2, b_orig, v_mask_neg)
                    
                    # Contrastive Loss
                    contras_loss, softmax_logit = compute_contrastive_loss(
                        feature_ori, feature_pos, feature_neg, 
                        loss_type=args.contras_loss_type, 
                        temperature=args.contras_temperature
                    )                
                    contras_loss_total += contras_loss.item() * q.size(0)    
                    loss = loss_neg + args.contras_loss_weight * contras_loss
                    
                    # Phase 3: GD Loss
                    if epoch + 1 > args.phase2_epochs and args.use_gd_loss:
                        vqa_grad_pos = torch.autograd.grad(pred_pos.sum(), v_orig, create_graph=True)[0]
                        vqa_grad_neg = torch.autograd.grad(pred_neg.sum(), v_orig, create_graph=True)[0]
                        
                        rwd = -torch.abs(vqa_grad_pos - vqa_grad_neg).mean() 
                        log_prob = torch.log(softmax_logit[:, 0] + 1e-8)
                        
                        gd_loss_v = -(rwd * log_prob).mean()
                        
                        gd_loss_v_total += gd_loss_v.item() * q.size(0)
                        loss += args.gd_loss_weight * gd_loss_v    

                    if torch.isnan(loss).any(): raise ValueError("NaN loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                    optim.step()
                    optim.zero_grad()
                    total_loss += loss.item() * q.size(0)

        # Logging
        total_loss /= len(train_loader.dataset) * (1 if mode=='updn' else 2)
        train_score = 100 * train_score / len(train_loader.dataset)
        contras_loss_avg = contras_loss_total / len(train_loader.dataset) if epoch+1 > args.phase1_epochs else 0
        ac_loss_avg = ac_loss_total / len(train_loader.dataset) if (epoch+1 > args.phase2_epochs and args.use_ac_loss) else 0
        gd_loss_v_avg = gd_loss_v_total / len(train_loader.dataset) if (epoch+1 > args.phase2_epochs and args.use_gd_loss) else 0

        if run_eval:
            model.train(False)
            results = evaluate(model, eval_loader, qid2type)
            results["epoch"] = epoch + 1
            results["train_loss"] = total_loss
            results["train_score"] = train_score
            model.train(True)
            eval_score = results["score"]

        # Logging to W&B and File
        if args.use_wandb and WANDB_AVAILABLE:
            log_dict = {
                "train/total_loss": total_loss, "train/score": train_score,
                "train/contras_loss": contras_loss_avg, "train/ac_loss": ac_loss_avg, "train/gd_loss": gd_loss_v_avg,
                "train/lr": optim.param_groups[0]['lr']
            }
            if run_eval:
                log_dict.update({
                    "eval/score": eval_score * 100, "eval/yesno": results['score_yesno'] * 100,
                    "eval/other": results['score_other'] * 100, "eval/number": results['score_number'] * 100
                })
            wandb.log(log_dict, step=epoch + 1)

        logger.write('Epoch %d (Time: %.2f) | Train Loss: %.2f | Score: %.2f' % (epoch + 1, time.time() - t, total_loss, train_score))
        if run_eval:
            logger.write('\tEval Score: %.2f (YN: %.2f, Other: %.2f, Num: %.2f)' % (
                100*eval_score, 100*results['score_yesno'], 100*results['score_other'], 100*results['score_number']))

        # Save Checkpoints
        if eval_score > best_eval_score:
            utils.save_model(os.path.join(output, 'model.pth'), model, epoch, eval_score, optim)
            best_eval_score = eval_score

        for chk_epoch in [args.phase1_epochs, 10, args.phase2_epochs]:
            if epoch + 1 == chk_epoch:
                utils.save_model(os.path.join(output, f"model_{chk_epoch}_{dataset}.pth"), model, epoch, eval_score, optim)

def evaluate(model, dataloader, qid2type):
    score = 0
    upper_bound = 0
    score_yesno, score_number, score_other = 0, 0, 0
    total_yesno, total_number, total_other = 0, 0, 0

    for v, q, a, b, qids, _ in tqdm(dataloader, ncols=100, desc="eval"):
        v = Variable(v, requires_grad=False).cuda()
        q = Variable(q, requires_grad=False).cuda()
        pred, _, _ = model(v, q, None, None, None)
        
        batch_score = compute_score_with_logits(pred, a.cuda()).cpu().numpy().sum(1)
        score += batch_score.sum()
        upper_bound += (a.max(1)[0]).sum()
        qids = qids.detach().cpu().int().numpy()
        
        for j, qid in enumerate(qids):
            typ = qid2type[str(qid)]
            if typ == 'yes/no':
                score_yesno += batch_score[j]
                total_yesno += 1
            elif typ == 'other':
                score_other += batch_score[j]
                total_other += 1
            elif typ == 'number':
                score_number += batch_score[j]
                total_number += 1

    return dict(
        score=score / len(dataloader.dataset),
        upper_bound=upper_bound / len(dataloader.dataset),
        score_yesno=score_yesno / total_yesno if total_yesno else 0,
        score_other=score_other / total_other if total_other else 0,
        score_number=score_number / total_number if total_number else 0,
    )

def sort_sample_orig(order, *args):
    args = [[arg[q_len_a[1]] for q_len_a in order] for arg in args]
    args = [Variable(torch.stack(arg, dim=0).cuda()) for arg in args]
    return args