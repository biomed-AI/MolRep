

import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_contrastive_loss(X, Y, margin=4.0):
    CL_loss_1 = do_CL(X, Y)
    CL_loss_2 = do_CL(Y, X)

    CD_loss_1 = cdist_loss(X, Y, margin)
    CD_loss_2 = cdist_loss(Y, X, margin)
    return 0.9 * (CL_loss_1 + CL_loss_2) / 2 + 0.1 * (CD_loss_1 + CD_loss_2) / 2


def cdist_loss(X, Y, margin):
    dist = torch.cdist(X, Y, p=2)
    pos = torch.diag(dist)

    bs = X.size(0)
    mask = torch.eye(bs)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        mask = mask.to(device)
    neg = (1 - mask) * dist + mask * margin
    neg = torch.relu(margin - neg)
    loss = torch.mean(pos) + torch.sum(neg) / bs / (bs - 1)
    return loss


def do_CL(X, Y, normalize=False, T=0.1):
    if normalize:
        X = F.normalize(X, dim=-1)
        Y = F.normalize(Y, dim=-1)

    criterion = nn.CrossEntropyLoss()
    B = X.size()[0]
    logits = torch.mm(X, Y.transpose(1, 0))  # B*B
    logits = torch.div(logits, T)
    labels = torch.arange(B).long().to(logits.device)  # B*1

    CL_loss = criterion(logits, labels)

    return CL_loss



def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item())/len(pred)


def do_AttrMasking(batch, criterion, node_repr, molecule_atom_masking_model):
    target = batch.mask_node_label[:, 0]
    node_pred = molecule_atom_masking_model(node_repr[batch.masked_atom_indices])
    attributemask_loss = criterion(node_pred.double(), target)
    attributemask_acc = compute_accuracy(node_pred, target)
    return attributemask_loss, attributemask_acc



def do_InfoGraph(node_repr, molecule_repr, batch,
                 criterion, infograph_discriminator_SSL_model):

    summary_repr = torch.sigmoid(molecule_repr)
    positive_expanded_summary_repr = summary_repr[batch.batch]
    shifted_summary_repr = summary_repr[cycle_index(len(summary_repr), 1)]
    negative_expanded_summary_repr = shifted_summary_repr[batch.batch]

    positive_score = infograph_discriminator_SSL_model(
        node_repr, positive_expanded_summary_repr)
    negative_score = infograph_discriminator_SSL_model(
        node_repr, negative_expanded_summary_repr)
    infograph_loss = criterion(positive_score, torch.ones_like(positive_score)) + \
                     criterion(negative_score, torch.zeros_like(negative_score))

    num_sample = float(2 * len(positive_score))
    infograph_acc = (torch.sum(positive_score > 0) +
                     torch.sum(negative_score < 0)).to(torch.float32) / num_sample
    infograph_acc = infograph_acc.detach().cpu().item()

    return infograph_loss, infograph_acc

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

def do_ContextPred(batch, criterion, args, molecule_substruct_model,
                   molecule_context_model, molecule_readout_func, molecule_img_repr=None):

    # creating substructure representation
    substruct_repr = molecule_substruct_model(
        batch.x_substruct, batch.edge_index_substruct,
        batch.edge_attr_substruct)[batch.center_substruct_idx]

    # creating context representations
    overlapped_node_repr = molecule_context_model(
        batch.x_context, batch.edge_index_context,
        batch.edge_attr_context)[batch.overlap_context_substruct_idx]

    # positive context representation
    # readout -> global_mean_pool by default
    context_repr = molecule_readout_func(overlapped_node_repr,
                                         batch.batch_overlapped_context)
    
    # Use image embedding
    if molecule_img_repr is not None and args.use_image:
        # context_repr = torch.cat([context_repr, molecule_img_repr], dim=1)

        # if args.normalize:
        #     context_repr = F.normalize(context_repr, dim=-1)
        #     molecule_img_repr = F.normalize(molecule_img_repr, dim=-1)
        context_repr = 0.8 * context_repr + 0.2 * molecule_img_repr

    # negative contexts are obtained by shifting
    # the indices of context embeddings
    neg_context_repr = torch.cat(
        [context_repr[cycle_index(len(context_repr), i + 1)]
         for i in range(args.contextpred_neg_samples)], dim=0)

    num_neg = args.contextpred_neg_samples
    pred_pos = torch.sum(substruct_repr * context_repr, dim=1)
    pred_neg = torch.sum(substruct_repr.repeat((num_neg, 1)) * neg_context_repr, dim=1)

    loss_pos = criterion(pred_pos.double(),
                         torch.ones(len(pred_pos)).to(pred_pos.device).double())
    loss_neg = criterion(pred_neg.double(),
                         torch.zeros(len(pred_neg)).to(pred_neg.device).double())

    contextpred_loss = loss_pos + num_neg * loss_neg

    num_pred = len(pred_pos) + len(pred_neg)
    contextpred_acc = (torch.sum(pred_pos > 0).float() +
                       torch.sum(pred_neg < 0).float()) / num_pred
    contextpred_acc = contextpred_acc.detach().cpu().item()

    return contextpred_loss, contextpred_acc