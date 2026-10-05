###############################################################################
# 
#   This file was imported from the DETR codebase.
#   https://github.com/facebookresearch/detr/blob/main/models/detr.py
# 
#   We only imported a minimal amount of tools to improve readability.
#   We added contrastive loss and attribute specific losses to the implementation and performed small edits.
#
###############################################################################

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch.nn.functional as F
import torch.nn as nn
from salma.utils.misc import is_dist_avail_and_initialized, get_world_size
from salma.utils.box_ops  import generalized_box_iou, box_cxcywh_to_xyxy
import torch
import logging
logger = logging.getLogger(__name__)

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, 
                 matcher, 
                 weight_dict, 
                 losses, 
                 device,
                 eos_coef: float = 0.1, 
                 slot_contrast_tau = 0.1, 
                 ):
        """ Create the criterion.
        Parameters:
            num_species: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([eos_coef]).to(device))

        self.slot_contrast_tau = slot_contrast_tau
    
    def loss_is_animal(self, outputs, targets, indices, **kwargs):

        src_logits = outputs["pred_is_animal"]
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.zeros(src_logits.shape[:2], dtype=torch.float, device=src_logits.device)
        target_classes[idx] += 1.0

        loss_bce = self.bce_loss(
            src_logits,
            target_classes,
        )

        return {"loss_object": loss_bce}

    def loss_species(self, outputs, targets, indices, **kwargs):

        src_logits = outputs["pred_species"]
        idx = self._get_src_permutation_idx(indices)

        if idx[0].numel() == 0:
            return {"loss_species": src_logits.sum() * 0.0}

        # logits only at matched queries
        src_logits_matched = src_logits[idx]  # [num_matched, C]

        target_classes = torch.cat(
            [t["species"][J] for t, (_, J) in zip(targets, indices)]
        )

        loss_ce = self.ce_loss(
            src_logits_matched,
            target_classes,
        )

        return {"loss_species": loss_ce}


    def loss_dage(self, outputs, targets, indices, **kwargs):
        assert "pred_dages" in outputs

        src_logits = outputs["pred_dages"]
        idx = self._get_src_permutation_idx(indices)

        if idx[0].numel() == 0:
            return {"loss_dage": src_logits.sum() * 0.0}

        # logits only at matched queries
        src_logits_matched = src_logits[idx]  # [num_matched, C]

        target_classes = torch.cat(
            [t["dages"][J] for t, (_, J) in zip(targets, indices)]
        )

        loss_ce = self.ce_loss(
            src_logits_matched,
            target_classes,
        )

        return {"loss_dage": loss_ce}
    
    def loss_dsex(self, outputs, targets, indices, **kwargs):
        assert "pred_dsexes" in outputs

        src_logits = outputs["pred_dsexes"]
        idx = self._get_src_permutation_idx(indices)

        if idx[0].numel() == 0:
            return {"loss_dsex": src_logits.sum() * 0.0}

        # logits only at matched queries
        src_logits_matched = src_logits[idx]  # [num_matched, C]

        target_classes = torch.cat(
            [t["dsexes"][J] for t, (_, J) in zip(targets, indices)]
        )

        loss_ce = self.ce_loss(
            src_logits_matched,
            target_classes,
        )

        return {"loss_dsex": loss_ce}

    def loss_activities(self, outputs, targets, indices, **kwargs):
        assert "pred_activities" in outputs

        src_logits = outputs["pred_activities"]
        idx = self._get_src_permutation_idx(indices)

        if idx[0].numel() == 0:
            return {"loss_activities": src_logits.sum() * 0.0}

        # logits only at matched queries
        src_logits_matched = src_logits[idx]  # [num_matched, C]

        target_classes = torch.cat(
            [t["activities"][J] for t, (_, J) in zip(targets, indices)]
        )

        loss_ce = self.ce_loss(
            src_logits_matched,
            target_classes,
        )

        return {"loss_activities": loss_ce}

    def loss_actions(self, outputs, targets, indices, **kwargs):
        assert "pred_actions" in outputs

        src_logits = outputs["pred_actions"]  # [B, Q, C_actions+1]
        idx = self._get_src_permutation_idx(indices)

        if idx[0].numel() == 0:
            return {"loss_actions": src_logits.sum() * 0.0}

        # matched logits only
        src_logits_matched = src_logits[idx]  # [num_matched, C]

        # shape: [num_matched, 2]
        target_actions = torch.cat(
            [t["actions"][J] for t, (_, J) in zip(targets, indices)]
        )

        # Scale targets if they have two attribute
        denom = torch.sum(target_actions, axis=-1, keepdims=True)
        target_actions = target_actions / (denom + 1e-8)

        loss_ce = self.ce_loss(
            src_logits_matched,
            target_actions,
        )

        return {"loss_actions": loss_ce}
    
    def loss_weather(self, outputs, targets, weather_targets, **kwargs):
        assert "pred_weather" in outputs
        src_logits = outputs["pred_weather"].squeeze(1) # (B, num_weather)
        weather_targets = torch.stack(weather_targets)
        
        loss_ce = self.ce_loss(
            src_logits,
            weather_targets,
        )
        return {"loss_weather": loss_ce}


    def loss_boxes(self, outputs, targets, indices, num_boxes, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / (num_boxes + 1e-8)

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
    
    def loss_slot_contrast(self, outputs, **kwargs):
        """
        Slot-slot temporal contrastive loss (TC-Slot style). 
        The implementation is strongly inspired from the Slot_Slot_Contrastive_Loss.
        (https://github.com/martius-lab/slotcontrast/blob/main/slotcontrast/losses.py)

        For each video in the batch:
            - Positive = same query index across adjacent frames (t and t+1)
            - Negatives = all other slots (different queries or different videos)

        Args:
            outputs['animal_queries']: Tensor [B, T, Q, D]

        Returns:
            dict(loss_cont = scalar)
        """
        if 'animal_queries' not in outputs:
            raise KeyError("Outputs must contain 'animal_queries' for slot contrast loss")

        feats = outputs['animal_queries']  # [B, T, Q, D]

        # Compute l2-norm manually
        l2_norm = torch.linalg.vector_norm(feats, ord=2, dim=-1, keepdims=True)
        feats = (feats / (l2_norm + 1e-6))
        l2_loss = ((l2_norm - 1.0) ** 2).mean()

        B, T, Q, D = feats.shape
        tau = getattr(self, "slot_contrast_tau", 0.1)

        # Normalize slot features
        feats = feats.transpose(0, 1).flatten(1, 2)  # [T, BQ, D]
        if is_dist_avail_and_initialized():
            feats = self.gather_queries(feats)  # [T, BQ * world_size, D]
        
        # Now BQ_global = BQ * world_size
        BQ_global = feats.shape[1]

        # Prepare slots at t and t+1
        s1 = feats[:-1, :, :]  # [T-1, BQ * world_size, D]
        s2 = feats[1:, :, :]    # [T-1, BQ * world_size, D]

        # Compute similarity matrix for each time step
        # Forward similarity: t -> t+1
        ss_fwd = torch.matmul(s1, s2.transpose(-2, -1)) / tau  # [T-1, BQ * world_size, BQ * world_size]
        ss_fwd = torch.nan_to_num(ss_fwd, nan=0.0, posinf = 1e6, neginf = -1e6) # Very rare nan after gathering features

        # Backward similarity: t+1 -> t
        ss_bwd = ss_fwd.transpose(-2, -1)

        # Target = identity (same slot index) but not one hot encoded.
        target = torch.arange(BQ_global, device=feats.device).expand(T-1, -1)

        # Use CrossEntropyLoss (expects [T-1, BQ, BQ] logits, target indices).
        # This means that the product of queries at the same index, adjacent 
        # in time should be close to 1 and the others should be close to 0.
        loss_fwd = nn.CrossEntropyLoss()(ss_fwd, target)
        loss_bwd = nn.CrossEntropyLoss()(ss_bwd, target)

        loss = 0.5 * (loss_fwd + loss_bwd)

        return {"loss_cont": loss, "loss_l2_norm": l2_loss}

    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'species': self.loss_species,
            'activities': self.loss_activities,
            'actions': self.loss_actions,
            'dages': self.loss_dage,
            'dsexes': self.loss_dsex,
            'boxes': self.loss_boxes,
            'slot_contrast': self.loss_slot_contrast,
            'weather': self.loss_weather,
            'is_animal': self.loss_is_animal
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets=targets, indices=indices, num_boxes=num_boxes, **kwargs)

    def forward(self, outputs, targets, weather_targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["species"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, weather_targets=weather_targets))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def gather_queries(self, t: torch.Tensor):
        """
        Gathers a tensor from all processes and concatenates along the first dimension.
        No gradients flow through all_gather, so we use autograd-friendly version.
        """
        world_size = torch.distributed.get_world_size()
        if world_size == 1:
            return t

        tensors = [torch.zeros_like(t) for _ in range(world_size)]
        torch.distributed.all_gather(tensors, t)


        # Replace the slot corresponding to the current rank with the original tensor
        # to maintain gradient flow
        rank = torch.distributed.get_rank()
        tensors[rank] = t

        return torch.cat(tensors, dim=1)