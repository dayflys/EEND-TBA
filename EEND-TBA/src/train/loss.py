import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import permutations
from collections import OrderedDict

class SpeakerLoss(nn.Module):
    def __init__(self, all_num_speakers, spk_vector_dim):
        super(SpeakerLoss,self).__init__()
        self.spk_vector_dim = spk_vector_dim
        self.embed = nn.Embedding(all_num_speakers, spk_vector_dim)
        self.alpha = nn.Parameter(torch.rand(1)[0] + torch.Tensor([0.5])[0])
        self.beta = nn.Parameter(torch.rand(1)[0] + torch.Tensor([0.5])[0])

    def forward(self, spksvecs, ys, ts, ss, sigmas, ns, ilens):
        
        spksvecs = [[spkvec[:ilen] for spkvec in spksvec]
                    for spksvec, ilen in zip(spksvecs, ilens)]
        loss = torch.stack(
                [self.spk_loss(spksvec, y[:ilen], t[:ilen], s, sigma, n)
                    for(spksvec, y,  t,  s,  sigma,  n,  ilen)
                    in zip(spksvecs, ys, ts, ss, sigmas, ns, ilens)])
        loss = torch.mean(loss)

        return loss
    
    def set_emb(self, all_num_speakers):
        self.embed = nn.Embedding(all_num_speakers, self.spk_vector_dim)

    def modfy_emb(self, weight):
        self.embed = nn.Embedding.from_pretrained(weight)

    def spk_loss(self, spksvec, y, t, s, sigma, n): #spksvec: (3,T,SE), n: (all_ns,1),y : (T,3)
        embeds = self.embed(n).squeeze() #embeds: (all_ns,SE)
        #z = torch.sigmoid(y.transpose(1, 0)) #z : (3,T)
        z = t.transpose(1, 0) #z : (3,T)

        losses = []
        for spkid, spkvec in enumerate(spksvec): #spkvec: (T,SE) -> (E)
            norm_spkvec_inv = 1.0 / torch.norm(spkvec, dim=1) # norm_spkvec_inv : (SE)
            # Normalize speaker vectors before weighted average
            spkvec = torch.mul(
                    spkvec.transpose(1, 0), norm_spkvec_inv).transpose(1, 0) #spkvec : (T,SE)
            wavg_spkvec = torch.mul(
                    spkvec.transpose(1, 0), z[sigma[spkid]]).transpose(1, 0) #wavg_spkvec : (T,SE)
            sum_wavg_spkvec = torch.sum(wavg_spkvec, dim=0) #sum_wavg_spkvec : (SE)

            #nmz_wavg_spkvec = spkvec / torch.norm(spkvec) # nmz_wavg_spkvec : (SE) -> (E)
            nmz_wavg_spkvec = sum_wavg_spkvec / torch.norm(sum_wavg_spkvec)
            
            nmz_wavg_spkvec = torch.unsqueeze(nmz_wavg_spkvec, 0) #nmz_wavg_spkvec : (1,SE) -> (E)
            
            norm_embeds_inv = 1.0 / torch.norm(embeds, dim=1) # norm_embeds_inv : (all_ns)
            
            embeds = torch.mul(
                    embeds.transpose(1, 0), norm_embeds_inv).transpose(1, 0) #(all_ns, SE) -> (all_ns,E)
            
            dist = torch.cdist(nmz_wavg_spkvec, embeds)[0]
            d = torch.add(
                    torch.clamp(
                        self.alpha,
                        min=sys.float_info.epsilon) * torch.pow(dist, 2),
                        self.beta)

            round_t = torch.round(t.transpose(1, 0)[sigma[spkid]])
            if torch.sum(round_t) > 0:
                loss = -F.log_softmax(-d, 0)[s[sigma[spkid]]]
            else:
                loss = torch.tensor(0.0).to(y.device)
            losses.append(loss)

        return torch.mean(torch.stack(losses))

class DiarizationLoss():
    def pit_loss(self, pred, label):
        """
        Permutation-invariant training (PIT) cross entropy loss function.

        Args:
        pred:  (T,C)-shaped pre-activation values
        label: (T,C)-shaped labels in {0,1}

        Returns:
        min_loss: (1,)-shape mean cross entropy
        label_perms[min_index]: permutated labels
        sigma: permutation
        """

        T = len(label)
        C = label.shape[-1]

        label_perms = [label[..., list(p)] for p
                    in permutations(range(C))]
        losses = torch.stack(
            [F.binary_cross_entropy_with_logits(
                pred[:, ...],
                l[:, ...]) for l in label_perms])
        min_loss = losses.min() * T
        min_index = losses.argmin().detach()
        sigma = list(permutations(range(C)))[min_index]

        return min_loss, label_perms[min_index], sigma


    def batch_pit_loss(self, ys, ts, ilens=None):
        """
        PIT loss over mini-batch.

        Args:
        ys: B-length list of predictions
        ts: B-length list of labels

        Returns:
        loss: (1,)-shape mean cross entropy over mini-batch
        sigmas: B-length list of permutation
        """
        if ilens is None:
            ilens = [t.shape[0] for t in ts]

        loss_w_labels_w_sigmas = [self.pit_loss(y[:ilen, :], t[:ilen, :])
                                for (y, t, ilen) in zip(ys, ts, ilens)]
        losses, label, sigmas = zip(*loss_w_labels_w_sigmas)
        loss = torch.sum(torch.stack(losses))
        n_frames = np.sum([ilen for ilen in ilens])
        loss = loss / n_frames

        return loss, label, sigmas

    def speaker_count_pit_loss(self, xs, n_speakers): #xs : (B,3,1) 
                
        # batch_size = len(xs)
        # # zeros = torch.Size([8, 2(n_speakers)+1, 256])
        # zeros = torch.zeros((batch_size, max(n_speakers) + 1, self.n_units),
        #                     dtype=torch.float32, device=xs[0].device)
        # # attractors: torch.Size([B, n_speakers+1, E])
        # attractors = self.forward(xs, zeros)

        # label = torch.Size([1, B*(n_speakers)])
        labels = torch.cat([torch.tensor([[1.0] * n_spk], dtype=torch.float32)
                            for n_spk in n_speakers], dim=1).to(xs[0].device)
        # logit = torch.Size([1, B*(n_speakers)])
        logit = torch.cat([x.view(1,-1)[:,:n_spk]
                           for x, n_spk in zip(xs, n_speakers)], dim=1)

        loss = nn.functional.binary_cross_entropy_with_logits(logit, labels)

        # # The final attractor does not correspond to a speaker so remove it
        # attractors = [att[:n_spk, :] for att, n_spk in zip(xs, n_speakers)]
        
        return loss
    