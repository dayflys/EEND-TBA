import os
import numpy as np
from functools import partial
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data as td
from ..models.utils import fix_state_dict


class TrainHandler:
    def __init__(self, args, model, loaders, sampler, optimizer, scheduler, logger):
        self.args = args
        self.model = model
        self.train_loader, self.val_loader = loaders[0], loaders[1]
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sampler = sampler
        self.logger = logger
        self.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
        self.idx_for_log = len(self.train_loader) // self.args.log_report_num +1
        self.pit_loss_ratio = abs(1 - self.args.spk_loss_ratio-self.args.spk_count_loss_ratio)
        self.spk_loss_ratio = self.args.spk_loss_ratio
        self.spk_count_loss = self.args.spk_count_loss_ratio

        if args.initmodel: # use init model
            self.init_model_states()

    def run(self):
        if not os.path.exists(self.args.model_save_dir):
            os.mkdir(self.args.model_save_dir)
        self.global_step = 0
        self.local_step = 0
        with tqdm(total = self.args.max_epochs, ncols = 150) as pbar:
            for epoch in range(0, self.args.max_epochs):
                self.train(epoch)
                self.eval(epoch)
                self.save(epoch)
                pbar.update()
                dist.barrier()
        if self.logger:
            self.logger.finish()


    def train(self, epoch):
        self.model.train()
        if not self.args.initmodel:
            self.sampler.set_epoch(epoch)

        _loss = 0
        _loss_pit = 0
        _loss_spk = 0
        _loss_spk_count = 0
        with tqdm(total = len(self.train_loader), ncols = 150) as pbar:
            for step, inputs in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                description = '%s epoch: %d '%(self.args.name, epoch)
                xs = inputs[0].to(self.device)
                ts = inputs[1].to(self.device)
                ss = inputs[2].to(self.device)
                ns = inputs[3].to(self.device) #ns : (B, all_ns, 1)
                ilens = inputs[4].to(self.device)
                ilens = [ilen.item() for ilen in ilens]
                
                ys, spksvecs, pit_loss, spk_loss,speaker_count_loss, label = self.model(xs, ts, ss, ns, ilens)

                loss = pit_loss * self.pit_loss_ratio + spk_loss * self.spk_loss_ratio + speaker_count_loss*self.spk_count_loss
                loss.backward()
                
                description += 'loss:%.3f '%(loss)
                description += 'pit_loss:%.3f '%(pit_loss)
                description += 'spk_loss:%.3f '%(spk_loss)
                description += 'spk_count_loss:%.3f '%(speaker_count_loss)
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    if self.args.optimizer_name == 'noam':
                        self.scheduler.step()
                    if self.args.gradclip > 0:
                        nn.utils.clip_grad_value_(self.model.parameters(), self.args.gradclip)
        

                self.global_step+=1
                _loss += loss.item()
                _loss_pit += pit_loss.item()
                _loss_spk += spk_loss.item()
                _loss_spk_count += speaker_count_loss.item()
 
                if self.global_step % self.idx_for_log == 0:
                    if not self.args.initmodel:
                        _loss_tensor = torch.tensor(_loss).to(self.args.device)
                        gathered_loss_tensor = [torch.zeros_like(_loss_tensor) for _ in range(dist.get_world_size())]
                        dist.all_gather(gathered_loss_tensor, _loss_tensor)
                        _loss = torch.stack(gathered_loss_tensor).mean().item()

                        _loss_loss_pit = torch.tensor(_loss_pit).to(self.args.device)
                        gathered_loss_loss_pit = [torch.zeros_like(_loss_loss_pit) for _ in range(dist.get_world_size())]
                        dist.all_gather(gathered_loss_loss_pit, _loss_loss_pit)
                        _loss_pit = torch.stack(gathered_loss_loss_pit).mean().item()
 
                        _loss_spk_tensor = torch.tensor(_loss_spk).to(self.args.device)
                        gathered_loss_spk_tensor = [torch.zeros_like(_loss_spk_tensor) for _ in range(dist.get_world_size())]
                        dist.all_gather(gathered_loss_spk_tensor, _loss_spk_tensor)
                        _loss_spk = torch.stack(gathered_loss_spk_tensor).mean().item()
                        
                        _loss_spk_count_tensor = torch.tensor(_loss_spk_count).to(self.args.device)
                        gathered_loss_spk_count_tensor = [torch.zeros_like(_loss_spk_count_tensor) for _ in range(dist.get_world_size())]
                        dist.all_gather(gathered_loss_spk_count_tensor, _loss_spk_count_tensor)
                        _loss_spk_count = torch.stack(gathered_loss_spk_count_tensor).mean().item()
                
                    if self.args.flag_parent:
                        self.train_logging(_loss, _loss_pit, _loss_spk,_loss_spk_count)
                    _loss = 0
                    _loss_pit = 0
                    _loss_spk = 0
                    self.local_step += 1
                pbar.set_description(description)
                pbar.update(1)
        #if self.args.optimizer_name == 'adam':
        #    self.scheduler.step()

    def train_logging(self, _loss, _loss_pit, _loss_spk,_loss_spk_count):
        if self.args.optimizer_name == 'noam': 
            self.logger.log_metric("lrate", self.scheduler.get_lr()[0], log_step = self.local_step)
        else:
            for p_group in self.optimizer.param_groups:
                lr = p_group['lr']
                self.logger.log_metric("lrate", lr, log_step = self.local_step)
                break
                
        self.logger.log_metric("train_loss", _loss/self.idx_for_log, log_step = self.local_step)
        self.logger.log_metric("train_loss_pit", _loss_pit/self.idx_for_log, log_step = self.local_step)
        self.logger.log_metric("train_loss_spk", _loss_spk/self.idx_for_log, log_step = self.local_step)
        self.logger.log_metric("train_loss_spk_count", _loss_spk_count/self.idx_for_log, log_step = self.local_step)


    def eval(self, epoch):
        self.model.eval()
        with torch.no_grad():
            cnt = 0
            _loss = 0
            _loss_pit = 0
            _loss_spk = 0
            _loss_spk_count = 0
            dev_stats_avg = {}
            with tqdm(total = len(self.val_loader), ncols = 150) as pbar:
                for step, inputs in enumerate(self.val_loader):
                    xs = inputs[0].to(self.device)
                    ts = inputs[1].to(self.device)
                    ss = inputs[2].to(self.device)
                    ns = inputs[3].to(self.device)
                    ilens = inputs[4].to(self.device)
                    ilens = [ilen.item() for ilen in ilens]

                    ys, spksvecs, pit_loss, spk_loss,speaker_count_loss, label = self.model(xs, ts, ss, ns, ilens)

                    loss = pit_loss * self.pit_loss_ratio + spk_loss * self.spk_loss_ratio + speaker_count_loss*self.spk_count_loss

                    _loss += loss.item()
                    _loss_pit += pit_loss.item()
                    _loss_spk += spk_loss.item()
                    _loss_spk_count += speaker_count_loss.item()    
                                        
                    stats = self.report_diarization_error(ys, label)
        
                    for k, v in stats.items():
                        if not self.args.initmodel:
                            v_tensor = torch.tensor(v).to(self.args.device)
                            gathered_tensor = [torch.zeros_like(v_tensor) for _ in range(dist.get_world_size())]
                            dist.all_gather(gathered_tensor, v_tensor)
                            v = torch.sum(torch.stack(gathered_tensor)).item()
                        if self.args.flag_parent:
                            dev_stats_avg[k] = dev_stats_avg.get(k, 0) + v

                    cnt += 1
                    pbar.update()

        if self.args.flag_parent:
            self.eval_logging(epoch, cnt, _loss, _loss_pit, _loss_spk,_loss_spk_count, dev_stats_avg)
        

    def eval_logging(self, epoch, cnt, _loss, _loss_pit, _loss_spk,_loss_spk_count, stats_avg):
        stats_avg = {k:v/cnt for k,v in stats_avg.items()}
        stats_avg['DER'] = stats_avg['diarization_error'] / stats_avg['speaker_scored'] * 100
        stats_avg['DER_miss'] = stats_avg['speaker_miss'] / stats_avg['speaker_scored'] * 100
        stats_avg['DER_FA'] = stats_avg['speaker_falarm'] / stats_avg['speaker_scored'] * 100
        stats_avg['DER_confusion'] = stats_avg['speaker_error'] / stats_avg['speaker_scored'] * 100
        for k in stats_avg.keys():
            stats_avg[k] = round(stats_avg[k], 2)

        self.logger.log_metric("dev_loss", _loss/len(self.val_loader), epoch_step = epoch)
        self.logger.log_metric("dev_loss_pit", _loss_pit/len(self.val_loader), epoch_step = epoch)
        self.logger.log_metric("dev_loss_spk", _loss_spk/len(self.val_loader), epoch_step = epoch)
        self.logger.log_metric("dev_loss_spk_count", _loss_spk_count/len(self.val_loader), epoch_step = epoch)
        self.logger.log_metric("dev_DER", stats_avg['DER'], epoch_step = epoch)
        self.logger.log_metric("dev_DER_miss", stats_avg['DER_miss'], epoch_step = epoch)
        self.logger.log_metric("dev_DER_FA", stats_avg['DER_FA'], epoch_step = epoch)
        self.logger.log_metric("dev_DER_confusion", stats_avg['DER_confusion'], epoch_step = epoch)
        

    def save(self, epoch):
        model_filename = os.path.join(self.args.model_save_dir, self.args.model_filename.format(epoch))
        torch.save(self.model.state_dict(), model_filename)

    def save_train_model(self):
        model_filename = os.path.join(self.args.model_save_dir, self.args.avg_model_filename)
        torch.save(self.model.state_dict(), model_filename)

    def calc_diarization_error(self, pred, label, label_delay=0):
        """
        Calculates diarization error stats for reporting.

        Args:
        pred (torch.FloatTensor): (T,C)-shaped pre-activation values
        label (torch.FloatTensor): (T,C)-shaped labels in {0,1}
        label_delay: if label_delay == 5:
            pred: 0 1 2 3 4 | 5 6 ... 99 100 |
            label: x x x x x | 0 1 ... 94  95 | 96 97 98 99 100
            calculated area: | <------------> |

        Returns:
        res: dict of diarization error stats
        """
        label = label[:len(label) - label_delay, ...]
        decisions = torch.sigmoid(pred[label_delay:, ...]) > 0.5
        n_ref = label.sum(axis=-1).long()
        n_sys = decisions.sum(axis=-1).long()
        res = {}
        res['speech_scored'] = (n_ref > 0).sum()
        res['speech_miss'] = ((n_ref > 0) & (n_sys == 0)).sum()
        res['speech_falarm'] = ((n_ref == 0) & (n_sys > 0)).sum()
        res['speaker_scored'] = (n_ref).sum()
        res['speaker_miss'] = torch.max((n_ref - n_sys), torch.zeros_like(n_ref)).sum()
        res['speaker_falarm'] = torch.max((n_sys - n_ref), torch.zeros_like(n_ref)).sum()
        n_map = ((label == 1) & (decisions == 1)).sum(axis=-1)
        res['speaker_error'] = (torch.min(n_ref, n_sys) - n_map).sum()
        res['correct'] = (label == decisions).sum() / label.shape[1]
        res['diarization_error'] = (
            res['speaker_miss'] + res['speaker_falarm'] + res['speaker_error'])
        res['frames'] = len(label)
        return res


    def report_diarization_error(self, ys, labels):
        """
        Reports diarization errors
        Should be called with torch.no_grad

        Args:
        ys: B-length list of predictions (torch.FloatTensor)
        labels: B-length list of labels (torch.FloatTensor)
        """
        stats_avg = {}
        cnt = 0
        for y, t in zip(ys, labels):
            stats = self.calc_diarization_error(y, t)
            for k, v in stats.items():
                stats_avg[k] = stats_avg.get(k, 0) + float(v)
            cnt += 1
        
        stats_avg = {k:v/cnt for k,v in stats_avg.items()}
        return stats_avg        


    def init_model_states(self):        
        model_parameter_dict = torch.load(self.args.initmodel)
        fix_model_parameter_dict = fix_state_dict(model_parameter_dict)

        old_all_n_speakers = fix_model_parameter_dict["speaker_loss.embed.weight"].shape[0]
        self.model.speaker_loss.set_emb(old_all_n_speakers)
        print("old all_n_speakers : {}".format(old_all_n_speakers))
        
        self.model.load_state_dict(fix_model_parameter_dict)
        npz = np.load(self.args.spkv_lab)
        spkvecs = npz['arr_0']
        spklabs = npz['arr_1']
        spkidx_tbl = npz['arr_2']
        # init
        spk_num = len(np.unique(spklabs))
        fet_dim = spkvecs.shape[1]
        fet_arr = np.zeros([spk_num, fet_dim])
        # sum
        bs = spklabs.shape[0]
        for i in range(bs):
            if spkidx_tbl[spklabs[i]] == -1:
                raise ValueError(spklabs[i])
            fet_arr[spkidx_tbl[spklabs[i]]] += spkvecs[i]
        # normalize
        for spk in range(spk_num):
            org = fet_arr[spk]
            norm = np.linalg.norm(org, ord=2)
            fet_arr[spk] = org / norm
        weight = torch.from_numpy(fet_arr.astype(np.float32)).clone()
        print("new all_n_speakers : {}".format(weight.shape[0]))
        self.model.speaker_loss.modfy_emb(weight.to(self.device))

