
import torch 
import numpy as np 
import os
from functools import partial
from .features import get_input_dim
from .datasets import DiarizationDatasetFromFeat, DiarizationDatasetFromWave


def collate_fn(batch):
    xs, ts, ss, ns, ilens = list(zip(*batch))
    ilens = np.array(ilens)
    xs = np.array([np.pad(x, [(0, np.max(ilens) - len(x)), (0, 0)],'constant', constant_values=(-1,)) for x in xs])
    ts = np.array([np.pad(t, [(0, np.max(ilens) - len(t)), (0, 0)],'constant', constant_values=(+1,)) for t in ts])
    ss = np.array(ss)
    ns = np.array(ns)
    
    return (xs, ts, ss, ns, ilens)
    
    
def collate_fn_torch(batch):
    xs, ts, ss, ns, ilens = collate_fn(batch)

    return (torch.from_numpy(xs), torch.from_numpy(ts), torch.from_numpy(ss), torch.from_numpy(ns), torch.from_numpy(ilens))


def collate_fn_ns(batch, n_speakers, spkidx_tbl):
    xs, ts, ss, ns, ilens = list(zip(*batch))
    valid_chunk_indices1 = [i for i in range(len(ts))
                            if ts[i].shape[1] == n_speakers]
    valid_chunk_indices2 = []

    # n_speakers (rec-data) > n_speakers (model)
    invalid_chunk_indices1 = [i for i in range(len(ts))
                                if ts[i].shape[1] > n_speakers]

    ts = list(ts)
    ss = list(ss)
    for i in invalid_chunk_indices1:
        s = np.sum(ts[i], axis=0)
        cs = ts[i].shape[0]
        if len(s[s > 0.5]) <= n_speakers:
            # n_speakers (chunk-data) <= n_speakers (model)
            # update valid_chunk_indices2
            valid_chunk_indices2.append(i)
            idx_arr = np.where(s > 0.5)[0]
            ts[i] = ts[i][:, idx_arr]
            ss[i] = ss[i][idx_arr]
            if len(s[s > 0.5]) < n_speakers:
                # n_speakers (chunk-data) < n_speakers (model)
                # update ts[i] and ss[i]
                n_speakers_real = len(s[s > 0.5])
                zeros_ts = np.zeros((cs, n_speakers), dtype=np.float32)
                zeros_ts[:, :-(n_speakers-n_speakers_real)] = ts[i]
                ts[i] = zeros_ts
                mones_ss = -1 * np.ones((n_speakers,), dtype=np.int64)
                mones_ss[:-(n_speakers-n_speakers_real)] = ss[i]
                ss[i] = mones_ss
            else:
                # n_speakers (chunk-data) == n_speakers (model)
                pass
        else:
            # n_speakers (chunk-data) > n_speakers (model)
            pass

    # valid_chunk_indices: chunk indices using for training
    valid_chunk_indices = sorted(valid_chunk_indices1 + valid_chunk_indices2)

    ilens = np.array(ilens)
    ilens = ilens[valid_chunk_indices]
    ns = np.array(ns)[valid_chunk_indices]
    ss = np.array([ss[i] for i in range(len(ss))
                    if ts[i].shape[1] == n_speakers])
    xs = [xs[i] for i in range(len(xs)) if ts[i].shape[1] == n_speakers]
    ts = [ts[i] for i in range(len(ts)) if ts[i].shape[1] == n_speakers]
    xs = np.array([np.pad(x, [(0, np.max(ilens) - len(x)), (0, 0)],
                            'constant', constant_values=(-1,)) for x in xs])
    ts = np.array([np.pad(t, [(0, np.max(ilens) - len(t)), (0, 0)],
                            'constant', constant_values=(+1,)) for t in ts])

    if spkidx_tbl is not None:
        # Update global speaker ID
        all_n_speakers = np.max(spkidx_tbl) + 1
        bs = len(ns)
        ns = np.array([
                np.arange(
                    all_n_speakers,
                    dtype=np.int64
                    ).reshape(all_n_speakers, 1)] * bs)
        ss = np.array([spkidx_tbl[ss[i]] for i in range(len(ss))])

    return (xs, ts, ss, ns, ilens)


class DiarizationDataLoader:
    def __init__(self, args):
        self.args = args
        self.input_dim = get_input_dim(self.args.frame_size,
                                      self.args.context_size,
                                      self.args.input_transform)
        
        
    def get_dataset_from_wav(self, data_dir):
        dataset = DiarizationDatasetFromWave(
            data_dir,
            chunk_size=self.args.chunk_size,
            context_size=self.args.context_size,
            input_transform=self.args.input_transform,
            frame_size=self.args.frame_size,
            frame_shift=self.args.frame_shift,
            subsampling=self.args.subsampling,
            rate=self.args.sampling_rate,
            n_speakers=self.args.num_speakers,
            )
        
        return dataset
    
    def get_dataset_from_feat(self, data_dir):
        featlab_chunk_indices_path = f'{data_dir}/{self.args.preprocess_dir}/{self.args.preprocess_trial}'
        dataset = DiarizationDatasetFromFeat(
            featlab_chunk_indices_path,
            self.input_dim,
            )
        
        return dataset 
    
    def get_train_dataloader(self, dataset, dataset_sampler=None):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batchsize,
            shuffle=False,
            pin_memory=True,
            sampler=dataset_sampler,
            num_workers=self.args.num_workers)
        
        return dataloader
    
    def get_val_dataloader(self, dataset, dataset_sampler=None):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batchsize,
            pin_memory=True,
            sampler=dataset_sampler,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn_torch)
        
        return dataloader

    def get_preprocess_dataloader(self, dataset):
        # Count n_chunks
        device = [device_id for device_id in range(torch.cuda.device_count())]
        print('GPU device {} is used'.format(device))
        batchsize = self.args.batchsize * len(device) * self.args.batchsize_per_gpu

        spkidx_tbl = None
        if self.args.initmodel:
            # adaptation
            npz = np.load(self.args.spkv_lab)
            spkidx_tbl = npz['arr_2']

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batchsize,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=partial(collate_fn_ns, n_speakers=self.args.num_speakers, spkidx_tbl=spkidx_tbl))   
        
        return dataloader
            
    def get_save_spkv_dataloader(self, dataset):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=partial(collate_fn_ns, n_speakers=self.args.num_speakers, spkidx_tbl=None))   
        
        return dataloader

    def get_input_dim(self):
        return self.input_dim
    


