# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved

import os
import random
import h5py
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance
import torch
from ..data import features 
from ..data.kaldi import KaldiData
from tqdm import tqdm
from scipy.signal import medfilt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class InferHandler:
    def __init__(self, args, model=None):
        self.args = args
        self.model = model
        self.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    def run(self):
        if not os.path.exists(self.args.infer_dir):
            os.mkdir(self.args.infer_dir)
        if not os.path.exists(self.args.out_dir):
                os.mkdir(self.args.out_dir)
        infer_file_list = []
        kaldi_obj = KaldiData(self.args.data_dir)
        for recid in kaldi_obj.wavs:
            print("recid : {}".format(recid))
            data, _ = kaldi_obj.load_wav(recid)
            with torch.no_grad():
                Y_chunked_list = self.preprocess(data)
                acti, svec = self.inference(Y_chunked_list)
                if self.args.est_nspk == 0: # if == 0, use the oracle number of speakers
                    filtered_segments = kaldi_obj.segments[recid]
                    cls_num = len(np.unique([kaldi_obj.utt2spk[seg['utt']] for seg in filtered_segments]).tolist())
                    outdata = self.postprocess(acti, svec, cls_num, None)
                else:
                    outdata = self.postprocess(acti, svec, None, self.args.ahc_dis_th)

            # Saving the resuts
            
            outfname = recid + '.h5'
            outpath = os.path.join(self.args.out_dir, outfname)
            infer_file_list.append(outpath)
            
            with h5py.File(outpath, 'w') as wf:
                # 'T_hat': key
                wf.create_dataset('T_hat', data=outdata)
        self.scoring()

    def scoring(self):
        # https://github.com/foundintranslation/Kaldi/blob/master/tools/sctk-2.4.0/src/md-eval/md-eval.pl
        instruction_scoring = './md-eval.pl -c {} -r {} -s {} > {} 2>/dev/null || exit'
        ground_truth_rttm = f'{self.args.data_dir}/rttm'
        if not os.path.exists(self.args.score_dir):
            os.mkdir(self.args.score_dir)
        
        for threshold in map(float, self.args.threshold.split(',')):
            for median in map(int, self.args.median.split(',')):
                data_out = []
                kaldi_obj = KaldiData(self.args.data_dir)
                for recid in kaldi_obj.wavs:
                    file = os.path.join(self.args.out_dir, recid + '.h5')
                    session, _ = os.path.splitext(os.path.basename(file))
                    data = h5py.File(file, 'r')['T_hat'][:]
                    out = self.make_rttm(data, session, threshold, median)
                    data_out.append(out)

                # save rttm file
                out_rttm_path = f'{self.args.score_dir}/hyp_{threshold}_{median}.rttm'
                with open(out_rttm_path, 'w') as f:
                    f.write('\n'.join(data_out) + '\n')

                # scoring DER using kaldi method
                ins_out_path = f'{self.args.score_dir}/result_th{threshold}_med{median}_collar{self.args.collar}'
                ins = instruction_scoring.format(self.args.collar, ground_truth_rttm, out_rttm_path, ins_out_path) 
                os.system(ins)

        ins_score = f'grep OVER {self.args.score_dir}/result_th0.[^_]*_med[^_]*_collar{self.args.collar} | grep -v nooverlap | sort -nrk 7'
        out = os.popen(ins_score).read()
        with open(f'{self.args.score_dir}/0.scores', 'a') as f:
            f.write(out)
        ins_best_score = f'{ins_score} | tail -n 1'
        out = os.popen(ins_best_score).read()
        print('=============')
        print('BEST SCORE')
        print('IN : ', out.split(':')[0].split('/')[-1])
        print('DER : ', out.split('= ')[-1].split(' ')[0])
        print('=============')
    
    def speakercount_scoreing(self):
        if not os.path.exists(self.args.infer_dir):
            os.mkdir(self.args.infer_dir)
        if not os.path.exists(self.args.out_dir):
                os.mkdir(self.args.out_dir)
        kaldi_obj = KaldiData(self.args.data_dir)
        distMat_lst = []
        oracle_cls_num_lst = []
        for recid in kaldi_obj.wavs:
            data, _ = kaldi_obj.load_wav(recid)
            with torch.no_grad():
                Y_chunked_list = self.preprocess(data)
                acti, svec = self.inference(Y_chunked_list)
                filtered_segments = kaldi_obj.segments[recid]
                oracle_cls_num = len(np.unique([kaldi_obj.utt2spk[seg['utt']] for seg in filtered_segments]).tolist())
                oracle_cls_num_lst.append(oracle_cls_num)

                cl_lst, sil_lst = self.get_cl_sil(acti, cls_num=None) # acti = [n_chunks, T, n_speaker]

                org_svec_len = len(svec)
                svec = np.delete(svec, sil_lst, 0)

                # update cl_lst idx
                _tbl = [i - sum(sil < i for sil in sil_lst) for i in range(org_svec_len)] # [0,1,1,2,3,3,4,5,5,..]
                cl_lst = [(_tbl[_cl[0]], _tbl[_cl[1]]) for _cl in cl_lst]

                distMat = distance.cdist(svec, svec, metric='euclidean')
                for cl in cl_lst:
                    distMat[cl[0], cl[1]] = self.args.clink_dis
                    distMat[cl[1], cl[0]] = self.args.clink_dis

                distMat_lst.append(distMat)
        spk_num = self.args.data_dir.split('_')[-1]
        self.cls_score(distMat_lst,oracle_cls_num_lst,len(kaldi_obj.wavs),spk_num)
        
        
    def cls_score(self,distMat_lst,oracle_cls_num_lst,data_len,spk_num):
        
        with open(f'{self.args.out_dir}/cls_{spk_num}.txt','w') as f:
            best_score = 0.
            best_correct = 0
            best_th = 0
            best_estimated_cls_dict = {}
            best_oracle_cls_dict = {}
            for ahc_dis_th in '0.9,1.0,1.1,1.2,1.3,1.4,1.5'.split(","):
                correct = 0
                estimated_cls_dict = {}
                oracle_cls_dict = {}
                
                for distMat, oracle_cls_num in zip(distMat_lst, oracle_cls_num_lst):
                    clusterer = AgglomerativeClustering(
                            n_clusters=None,
                            affinity='precomputed',
                            linkage='average',
                            distance_threshold=float(ahc_dis_th)) #self.args.ahc_dis_th
                    clusterer.fit(distMat)

                    estimated_cls_num = len(np.unique(clusterer.labels_))
                    print("estimated n_clusters by constraind AHC: {}".format(estimated_cls_num))
                    print("oracle n_clusters: {}".format(oracle_cls_num))
                    if oracle_cls_num == estimated_cls_num:
                        correct += 1
                    
                    try: estimated_cls_dict[estimated_cls_num] += 1
                    except: estimated_cls_dict[estimated_cls_num] = 1
                    
                    try: oracle_cls_dict[oracle_cls_num] += 1
                    except: oracle_cls_dict[oracle_cls_num] = 1
                    
                acc = correct / data_len * 100
                if best_score < acc:
                    best_score = acc
                    best_correct = correct
                    best_th = ahc_dis_th
                    best_estimated_cls_dict = estimated_cls_dict
                    best_oracle_cls_dict = oracle_cls_dict
                print(f"\nahc_dis_th: {ahc_dis_th}, Acc(%) : {acc}")
            f.write(f'# spk_num : {spk_num} \n')
            f.write(f"# Data : {data_len} \n")
            f.write(f"# Correct : {best_correct} \n")
            f.write(f"Best threshold : {best_th} \n")
            f.write(f"Best score(%) : {best_score} \n")
            f.write(f'oracle : {best_oracle_cls_dict} \n')
            f.write(f'estimated : {best_estimated_cls_dict} \n')

    
    def preprocess(self, data):
        Y = features.stft(data, self.args.frame_size, self.args.frame_shift)
        Y = features.transform(Y, transform_type=self.args.input_transform)
        Y = features.splice(Y, context_size=self.args.context_size)
        Y = Y[::self.args.subsampling]

        Y_chunked_list = []
        for start, end in self._gen_chunk_indices(len(Y), self.args.chunk_size):
            if start > 0 and start + self.args.chunk_size > end:
                Y_chunked = torch.from_numpy(Y[end-self.args.chunk_size:end]).to('cuda') # Ensure last chunk size
            else:
                Y_chunked = torch.from_numpy(Y[start:end]).to('cuda')
            Y_chunked_list.append((Y_chunked, start, end))

        return Y_chunked_list
    

    def inference(self, Y_chunked_list):
        acti_list = []
        svec_list = []
        for (Y_chunked, start, end) in Y_chunked_list:
            # outputs: [([T, n_speaker]), ([D,],), ..., ([D,],)]
            outputs = self.batch_estimate(torch.unsqueeze(Y_chunked, 0))
            ys = outputs[0]

            for i in range(self.args.num_speakers):
                spkivecs = outputs[i+1]
                svec_list.append(spkivecs[0].cpu().detach().numpy())

            if start > 0 and start + self.args.chunk_size > end:
                # Ensure last chunk size
                ys = list(ys)
                ys[0] = ys[0][self.args.chunk_size-(end-start):self.args.chunk_size]

            acti = ys[0].cpu().detach().numpy()
            acti_list.append(acti)

        acti = np.array(acti_list) # [n_chunks, T, n_speaker]
        svec = np.array(svec_list) # [n_chunks * n_speakers, D]

        return acti, svec



    def postprocess(self, acti, svec, cls_num = None, ahc_dis_th = 1.0):
        # Get cannot-link index list and silence index list
        n_chunks = len(acti)
        cl_lst, sil_lst = self.get_cl_sil(acti, cls_num) # acti = [n_chunks, T, n_speaker]

        n_samples = n_chunks * self.args.num_speakers - len(sil_lst)
        min_n_samples = 2
        if cls_num is not None:
            min_n_samples = cls_num

        if n_samples >= min_n_samples:
            # clustering (if cls_num is None, update cls_num)
            clslab, cls_num =\
                    self.clustering(svec, cls_num, ahc_dis_th, cl_lst, sil_lst) # svec = [n_chunks * n_speakers, D]
            # merge
            acti, clslab = self.merge_acti_clslab(acti, clslab, cls_num) # clslab = [n_chunks, n_speakers]
            # stitching
            out_chunks = self.stitching(acti, clslab, cls_num)
        else:
            out_chunks = acti

        out = np.vstack(out_chunks)

        return out
    
    

    def make_rttm(self, data, session="AUDIO-ID", threshold=0.5, median=11):
        fmt_list = []
        a = np.where(data > threshold, 1, 0)
        if median > 1:
            a = medfilt(a, (median, 1))
        for spkid, frames in enumerate(a.T):
            frames = np.pad(frames, (1, 1), 'constant')
            changes, = np.where(np.diff(frames, axis=0) != 0)
            fmt = "SPEAKER {:s} 1 {:7.2f} {:7.2f} <NA> <NA> {:s} <NA>"
            t = self.args.frame_shift * self.args.subsampling / self.args.sampling_rate
            for s, e in zip(changes[::2], changes[1::2]):
                fmt_list.append(fmt.format(session, s * t, (e - s) * t, session + "_" + str(spkid)))
        
        return '\n'.join(fmt_list)


    def handle(self, data):
        Y_chunked_list = self.preprocess(data)
        acti, svec = self.inference(Y_chunked_list)
        out = self.postprocess(acti, svec)
        rttm_str = self.make_rttm(out)
    
        return out


    def _gen_chunk_indices(self, data_len, chunk_size): 
        step = chunk_size
        start = 0
        while start < data_len:
            end = min(data_len, start + chunk_size)
            yield start, end
            start += step

    def batch_estimate(self, xs):
        ys, spksvecs = self.model(xs)
        outputs = [
                self.estimate(spksvec, y)
                for (spksvec, y) in zip(spksvecs, ys)]
        outputs = list(zip(*outputs))

        return outputs

    def batch_estimate_with_perm(self, xs, ts, diarization_loss, ilens=None):
        ys, spksvecs = self.model(xs)
        # ys = out[0]
        if ts[0].shape[1] > ys[0].shape[1]:
            # e.g. the case of training 3-spk model with 4-spk data
            add_dim = ts[0].shape[1] - ys[0].shape[1]
            y_device = ys[0].device
            zeros = [torch.zeros(ts[0].shape).to(y_device) for i in range(len(ts))]
            _ys = []
            for zero, y in zip(zeros, ys):
                _zero = zero
                _zero[:, :-add_dim] = y
                _ys.append(_zero)
            _, _, sigmas = diarization_loss.batch_pit_loss(_ys, ts, ilens)
        else:
            _, _, sigmas = diarization_loss.batch_pit_loss(ys, ts, ilens)
        # spksvecs = out[1]

        outputs = [self.estimate(spksvec, y)
                   for (spksvec, y) in zip(spksvecs, ys)]
        outputs = list(zip(*outputs))
        zs = outputs[0]

        if ts[0].shape[1] > ys[0].shape[1]:
            # e.g. the case of training 3-spk model with 4-spk data
            add_dim = ts[0].shape[1] - ys[0].shape[1]
            z_device = zs[0].device
            zeros = [torch.zeros(ts[0].shape).to(z_device)
                     for i in range(len(ts))]
            _zs = []
            for zero, z in zip(zeros, zs):
                _zero = zero
                _zero[:, :-add_dim] = z
                _zs.append(_zero)
            zs = _zs
            outputs[0] = zs
        outputs.append(sigmas)

        # outputs: [zs, nmz_wavg_spk0vecs, nmz_wavg_spk1vecs, ..., sigmas]
        return outputs

    
    def estimate(self, spksvec, y):
        outputs = []
        z = torch.sigmoid(y.transpose(1, 0))

        outputs.append(z.transpose(1, 0))
        for spkid, spkvec in enumerate(spksvec):
            norm_spkvec_inv = 1.0 / torch.norm(spkvec, dim=1)
            # Normalize speaker vectors before weighted average
            spkvec = torch.mul(
                    spkvec.transpose(1, 0), norm_spkvec_inv
                    ).transpose(1, 0)
            wavg_spkvec = torch.mul(
                    spkvec.transpose(1, 0), z[spkid]
                    ).transpose(1, 0)
            sum_wavg_spkvec = torch.sum(wavg_spkvec, dim=0)
            nmz_wavg_spkvec = sum_wavg_spkvec / torch.norm(sum_wavg_spkvec)
            outputs.append(nmz_wavg_spkvec)

        # outputs: [z, nmz_wavg_spk0vec, nmz_wavg_spk1vec, ...]
        return outputs


    def get_cl_sil(self, acti, cls_num): #infer
        n_chunks = len(acti)
        mean_acti = np.array([np.mean(acti[i], axis=0)
                             for i in range(n_chunks)]).flatten()
        n = self.args.num_speakers
        sil_spk_th = self.args.sil_spk_th

        cl_lst = []
        sil_lst = []
        for chunk_idx in range(n_chunks):
            if cls_num is not None:
                if self.args.num_speakers > cls_num:
                    mean_acti_bi = np.array([mean_acti[n * chunk_idx + s_loc_idx]
                                            for s_loc_idx in range(n)])
                    min_idx = np.argmin(mean_acti_bi)
                    mean_acti[n * chunk_idx + min_idx] = 0.0

            for s_loc_idx in range(n):
                a = n * chunk_idx + (s_loc_idx + 0) % n
                b = n * chunk_idx + (s_loc_idx + 1) % n
                if mean_acti[a] > sil_spk_th and mean_acti[b] > sil_spk_th:
                    cl_lst.append((a, b))
                else:
                    if mean_acti[a] <= sil_spk_th:
                        sil_lst.append(a)

        return cl_lst, sil_lst


    def clustering(self, svec, cls_num, ahc_dis_th, cl_lst, sil_lst): #infer
        org_svec_len = len(svec)
        svec = np.delete(svec, sil_lst, 0)


        # update cl_lst idx
        _tbl = [i - sum(sil < i for sil in sil_lst) for i in range(org_svec_len)] # [0,1,1,2,3,3,4,5,5,..]
        cl_lst = [(_tbl[_cl[0]], _tbl[_cl[1]]) for _cl in cl_lst]

        distMat = distance.cdist(svec, svec, metric='euclidean')
        for cl in cl_lst:
            distMat[cl[0], cl[1]] = self.args.clink_dis
            distMat[cl[1], cl[0]] = self.args.clink_dis

        clusterer = AgglomerativeClustering(
                n_clusters=cls_num,
                affinity='precomputed',
                linkage='average',
                distance_threshold=ahc_dis_th)
        clusterer.fit(distMat)

        if cls_num is not None:
            print("oracle n_clusters is known")
        else:
            print("oracle n_clusters is unknown")
            print("estimated n_clusters by constraind AHC: {}"
                  .format(len(np.unique(clusterer.labels_))))
            cls_num = len(np.unique(clusterer.labels_))

        # clusterer.labels_: [0,1,0,1,0,1,...]
        sil_lab = cls_num
        insert_sil_lab = [sil_lab for i in range(len(sil_lst))]
        insert_sil_lab_idx = [sil_lst[i] - i for i in range(len(sil_lst))]
        #print("insert_sil_lab : {}".format(insert_sil_lab)): [3, 3, 3, 3, 3, 3, 3, 3, 3]
        #print("insert_sil_lab_idx : {}".format(insert_sil_lab_idx)): [1, 3, 8, 10, 12, 14, 19, 21, 23]
        clslab = np.insert(clusterer.labels_,
                           insert_sil_lab_idx,
                           insert_sil_lab).reshape(-1, self.args.num_speakers)
        #print("clslab : {}".format(clslab))

        return clslab, cls_num


    def merge_act_max(self, act, i, j): #infer
        for k in range(len(act)):
            act[k, i] = max(act[k, i], act[k, j])
            act[k, j] = 0.0
        return act


    def merge_acti_clslab(self, acti, clslab, cls_num): # infer
        sil_lab = cls_num
        for i in range(len(clslab)):
            _lab = clslab[i].reshape(-1, 1)
            distM = distance.cdist(_lab, _lab, metric='euclidean').astype(np.int64)
            for j in range(len(distM)):
                distM[j][:j] = -1
            idx_lst = np.where(np.count_nonzero(distM == 0, axis=1) > 1)
            merge_done = []
            for j in idx_lst[0]:
                for k in (np.where(distM[j] == 0))[0]:
                    if j != k and clslab[i, j] != sil_lab and k not in merge_done:
                        print("merge : (i, j, k) == ({}, {}, {})".format(i, j, k))
                        acti[i] = self.merge_act_max(acti[i], j, k)
                        clslab[i, k] = sil_lab
                        merge_done.append(j)

        return acti, clslab


    def stitching(self, acti, clslab, cls_num): #infer
        n_chunks = len(acti)
        s_loc = self.args.num_speakers
        sil_lab = cls_num
        s_tot = max(cls_num, s_loc-1)

        # Extend the max value of s_loc_idx to s_tot+1
        # acti = [n_chunks, T, n_speaker]
        add_acti = []
        for chunk_idx in range(n_chunks):
            zeros = np.zeros((len(acti[chunk_idx]), s_tot+1))
            if s_tot+1 > s_loc:
                zeros[:, :-(s_tot+1-s_loc)] = acti[chunk_idx]
            else:
                zeros = acti[chunk_idx]
            add_acti.append(zeros)
        acti = np.array(add_acti)

        out_chunks = []
        for chunk_idx in range(n_chunks):
            # Make sloci2lab_dct.
            # key: s_loc_idx
            # value: estimated label by clustering or sil_lab
            cls_set = set()
            for s_loc_idx in range(s_tot+1):
                cls_set.add(s_loc_idx)

            sloci2lab_dct = {}
            for s_loc_idx in range(s_tot+1):
                if s_loc_idx < s_loc:
                    sloci2lab_dct[s_loc_idx] = clslab[chunk_idx][s_loc_idx]
                    if clslab[chunk_idx][s_loc_idx] in cls_set:
                        cls_set.remove(clslab[chunk_idx][s_loc_idx])
                    else:
                        if clslab[chunk_idx][s_loc_idx] != sil_lab:
                            raise ValueError
                else:
                    sloci2lab_dct[s_loc_idx] = list(cls_set)[s_loc_idx-s_loc]

            # Sort by label value
            sloci2lab_lst = sorted(sloci2lab_dct.items(), key=lambda x: x[1])

            # Select sil_lab_idx
            sil_lab_idx = None
            for idx_lab in sloci2lab_lst:
                if idx_lab[1] == sil_lab:
                    sil_lab_idx = idx_lab[0]
                    break
            if sil_lab_idx is None:
                raise ValueError

            # Get swap_idx
            # [idx of label(0), idx of label(1), ..., idx of label(s_tot)]
            swap_idx = [sil_lab_idx for j in range(s_tot+1)]
            for lab in range(s_tot+1):
                for idx_lab in sloci2lab_lst:
                    if lab == idx_lab[1]:
                        swap_idx[lab] = idx_lab[0]

            #print("swap_idx {}".format(swap_idx)): swap_idx [0, 2, 3, 1]
            swap_acti = acti[chunk_idx][:, swap_idx]
            swap_acti = np.delete(swap_acti, sil_lab, 1)
            out_chunks.append(swap_acti)

        return out_chunks

    def visual(self,loader,iter_idx,types):
        self.model.eval()
        if not os.path.exists(f'{self.args.infer_dir}/embeddings'):
            os.mkdir(f'{self.args.infer_dir}/embeddings')
        if not os.path.exists(f'{self.args.infer_dir}/embeddings/{types}'):
            os.mkdir(f'{self.args.infer_dir}/embeddings/{types}')
        if not os.path.exists(f'{self.args.infer_dir}/save_img'):
            os.mkdir(f'{self.args.infer_dir}/save_img')
        if not os.path.exists(f'{self.args.infer_dir}/save_img/{types}'):
            os.mkdir(f'{self.args.infer_dir}/save_img/{types}')
        
        with torch.no_grad():
            with tqdm(total = iter_idx, ncols = 150) as pbar:
                for step, inputs in enumerate(loader):
                    xs = inputs[0].to(self.device)
                    ts = inputs[1].to(self.device)
                    ss = inputs[2].to(self.device)
                    ns = inputs[3].to(self.device)
                    ilens = inputs[4].to(self.device)
                    ilens = [ilen.item() for ilen in ilens]
                    list1 = torch.tensor([[1.0],[2.0],[4.0]]).repeat(xs.size()[0],1,1).to(self.device)
                    ys,spksvecs,prompt,emb = self.model(xs, ts, ss, ns, ilens)
                    
                    # ys : (B,T,3)
                    # emb : (B,T,256)
                    # spksvecs : (B,3,T,E)
                    # ts_ : (B,T,3)
                    # prompt : (B,3,256)
                    
                                        

                    ts_ = ts.squeeze()
                    labels = ts.matmul(list1).squeeze().to('cpu')
                    ys_ = [i.to('cpu') for i in spksvecs[0]]
                    ts_ = ts_.to('cpu')
                    emb_ = emb.squeeze().to('cpu')
                    prompt_ = prompt.squeeze().to('cpu')
                    
                    
                    dict1 = {'ys': ys_, 'ts':ts_,'emb':emb_, 'prompt':prompt_,'label':labels}
                    path = f'{self.args.infer_dir}/embeddings/{types}/{step}.pk'
                    with open(path,'wb') as fw:
                        pickle.dump(dict1, fw)
                    
                    
                    if step == iter_idx:
                        break
        id_list = [126,287] if types == 'sim' else [37,74]
        for pk_idx in id_list:            
            path = f'{self.args.infer_dir}/embeddings/{types}/{pk_idx}.pk'
            embed = []
            empty = []
            key= []
            prompt = []
            label_ = []
            with open(path,'rb') as fr1:
                dict1 = pickle.load(fr1)
                embed.append(np.array(dict1['ys']))
                empty.append(np.array(dict1['emb']))
                key.append(np.array(dict1['ts']))
                prompt.append(np.array(dict1['prompt']))
                label_.append(np.array(dict1['label']))


            pca = PCA(n_components=2, random_state=1)
            # tsne = TSNE(n_components=2, random_state=1)
            emb_cluster = []
            for i in embed[0]:
            
                emb_cluster.append(np.array(pca.fit_transform(np.array(i))))
            
            empty_cluster = np.array(pca.fit_transform(np.array(empty[0])))
            prompt_cluster = np.array(pca.fit_transform(np.array(prompt[0])))
            
            
            # emb_cluster : (3,T,2)
            # key : (T,3)
            # label : (T,)
            
            key=np.array(key[0], dtype='int')
            setting = list(set(label_[0]))
            
            label_=np.array(label_[0], dtype='int')
            
            
            color = ['blue','green','orange','purple','red','purple','purple','purple','purple']
            marker = ['o','v','P']
            labels = ['silence', 'spk 1', 'spk 2','overlap 1,2', 'spk 3','overlap 1,3','overlap 2,3','overlap 1,2,3']
            
            plt.figure(figsize=(10, 10))
        
            for i in range(8):
                label_idx = np.where(label_ == i)
                if label_idx[0].size == 0:
                    pass
                embs = [clust[label_idx[0],:] for clust in emb_cluster]    
                if i == 0:
                    plt.scatter(empty_cluster[label_idx[0],0],empty_cluster[label_idx[0],1], marker=f'{marker[0]}',s=100,facecolors='none', edgecolors=f'{color[0]}')
                    
                elif i == 1:
                    plt.scatter(embs[0][:,0],embs[0][:,1], marker=f'{marker[0]}',s=100, c=f'{color[1]}')
                elif i == 2:
                    plt.scatter(embs[1][:,0],embs[1][:,1], marker=f'{marker[0]}',s=100, c=f'{color[2]}')
                elif i == 3:
                    plt.scatter(embs[0][:,0],embs[0][:,1], marker=f'{marker[0]}',s=100, c=f'{color[1]}')
                    plt.scatter(embs[1][:,0],embs[1][:,1], marker=f'{marker[0]}',s=100, c=f'{color[2]}')
                elif i == 4:
                    plt.scatter(embs[2][:,0],embs[2][:,1], marker=f'{marker[0]}',s=100, c=f'{color[4]}')
                elif i == 5:
                    plt.scatter(embs[0][:,0],embs[0][:,1], marker=f'{marker[0]}',s=100, c=f'{color[1]}')
                    plt.scatter(embs[2][:,0],embs[2][:,1], marker=f'{marker[0]}',s=100, c=f'{color[4]}')
                elif i == 6:
                    plt.scatter(embs[1][:,0],embs[1][:,1], marker=f'{marker[0]}',s=100, c=f'{color[2]}')
                    plt.scatter(embs[2][:,0],embs[2][:,1], marker=f'{marker[0]}',s=100, c=f'{color[4]}')
                elif i == 7:
                    plt.scatter(embs[0][:,0],embs[0][:,1], marker=f'{marker[0]}',s=100, c=f'{color[1]}')
                    plt.scatter(embs[1][:,0],embs[1][:,1], marker=f'{marker[0]}',s=100, c=f'{color[2]}')
                    plt.scatter(embs[2][:,0],embs[2][:,1], marker=f'{marker[0]}',s=100, c=f'{color[4]}')
         

            plt.scatter(prompt_cluster[0,0],prompt_cluster[0,1],c=f'{color[3]}',marker='X',s=200, label='token 1')
            plt.scatter(prompt_cluster[1,0],prompt_cluster[1,1],c=f'{color[3]}',marker='X',s=200, label='token 2')
            plt.scatter(prompt_cluster[2,0],prompt_cluster[2,1],c=f'{color[3]}',marker='X',s=200, label='token 3')
            
            variable = [mpatches.Patch(color=f'{i}', label=f'{j}') for (i,j) in zip(color,labels)]
            variable = [variable[int(i)] for i in setting if i in [0.0,1.0,2.0,4.0]]
            variable += [mpatches.Patch(color=f'{color[3]}',label='tokens')]
            plt.legend(handles=variable,fontsize = 12,loc='upper right')
            
            plt.savefig(f'{self.args.infer_dir}/save_img/{types}/{pk_idx}_batch_img.png')
