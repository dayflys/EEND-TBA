import torch 
import numpy as np
import os

class SaveSpkvLab:
    def __init__(self, dataloader, infer_handler, diarization_loss, all_num_speakers, num_speakers, out_dir):
        self.dataloader = dataloader
        self.infer_handler = infer_handler
        self.diarization_loss = diarization_loss
        self.org_data_all_n_speakers = all_num_speakers
        self.num_speakers = num_speakers
        self.out_dir = out_dir
        
    def run(self):
        #Inference and saving filtered data (spkvec_lab.npz)
        with torch.no_grad():
            all_outputs = []
            all_labels = []

            # Exclude samples that exceed args.num_speakers speakers in a chunk
            for batch_data in self.dataloader:
                # batch_data: (xs, ts, ss, ns, ilens)
                for chunk_data in list(zip(*batch_data)):
                    # chunk_data: (x, t, s, n, ilen)
                    Y_chunked = torch.from_numpy(chunk_data[0]).to('cuda')
                    t_chunked = torch.from_numpy(chunk_data[1]).to('cuda')
                    
                    # outputs: [zs, nmz_wavg_spk0vecs, nmz_wavg_spk1vecs, ..., sigmas]
                    outputs = self.infer_handler.batch_estimate_with_perm(
                            torch.unsqueeze(Y_chunked, 0), # 
                            torch.unsqueeze(t_chunked, 0), # t_chunked: [T, n_speaker]
                            self.diarization_loss)
                    sigma = outputs[self.num_speakers+1][0]
                    t_chunked_t = t_chunked.transpose(1, 0)

                    for i in range(self.num_speakers):
                        # Exclude samples corresponding to silent speaker
                        if torch.sum(t_chunked_t[sigma[i]]) > 0:
                            vec = outputs[i+1][0].cpu().detach().numpy() # [D,]
                            lab = chunk_data[2][sigma[i]] # [1,] -> chunk_data[2]: [n_speaker,]
                            all_outputs.append(vec)
                            all_labels.append(lab)

            
            # Generate spkidx_tbl to convert speaker ID
            spkidx_tbl = np.array([-1 for _ in range(self.org_data_all_n_speakers)])
            for i, idx in enumerate(list(set(all_labels))):
                spkidx_tbl[idx] = i
            # In this line, if speaker_tbl[_idx] == -1, the speaker whose
            # original speaker ID is _idx is excluded for training

            print("number of speakers in the original data: {}"
                  .format(self.org_data_all_n_speakers))
            print("number of speakers in the filtered data: {}"
                  .format(len(set(all_labels))))

            emb_npz_path = self.out_dir + '/spkvec_lab'
            if not os.path.exists(self.out_dir):
                os.mkdir(self.out_dir)
                
            np.savez(emb_npz_path,
                     np.array(all_outputs),
                     np.array(all_labels),
                     spkidx_tbl)
            print("Saved {}".format(emb_npz_path + '.npz'))
            
    