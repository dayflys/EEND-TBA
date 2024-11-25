import torch
import torch.nn.functional as F
import torch.nn as nn
from .transformer import TransformerEncoder


class EEND_TBA(nn.Module):
    def __init__(self,
                 num_speakers,
                 input_dim,
                 hidden_size,
                 transformer_encoder_n_heads,
                 transformer_encoder_n_layers,
                 transformer_encoder_dropout,
                 shuffle,
                 speaker_loss=None,
                 diarization_loss=None
                 ):
        super(EEND_VC, self).__init__()
        self.enc = TransformerEncoder(
            input_dim,
            transformer_encoder_n_layers,
            hidden_size,
            n_heads=transformer_encoder_n_heads,
            dropout_rate=transformer_encoder_dropout)
        
        # self.linear = nn.Linear(hidden_size, num_speakers)
        self.linear = nn.Linear(hidden_size, 1)
        # for i in range(num_speakers):
        #     setattr(self, '{}{:d}'.format("linear", i), nn.Linear(hidden_size, spk_vector_dim))
        self.shuffle = shuffle
        self.num_speakers = num_speakers
        self.speaker_loss = speaker_loss
        self.diarization_loss = diarization_loss
        self.prompt = nn.Parameter(torch.randn([1, num_speakers, input_dim]))
        nn.init.xavier_uniform_(self.prompt)
    

    def forward(self, xs, ts=None, ss=None, ns=None, ilens=None):
        # Since xs is pre-padded, the following code is extra,
        # but necessary for reproducibility
        pad_shape = xs.shape #[B, 150, 345]
        xs = torch.cat((self.prompt.repeat(xs.size(0), 1, 1), xs), dim=1)
        
        
        emb_list = self.enc(xs) # emb: (B*(T+3), E)
        
        
        ys_list = []
        
        for idx,emb in enumerate(emb_list):
            if idx == len(emb_list)-1:
                ys, attractors, emb = self.make_before_pit(pad_shape[0],emb)
                
            else:
                ys, _ , _ = self.make_before_pit(pad_shape[0],emb)
            
            ys_list.append(ys)
        # emb = emb.reshape(pad_shape[0], -1, emb.size()[-1]) # emb: (B,(T+3), E)
        
        # attractors = emb[:,:self.num_speakers,:] #attractors : (B, 3, E)
        
        # emb = emb[:,self.num_speakers:,:] #emb: (B, T, E)  
        
        # ys = emb.matmul(attractors.transpose(2,1)) # ys : (B,T,3)
        # ys = self.linear(emb)
 
        spksvecs = []
        for i in range(self.num_speakers):
            spkivecs = emb * torch.unsqueeze(attractors[:, i, :], 1)
            #spkivecs = getattr(self, '{}{:d}'.format("linear", i))(emb)
            spksvecs.append(spkivecs)

        spksvecs = list(zip(*spksvecs)) #spksvecs: (B,3,T,E) -> (B,3,E)
        
        spkcount = self.linear(attractors) #(B,3,1)
        
        if self.diarization_loss and self.speaker_loss:
            return self.cal_loss(ys_list, spksvecs, ts, ss, ns, ilens, spkcount)
        
        return ys, spksvecs, attractors, emb
        

    def cal_loss(self, ys_list, spksvecs, ts, ss, ns, ilens, spkcount):
        total_pit_loss = 0
        
        for idx,ys in enumerate(ys_list):
            if idx == len(ys_list)-1:
                pit_loss, label, sigmas = self.diarization_loss.batch_pit_loss(ys, ts, ilens)
            else:
                pit_loss, _, _ = self.diarization_loss.batch_pit_loss(ys, ts, ilens)

            total_pit_loss += pit_loss / len(ys_list)
        
        ys = ys_list[-1]
        
        n_speakers = [t.shape[1] for t in ts]
        
        speaker_count_loss = self.diarization_loss.speaker_count_pit_loss(spkcount, n_speakers)
        
        ss = [[i.item() for i in s] for s in ss]
        spk_loss = self.speaker_loss(spksvecs, ys, ts, ss, sigmas, ns, ilens)

        return ys, spksvecs, total_pit_loss, spk_loss, speaker_count_loss, label
               
    def make_before_pit(self,pad_shape,emb):
        
        emb = emb.reshape(pad_shape, -1, emb.size()[-1]) # emb: (B,(T+3), E)
        
        attractors = emb[:,:self.num_speakers,:] #attractors : (B, 3, E)
        
        emb = emb[:,self.num_speakers:,:] #emb: (B, T, E)  
        
        ys = emb.matmul(attractors.transpose(2,1))
        
        return ys, attractors, emb