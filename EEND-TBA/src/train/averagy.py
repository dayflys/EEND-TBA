import torch 
from collections import OrderedDict
import os 

class ModelAveragy:
    def __init__(self, args):
        self.args = args
        
    def run(self):
        infiles = [os.path.join(self.args.model_save_dir, self.args.model_filename.format(i))
                   for i in range(self.args.start_avg_epoch, self.args.end_avg_epoch)]
        outfile = os.path.join(self.args.model_save_dir, self.args.avg_model_filename)
        self.averagy(infiles, outfile)

    def averagy(self, infiles, outfile):
        omodel = OrderedDict()
        
        for infile in infiles:
            tmpmodel = torch.load(infile, map_location="cuda:0")
            for k, v in tmpmodel.items():
                omodel[k] = omodel.get(k, 0) + v

        for k, v in omodel.items():
            omodel[k] = v / len(infiles)

        torch.save(omodel, outfile)
