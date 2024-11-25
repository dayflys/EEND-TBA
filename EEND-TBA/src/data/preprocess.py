import os
import numpy as np

class PreProcess:
    def __init__(self,args):
        self.args = args

    def run(self, dataloader, data_dir):
        n_chunks = len(dataloader)

        print("n_chunks : {}".format(n_chunks))
        os.makedirs(f"{data_dir}/{self.args.preprocess_dir}/", exist_ok=True)

        if n_chunks % self.args.feature_nj == 0:
            max_num_per_dir = n_chunks // self.args.feature_nj
        else:
            max_num_per_dir = n_chunks // self.args.feature_nj + 1
        print("max_num_per_dir : {}".format(max_num_per_dir))


        # Save featlab_XXXXXXXX.npy and featlab_chunk_indices.txt
        f = open(f"{data_dir}/{self.args.preprocess_dir}/{self.args.preprocess_trial}", 'w')
        idx = 0
        digit_num = len(str(self.args.feature_nj-1))
        fmt = "{}/{}/{:0={}}/featlab_{:0=8}.npy"
        
        for data in dataloader:
            dir_num = idx // max_num_per_dir
            os.makedirs("{}/{}/{:0={}}/".
                        format(data_dir, self.args.preprocess_dir, dir_num, digit_num),
                        exist_ok=True)
            output_npy_path = fmt.format(data_dir, self.args.preprocess_dir, dir_num, digit_num, idx)
            print(output_npy_path)
            bs = data[0].shape[0]
            cs = data[0].shape[1]
            # data0 (feature)
            data0 = data[0]
            # data1 (reference speech activity)
            data1 = data[1]
            # data2 (reference speaker ID)
            data2 = np.zeros([bs, cs, data[2].shape[1]], dtype=np.float32)
            for j in range(bs):
                data2[j, :, :] = data[2][j, :]
            # data3 (reference number of all speakers)
            data3 = np.ones([bs, cs, 1], dtype=np.float32) * len(data[3][0])
            # data4 (real chunk size)
            data4 = np.zeros([bs, cs, 1], dtype=np.float32)
            for j in range(bs):
                data4[j, :, :] = data[4][j]
                
            save_data = np.concatenate((data0,data1,data2,data3,data4), axis=2)

            np.save(output_npy_path, save_data)
            for j in range(save_data.shape[0]):
                f.write("{} {}\n".format(output_npy_path, j))
            idx += 1
        f.close()

        # Create completion flag
        preprocess_flag = f'{self.args.train_data_dir}/{self.args.preprocess_dir}/{self.args.preprocess_flag}'
        f = open(preprocess_flag, 'w')
        f.write("")
        f.close()
        print('Finished!')
    
    
        
    







