import torch
from collections import OrderedDict

def prepare_model_for_eval(model_file_path, model):
    model_parameter_dict = torch.load(model_file_path) #['model']
    device = [device_id for device_id in range(torch.cuda.device_count())]
    model.load_state_dict(fix_state_dict(model_parameter_dict), strict=False)
    model.eval()
    model = model.to("cuda")
    print('GPU device {} is used'.format(device))
    print('Prepared model')

    return model


def fix_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            # remove 'module.' of DataParallel
            k = k[7:]
        if k.startswith('net.'):
            # remove 'net.' of PadertorchModel
            k = k[4:]
        new_state_dict[k] = v

    return new_state_dict   