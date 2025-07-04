import os
import torch
from torch.nn.functional import dropout


from .ctefnet import CTEFNet
from .loss import get_loss


__all__ = [
    'init_model',
    'get_loss',
           ]

def init_model(cfg):
    model_params = cfg.get('model')
    data_params = cfg.get('data')
    summary_params = cfg.get('summary')
    assert model_params.get('name') in ['CTEFNet'], 'Unknown model name'
    
    if model_params.get('name') == 'CTEFNet':
        model = CTEFNet(
            input_size= [data_params.get('input_region')[1]-data_params.get('input_region')[0], data_params.get('input_region')[3]-data_params.get('input_region')[2]],
            in_channels = len(data_params.get('predictor')),
            dim = model_params.get('dim'),
            head = model_params.get('head'),
            depth = model_params.get('depth'),
            dim_feedforward = model_params.get('dim_feedforward'),
            dropout = model_params.get('dropout'),
            obs_time = data_params.get('obs_time'),
            pred_time = data_params.get('pred_time') if data_params.get('pred_type') == 'series' else 1,
            num_index= len(data_params.get('predictand'))
        )

    if model_params.get('load_pretrain'):
        stage = summary_params.get('stage') - 1
        # model_name = model_params.get('name') + str(data_params.get('obs_time')) + '_pretrain' + str(stage)+'.ckpt'
        model_name = model_params.get('name') + '_'+ model_params.get('mode') + str(stage) + '.ckpt'
        checkpoint = os.path.join(summary_params.get('summary_dir'), model_params.get('name'), ''.join(data_params.get('predictand')), model_name)
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)
        print('Load pretrain model', str(stage))


    return model
