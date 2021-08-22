# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Base classes for various  models.

The basic principle of differentiable rendering is two components:
    -- an field or so-called geometric field (GE)
    -- an raymarcher or so-called differentiable ray-marcher (RM)
So it can be composed as a GERM model
"""

import logging
import torch
import torch.nn as nn
import skimage.metrics
import imageio, os
import numpy as np
import copy
from collections import defaultdict

from fairseq.models import BaseFairseqModel
from fairseq.utils import with_torch_seed

from fairnr.modules.encoder import get_encoder
from fairnr.modules.field import get_field
from fairnr.modules.renderer import get_renderer
from fairnr.modules.reader import get_reader
from fairnr.data.geometry import ray, compute_normal_map
from fairnr.data.data_utils import recover_image

from fairnr.data.data_utils import Preprocessor

logger = logging.getLogger(__name__)


class BaseModel(BaseFairseqModel):
    """Base class"""

    ENCODER = 'abstract_encoder'
    FIELD = 'abstract_field'
    RAYMARCHER = 'abstract_renderer'
    READER = 'abstract_reader'

    def __init__(self, args, setups):
        super().__init__()
        self.args = args
        self.hierarchical = getattr(self.args, "hierarchical_sampling", False)
        
        self.reader = setups['reader']
        self.encoder = setups['encoder']
        self.field = setups['field']
        self.raymarcher = setups['raymarcher']

        self.cache = None
        self._num_updates = 0
        if getattr(self.args, "use_fine_model", False):
            self.field_fine = copy.deepcopy(self.field)
        else:
            self.field_fine = None

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        reader = get_reader(cls.READER)(args)
        encoder = get_encoder(cls.ENCODER)(args)
        field = get_field(cls.FIELD)(args)
        raymarcher = get_renderer(cls.RAYMARCHER)(args)
        setups = {
            'reader': reader,
            'encoder': encoder,
            'field': field,
            'raymarcher': raymarcher
        }
        return cls(args, setups)

    @classmethod
    def add_args(cls, parser):
        get_reader(cls.READER).add_args(parser)
        get_renderer(cls.RAYMARCHER).add_args(parser)
        get_encoder(cls.ENCODER).add_args(parser)
        get_field(cls.FIELD).add_args(parser)

        # model-level args
        parser.add_argument('--hierarchical-sampling', action='store_true',
            help='if set, a second ray marching pass will be performed based on the first time probs.')
        parser.add_argument('--use-fine-model', action='store_true', 
            help='if set, we will simultaneously optimize two networks, a coarse field and a fine field.')
    
    def set_num_updates(self, num_updates):
        self._num_updates = num_updates
        super().set_num_updates(num_updates)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        if (self.field_fine is None) and \
            ("field_fine" in [key.split('.')[0] for key in state_dict.keys()]):
            # load checkpoint has fine-field network, copying weights to field network
            for fine_key in [key for key in state_dict.keys() if "field_fine" in key]:
                state_dict[fine_key.replace("field_fine", "field")] = state_dict[fine_key]
                del state_dict[fine_key]


    @property
    def dummy_loss(self):
        return sum([p.sum() for p in self.parameters()]) * 0.0

    def forward(self, ray_split=1, **kwargs):
        with with_torch_seed(self.unique_seed):   # make sure different GPU sample different rays
            ray_start, ray_dir, uv = self.reader(**kwargs)

        # kwargs.update({
        #     'field_fn': self.field.forward,
        #     'input_fn': self.encoder.forward})

        if ray_split == 1:
            results = self._forward(ray_start, ray_dir, **kwargs)
        else:
            total_rays = ray_dir.shape[2]
            chunk_size = total_rays // ray_split
            results = [
                self._forward(
                    ray_start, ray_dir[:, :, i: i+chunk_size], **kwargs)
                for i in range(0, total_rays, chunk_size)
            ]
            results = self.merge_outputs(results)

        results['samples'] = {
            'sampled_uv': results.get('sampled_uv', uv),
            'ray_start': ray_start,
            'ray_dir': ray_dir
        }

        # caching the prediction
        self.cache = {
            w: results[w].detach() 
                if isinstance(w, torch.Tensor) 
                else results[w] 
            for w in results
        }
        return results

    def _forward(self, ray_start, ray_dir, **kwargs):
        S, V, P, _ = ray_dir.size()
        assert S == 1, "we only supports single object for now."

        encoder_states = self.preprocessing(**kwargs)
        ray_start, ray_dir, intersection_outputs, hits, sampled_uv = \
            self.intersecting(ray_start, ray_dir, encoder_states, **kwargs)
        # intersection_outputs.shape: 1 x N_of_views_in_current_batch*PIXEL_PER_VIEW x N_of_intersected_voxels
        # save the original rays
        ray_start0 = ray_start.reshape(-1, 3).clone()
        ray_dir0 = ray_dir.reshape(-1, 3).clone()

        P = ray_dir.size(1) // V
        all_results = defaultdict(lambda: None)

        if hits.sum() > 0:
            intersection_outputs = {
                name: outs[hits] for name, outs in intersection_outputs.items()}
            kwargs.update({'hits': hits})
            ray_start, ray_dir = ray_start[hits], ray_dir[hits]
            encoder_states = {name: s.reshape(-1, s.size(-1)) if s is not None else None
                for name, s in encoder_states.items()}
            
            samples, all_results = self.raymarching(               # ray-marching
                ray_start, ray_dir, intersection_outputs, encoder_states, **kwargs)
            
            if self.hierarchical:   # hierarchical sampling
                intersection_outputs = self.prepare_hierarchical_sampling(
                    intersection_outputs, samples, all_results)
                coarse_results = all_results.copy()
                
                samples, all_results = self.raymarching(
                    ray_start, ray_dir, intersection_outputs, encoder_states, fine=True, **kwargs)
                all_results['coarse'] = coarse_results

        hits = hits.reshape(-1)
        all_results = self.postprocessing(ray_start0, ray_dir0, all_results, hits, (S, V, P))
        if self.hierarchical:
            all_results['coarse'] = self.postprocessing(
                ray_start, ray_dir, all_results['coarse'], hits, (S, V, P))
        
        if sampled_uv is not None:
            all_results['sampled_uv'] = sampled_uv
        
        all_results['other_logs'] = self.add_other_logs(all_results)
        return all_results

    def preprocessing(self, **kwargs):
        raise NotImplementedError
    
    def postprocessing(self, ray_start, ray_dir, all_results, hits, sizes):
        raise NotImplementedError

    def intersecting(self, ray_start, ray_dir, encoder_states):
        raise NotImplementedError
    
    def raymarching(self, ray_start, ray_dir, intersection_outputs, encoder_states, fine=False):
        raise NotImplementedError

    def prepare_hierarchical_sampling(self, intersection_outputs, samples, all_results):
        raise NotImplementedError

    def add_other_logs(self, all_results):
        raise NotImplementedError

    def merge_outputs(self, outputs):
        new_output = {}
        for key in outputs[0]:
            if isinstance(outputs[0][key], torch.Tensor) and outputs[0][key].dim() > 2:
                new_output[key] = torch.cat([o[key] for o in outputs], 2)
            else:
                new_output[key] = outputs[0][key]        
        return new_output

    @torch.no_grad()
    def visualize(self, sample, output=None, shape=0, view=0, pprc=None, **kwargs):
        pprc = pprc if pprc else Preprocessor()

        width = int(sample['size'][shape, view][1].item())
        img_id = '{}_{}'.format(sample['shape'][shape], sample['view'][shape, view])
        
        if output is None:
            assert self.cache is not None, "need to run forward-pass"
            output = self.cache  # make sure to run forward-pass.
        sample.update(output['samples'])

        images = {}
        images = self._visualize(images, sample, output, [img_id, shape, view, width, 'render'])
        images = self._visualize(images, sample, sample, [img_id, shape, view, width, 'target'])
        if 'coarse' in output:  # hierarchical sampling
            images = self._visualize(images, sample, output['coarse'], [img_id, shape, view, width, 'coarse'])

        # undo the preprocessing only for color outputs (render/target colors) using preprocessor used for sampling dataset
        for tag in images.keys():
            if not 'color' in tag:
                continue
            images[tag]['img'] = pprc.preprocessInverse(images[tag]['img'].cpu())
            images[tag]['gamma'] = float(self.args.gamma_correction)
            images[tag]['max_val'] = float(self.args.max_color)
            images[tag]['bg'] = float(self.args.transparent_background)

        # here pass identity preprocessor since preprocessInverse has already been applied
        images = {
            tag: recover_image(width=width, pprc=None, **images[tag])
                for tag in images if images[tag] is not None
        }
        return images
        
    def _visualize(self, images, sample, output, state, **kwargs):
        img_id, shape, view, width, name = state
        if 'colors' in output and output['colors'] is not None:
            images['{}_color/{}:HWC'.format(name, img_id)] ={
                'img': output['colors'][shape, view],
                'min_val': float(self.args.min_color)
            }
        if 'depths' in output and output['depths'] is not None:
            min_depth, max_depth = output['depths'].min(), output['depths'].max()
            if getattr(self.args, "near", None) is not None:
                min_depth = self.args.near
                max_depth = self.args.far
            images['{}_depth/{}:HWC'.format(name, img_id)] = {
                'img': output['depths'][shape, view], 
                'min_val': min_depth, 
                'max_val': max_depth}
            normals = compute_normal_map(
                sample['ray_start'][shape, view].float(),
                sample['ray_dir'][shape, view].float(),
                output['depths'][shape, view].float(),
                sample['extrinsics'][shape, view].float().inverse(), width)
            images['{}_normal/{}:HWC'.format(name, img_id)] = {
                'img': normals, 'min_val': -1, 'max_val': 1}
            
            # generate point clouds from depth
            # images['{}_point/{}'.format(name, img_id)] = {
            #     'img': torch.cat(
            #         [ray(sample['ray_start'][shape, view].float(), 
            #             sample['ray_dir'][shape, view].float(),
            #             output['depths'][shape, view].unsqueeze(-1).float()),
            #          (output['colors'][shape, view] - self.args.min_color) / (1 - self.args.min_color)], 1),   # XYZRGB
            #     'raw': True }
            
        if 'z' in output and output['z'] is not None:
            images['{}_z/{}:HWC'.format(name, img_id)] = {
                'img': output['z'][shape, view], 'min_val': 0, 'max_val': 1}
        if 'normal' in output and output['normal'] is not None:
            images['{}_predn/{}:HWC'.format(name, img_id)] = {
                'img': output['normal'][shape, view], 'min_val': -1, 'max_val': 1}
        return images

    def add_eval_images(self, sample, output, criterion, shape=0, view=0):
        predict, target = output['colors'][shape, view], sample['colors'][shape, view]

        img_id = '{}_{}'.format(sample['shape'][shape], sample['view'][shape, view])
        width = int(sample['size'][shape, view][1])
        if hasattr(criterion, 'flip'):
            with torch.no_grad():
                pred = predict.transpose(1, 0).reshape(3, -1, width).unsqueeze(0)
                ref = target.transpose(1, 0).reshape(3, -1, width).unsqueeze(0)
                diff = criterion.flip(pred, ref).reshape(-1, 1)

        return {'loss_hdrflip/{}:HWC'.format(img_id): recover_image(diff, min_val=0, max_val=1, width=width)}

    def add_eval_scores(self, logging_output, sample, output, criterion, scores=['ssim', 'psnr', 'lpips', 'flip'], outdir=None):
        predicts, targets = output['colors'], sample['colors']
        ssims, psnrs, lpips, rmses, flips = [], [], [], [], []
        
        for s in range(predicts.size(0)):
            for v in range(predicts.size(1)):
                width = int(sample['size'][s, v][1])
                p = recover_image(predicts[s, v], width=width, min_val=float(self.args.min_color),
                                  max_val=float(self.args.max_color), gamma=float(self.args.gamma_correction),
                                  pprc=criterion.task.datasets['valid'].dataset.preprocessor)
                t = recover_image(targets[s, v],  width=width, min_val=float(self.args.min_color),
                                  max_val=float(self.args.max_color), gamma=float(self.args.gamma_correction),
                                  pprc=criterion.task.datasets['valid'].dataset.preprocessor)
                pn, tn = p.numpy(), t.numpy()
                p, t = p.to(predicts.device), t.to(targets.device)

                if 'ssim' in scores:
                    ssims += [skimage.metrics.structural_similarity(pn, tn, multichannel=True, data_range=1)]
                if 'psnr' in scores:
                    psnrs += [skimage.metrics.peak_signal_noise_ratio(pn, tn, data_range=1)]
                if 'lpips' in scores and hasattr(criterion, 'lpips'):
                    with torch.no_grad():
                        lpips += [criterion.lpips(
                            2 * p.unsqueeze(-1).permute(3,2,0,1) - 1,
                            2 * t.unsqueeze(-1).permute(3,2,0,1) - 1).item()]
                if 'depths' in sample:
                    td = sample['depths'][sample['depths'] > 0]
                    pd = output['depths'][sample['depths'] > 0]
                    rmses += [torch.sqrt(((td - pd) ** 2).mean()).item()]
                if 'flip' in scores and hasattr(criterion, 'flip'):
                    with torch.no_grad():
                        pred = criterion.task.datasets['valid'].dataset.preprocessor.preprocessInverse(predicts[s, v])
                        ref = criterion.task.datasets['valid'].dataset.preprocessor.preprocessInverse(targets[s, v])
                        pred = pred.transpose(1, 0).reshape(3, -1, width).unsqueeze(0)
                        ref = ref.transpose(1, 0).reshape(3, -1, width).unsqueeze(0)
                        flips += [float(criterion.flip(pred, ref).mean())]

                if outdir is not None:
                    def imsave(filename, image):
                        imageio.imsave(os.path.join(outdir, filename), (image * 255).astype('uint8'))
                    
                    figname = '-{:03d}_{:03d}.png'.format(sample['id'][s], sample['view'][s, v])
                    imsave('output' + figname, pn)
                    imsave('target' + figname, tn)
                    imsave('normal' + figname, recover_image(compute_normal_map(
                        sample['ray_start'][s, v].float(), sample['ray_dir'][s, v].float(),
                        output['depths'][s, v].float(), sample['extrinsics'][s, v].float().inverse(), width=width),
                        min_val=-1, max_val=1, width=width).numpy())
                    if 'featn2' in output:
                        imsave('featn2' + figname, output['featn2'][s, v].cpu().numpy())
                    if 'voxel' in output:
                        imsave('voxel' + figname, output['voxel'][s, v].cpu().numpy())

        if len(ssims) > 0:
            logging_output['ssim_loss'] = np.mean(ssims)
        if len(psnrs) > 0:
            logging_output['psnr_loss'] = np.mean(psnrs)
        if len(lpips) > 0:
            logging_output['lpips_loss'] = np.mean(lpips)
        if len(rmses) > 0:
            logging_output['rmses_loss'] = np.mean(rmses)
        if len(flips) > 0:
            logging_output['flips_loss'] = np.mean(flips)

    def adjust(self, **kwargs):
        raise NotImplementedError

    @property
    def text(self):
        return "fairnr BaseModel"

    @property
    def unique_seed(self):
        return self._num_updates * 137 + self.args.distributed_rank
