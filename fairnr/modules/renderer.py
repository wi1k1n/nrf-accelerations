# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from fairnr.modules.module_utils import FCLayer
from fairnr.data.geometry import ray
from fairseq.utils import with_torch_seed
from fairseq.modules import GradMultiply

import logging
logger = logging.getLogger(__name__)

MAX_DEPTH = 10000.0
RENDERER_REGISTRY = {}

def register_renderer(name):
    def register_renderer_cls(cls):
        if name in RENDERER_REGISTRY:
            raise ValueError('Cannot register duplicate module ({})'.format(name))
        RENDERER_REGISTRY[name] = cls
        return cls
    return register_renderer_cls


def get_renderer(name):
    if name not in RENDERER_REGISTRY:
        raise ValueError('Cannot find module {}'.format(name))
    return RENDERER_REGISTRY[name]


@register_renderer('abstract_renderer')
class Renderer(nn.Module):
    """
    Abstract class for ray marching
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, **kwargs):
        raise NotImplementedError
    
    @staticmethod
    def add_args(parser):
        pass


@register_renderer('volume_rendering')
class VolumeRenderer(Renderer):

    def __init__(self, args):
        super().__init__(args) 
        self.chunk_size = 1024 * getattr(args, "chunk_size", 64)
        self.valid_chunk_size = 1024 * getattr(args, "valid_chunk_size", self.chunk_size // 1024)
        self.discrete_reg = getattr(args, "discrete_regularization", False)
        self.raymarching_tolerance = getattr(args, "raymarching_tolerance", 0.0)
        self.trace_normal = getattr(args, "trace_normal", False)

    @staticmethod
    def add_args(parser):
        # ray-marching parameters
        parser.add_argument('--discrete-regularization', action='store_true',
                            help='if set, a zero mean unit variance gaussian will be added to encougrage discreteness')
        parser.add_argument('--discrete-regularization-light', action='store_true',
                            help='same as --discrete-regularization, but for light rays')
        
        # additional arguments
        parser.add_argument('--chunk-size', type=int, metavar='D', 
                            help='set chunks to go through the network (~K forward passes). trade time for memory. ')
        parser.add_argument('--valid-chunk-size', type=int, metavar='D', 
                            help='chunk size used when no training. In default the same as chunk-size.')
        parser.add_argument('--raymarching-tolerance', type=float, default=0)

        parser.add_argument('--trace-normal', action='store_true')

    def forward_once(
        self, input_fn, field_fn, ray_start, ray_dir, samples, encoder_states, 
        early_stop=None, output_types=['sigma', 'texture'], **kwargs):
        """
        chunks: set > 1 if out-of-memory. it can save some memory by time.
        """
        sampled_depth = samples['sampled_point_depth']
        sampled_idx = samples['sampled_point_voxel_idx'].long()
        
        # only compute when the ray hits
        sample_mask = sampled_idx.ne(-1)
        if early_stop is not None:
            sample_mask = sample_mask & (~early_stop.unsqueeze(-1))
        if sample_mask.sum() == 0:  # miss everything skip
            return None, 0

        sampled_xyz = ray(ray_start.unsqueeze(1), ray_dir.unsqueeze(1), sampled_depth.unsqueeze(2))
        sampled_dir = ray_dir.unsqueeze(1).expand(*sampled_depth.size(), ray_dir.size()[-1])
        samples['sampled_point_xyz'] = sampled_xyz
        samples['sampled_point_ray_direction'] = sampled_dir

        outputs = {'sample_mask': sample_mask}

        # apply mask
        samples = {name: s[sample_mask] for name, s in samples.items()}# if s.shape[:2] == sample_mask.shape[:2]}
        # get encoder features as inputs
        field_inputs = input_fn(samples, encoder_states)


        # forward implicit fields
        field_outputs = field_fn(field_inputs, outputs=output_types)
        
        def masked_scatter(mask, x):
            B, K = mask.size()
            if x.dim() == 1:
                return x.new_zeros(B, K).masked_scatter(mask, x)
            return x.new_zeros(B, K, x.size(-1)).masked_scatter(
                mask.unsqueeze(-1).expand(B, K, x.size(-1)), x)
        
        # post processing
        if 'sigma' in field_outputs:
            sigma, sampled_dists = field_outputs['sigma'], field_inputs['dists']
            # logger.info("Sigmas: {0}...{1}; <{2}>".format(sigma.min(), sigma.max(), sigma.mean()))
            noise = 0 if not self.discrete_reg and (not self.training) else torch.zeros_like(sigma).normal_()
            # noise = 0
            free_energy = torch.relu(noise + sigma) * sampled_dists
            free_energy = free_energy * 7.0  # ? [debug]
            # (optional) free_energy = (F.elu(sigma - 3, alpha=1) + 1) * dists
            # (optional) free_energy = torch.abs(sigma) * sampled_dists  ## ??
            outputs['free_energy'] = masked_scatter(sample_mask, free_energy)
            # outputs['free_energy_nf'] = masked_scatter(sample_mask, torch.relu(sigma) * sampled_dists)
        if 'sdf' in field_outputs:
            outputs['sdf'] = masked_scatter(sample_mask, field_outputs['sdf'])

        if 'albedo' in field_inputs:
            outputs['albedo'] = masked_scatter(sample_mask, field_inputs['albedo'].squeeze(-1))
        if 'roughness' in field_inputs:
            outputs['roughness'] = masked_scatter(sample_mask, field_inputs['roughness'].squeeze(-1))
        if 'normal_brdf' in field_inputs:
            outputs['normal_brdf'] = masked_scatter(sample_mask, field_inputs['normal_brdf'].squeeze(-1))
        if 'texture' in field_outputs:
            outputs['texture'] = masked_scatter(sample_mask, field_outputs['texture'])
        if 'normal' in field_outputs:
            outputs['normal'] = masked_scatter(sample_mask, field_outputs['normal'])
        if 'feat_n2' in field_outputs:
            outputs['feat_n2'] = masked_scatter(sample_mask, field_outputs['feat_n2'])
        if 'lightd' in field_inputs:
            outputs['lightd'] = masked_scatter(sample_mask, field_inputs['lightd'].squeeze(-1))
        # if 'light_intensity' in field_outputs:
        #     outputs['light_intensity'] = masked_scatter(sample_mask, field_outputs['light_intensity'].squeeze())
        if 'ray' in field_inputs:
            outputs['ray'] = masked_scatter(sample_mask, field_inputs['ray'])
        if 'light' in field_inputs:
            outputs['light'] = masked_scatter(sample_mask, field_inputs['light'])
        return outputs, sample_mask.sum()

    def forward_chunk(
        self, input_fn, field_fn, ray_start, ray_dir, samples, encoder_states,
        gt_depths=None, output_types=['sigma', 'texture'], global_weights=None, **kwargs):
        if self.trace_normal:
            output_types += ['normal']

        sampled_depth = samples['sampled_point_depth']
        sampled_idx = samples['sampled_point_voxel_idx'].long()
        original_depth = samples.get('original_point_depth', None)

        tolerance = self.raymarching_tolerance
        chunk_size = self.chunk_size if self.training else self.valid_chunk_size
        early_stop = None
        if tolerance > 0:
            tolerance = -math.log(tolerance)
            
        hits = sampled_idx.ne(-1).long()
        outputs = defaultdict(lambda: [])
        size_so_far, start_step = 0, 0
        accumulated_free_energy = 0
        accumulated_evaluations = 0
        for i in range(hits.size(1) + 1):
            if ((i == hits.size(1)) or (size_so_far + hits[:, i].sum() > chunk_size)) and (i > start_step):
                _outputs, _evals = self.forward_once(
                        input_fn, field_fn, 
                        ray_start, ray_dir, 
                        {name: s[:, start_step: i] 
                            for name, s in samples.items()},
                        encoder_states, 
                        early_stop=early_stop,
                        output_types=output_types,
                        hits=hits)
                if _outputs is not None:
                    accumulated_evaluations += _evals

                    if 'free_energy' in _outputs:
                        accumulated_free_energy += _outputs['free_energy'].sum(1)
                        if tolerance > 0:
                            early_stop = accumulated_free_energy > tolerance
                            hits[early_stop] *= 0
                    
                    for key in _outputs:
                        outputs[key] += [_outputs[key]]
                else:
                    for key in outputs:
                        outputs[key] += [outputs[key][-1].new_zeros(
                            outputs[key][-1].size(0),
                            sampled_depth[:, start_step: i].size(1),
                            *outputs[key][-1].size()[2:] 
                        )]
                start_step, size_so_far = i, 0
            
            if (i < hits.size(1)):
                size_so_far += hits[:, i].sum()

        # for k in outputs:
        #     dims = [v.dim() for v in outputs[k]]
        #     if len(set(dims)) != 1:
        #         print(k, dims)

        outputs = {key: torch.cat(outputs[key], 1) for key in outputs}
        results = {}

        # If hits.sum() == 0
        # ~~~ plan the code such that this case never happens! ~~~
        if not len(outputs.items()):
            outputs['sample_mask'] = torch.zeros_like(sampled_depth) > 0
            outputs['ray'] = torch.ones(*sampled_depth.size(), 3, device=sampled_depth.device) * 3 / (3 ** 0.5)
            free_energy = torch.zeros_like(sampled_depth)
            accumulated_evaluations = torch.Tensor([0]).to(sampled_depth.device).sum()
        
        if 'free_energy' in outputs:
            free_energy = outputs['free_energy']
            shifted_free_energy = torch.cat([free_energy.new_zeros(sampled_depth.size(0), 1), free_energy[:, :-1]], dim=-1)  # shift one step
            a = 1 - torch.exp(-free_energy.float())                             # probability of it is not empty here
            b = torch.exp(-torch.cumsum(shifted_free_energy.float(), dim=-1))   # probability of everything is empty up to now
            probs = (a * b).type_as(free_energy)                                # probability of the ray hits something here

            # fenf = outputs['free_energy_nf'] # noise free
            # sfenf = torch.cat([fenf.new_zeros(sampled_depth.size(0), 1), fenf[:, :-1]], dim=-1)
            # anf = 1 - torch.exp(-fenf.float())
            # bnf = torch.exp(-torch.cumsum(sfenf.float(), dim=-1))
            # probsnf = (anf * bnf).type_as(fenf)
        else:
            probs = outputs['sample_mask'].type_as(sampled_depth) / sampled_depth.size(-1)  # assuming a uniform distribution
            # probsnf = probs

        if global_weights is not None:
            probs = probs * global_weights

        # depth = (sampled_depth * probs).sum(-1)
        depth = (sampled_depth * probs).sum(-1)
        missed = 1 - probs.sum(-1)
        
        results.update({
            'probs': probs, 'depths': depth, 
            'max_depths': sampled_depth.masked_fill(hits.eq(0), -1).max(1).values,
            'min_depths': sampled_depth.min(1).values,
            'missed': missed, 'ae': accumulated_evaluations,
            'fe': free_energy   # free energy (sigmas)
        })
        if original_depth is not None:
            results['z'] = (original_depth * probs).sum(-1)


        if 'albedo' in outputs:
            results['albedo'] = (outputs['albedo'] * probs.unsqueeze(-1)).sum(-2)
        if 'normal_brdf' in outputs:
            results['normal_brdf'] = (outputs['normal_brdf'] * probs.unsqueeze(-1)).sum(-2)
        if 'roughness' in outputs:
            results['roughness'] = (outputs['roughness'] * probs).sum(-1)
        if 'light' in outputs:
            results['light'] = (outputs['light'] * probs.unsqueeze(-1)).sum(-2)
        if 'ray' in outputs:
            results['ray'] = (outputs['ray'] * probs.unsqueeze(-1)).sum(-2)


        # simply composite colors
        if 'texture' in outputs:
            compClr = outputs['texture'] * probs.unsqueeze(-1)
        # or evaluate brdf if needed
        elif all([k in results for k in ['albedo', 'normal_brdf', 'roughness', 'light']]):
            compClr = field_fn.brdf(results['light'].unsqueeze(1), results['ray'],
                            results['normal_brdf'], results['albedo'], results['roughness'].unsqueeze(1))
            compClr = compClr.squeeze()
            if field_fn.min_color == 0:
                compClr = torch.sigmoid(compClr)
        # else:
        #     raise Exception('Neither texture nor R is got for compositing')
        # if light_transmittance has to be considered
        if 'light_transmittance' in outputs:
            # carefully divide even though this almost only happens in masked elements
            lightdmask = (outputs['lightd'] <= 1e-6)
            lightd = torch.where(outputs['lightd'] <= 1e-6,
                                 torch.Tensor([1.]).expand_as(outputs['lightd']).to(outputs['lightd'].device),
                                 outputs['lightd'])
            fatt = (outputs['light_radius'] ** 2) / lightd  # torch handles zero division with inf values
            fatt[lightdmask] = 0.0

            # light_intensity = outputs['light_intensity'] if 'light_intensity' in outputs \
            #             else samples['point_light_intensity']
            light_intensity = samples['point_light_intensity']

            Ll = samples['point_light_intensity'] * fatt
            # if self.training:
            #     light_intensity = GradMultiply.apply(light_intensity, 1000.0)
            # Ll = light_intensity * fatt

            # NRF paper, using same sample points as for view ray
            if not len(outputs['light_transmittance']):
                tau = b.type_as(free_energy)
                # tau = torch.exp(-free_energy.float())  # wrong!
            # light_transmittance values are provided by child-class
            else:
                tau = outputs['light_transmittance']
            Li = tau * Ll
            if len(compClr.shape) == 3:
                compClr = compClr * Li.unsqueeze(-1)
            else:
                raise NotImplementedError('This branch of code has not been finished yet')
            # exp(-sum(v)) == prod(exp(-v))
        # final summation of compositing colors
        if 'compClr' in locals():
            results['colors'] = compClr.sum(-2)
        
        if 'normal' in outputs:
            results['normal'] = (outputs['normal'] * probs.unsqueeze(-1)).sum(-2)
            if not self.trace_normal:
                results['eikonal-term'] = (outputs['normal'].norm(p=2, dim=-1) - 1) ** 2
            else:
                results['eikonal-term'] = torch.log((outputs['normal'] ** 2).sum(-1) + 1e-6)
            results['eikonal-term'] = results['eikonal-term'][sampled_idx.ne(-1)]

        if 'feat_n2' in outputs:
            results['feat_n2'] = (outputs['feat_n2'] * probs).sum(-1)
            results['regz-term'] = outputs['feat_n2'][sampled_idx.ne(-1)]
            
        return results

    def forward(self, input_fn, field_fn, ray_start, ray_dir, samples, *args, **kwargs):
        chunk_size = self.chunk_size if self.training else self.valid_chunk_size
        if ray_start.size(0) <= chunk_size:
            results = self.forward_chunk(input_fn, field_fn, ray_start, ray_dir, samples, *args, **kwargs)
        else:
            # the number of rays is larger than maximum forward passes. pre-chuncking..
            results = [
                self.forward_chunk(input_fn, field_fn, 
                    ray_start[i: i+chunk_size], ray_dir[i: i+chunk_size],
                    {name: s[i: i+chunk_size] for name, s in samples.items()}, *args, **kwargs)
                for i in range(0, ray_start.size(0), chunk_size)
            ]
            # results = {name: torch.cat([r[name] for r in results], 0)
            #             if results[0][name].dim() > 0 else sum([r[name] for r in results])
            #         for name in results[0]}
            example_result = next(ri for ri in results if ri is not None)
            results = {name: torch.cat([r[name] for r in results if r is not None], 0)
                        if example_result[name].dim() > 0 else sum([r[name] for r in results if r is not None])
                    for name in example_result}

        if getattr(input_fn, "track_max_probs", False) and (not self.training):
            input_fn.track_voxel_probs(samples['sampled_point_voxel_idx'].long(), results['probs'])
        return results

@register_renderer('light_volume_renderer')
class LightVolumeRenderer(VolumeRenderer):
    def __init__(self, args):
        super().__init__(args)
        self.predict_l = getattr(args, 'predict_l', False)
        self.light_intensity_scale = args.light_intensity
        if self.predict_l:
            self.light_intensity = nn.Parameter(torch.Tensor([1.0]).to(self.args.device_id))
        else: self.light_intensity = torch.Tensor([1.]).to(self.args.device_id)
        self.light_radius = torch.Tensor([args.light_radius]).to(self.args.device_id)

    @staticmethod
    def add_args(parser):
        super(LightVolumeRenderer, LightVolumeRenderer).add_args(parser)

        parser.add_argument('--light-intensity', type=float, default=1.)
        parser.add_argument('--light-radius', type=float, default=0.1)
        parser.add_argument('--predict-l', action='store_true',
                            help='if set, the intensity value L will be predicted by model')


    def forward(self, input_fn, field_fn, ray_start, ray_dir, samples, *args, **kwargs):
        # Light stuff is only needed for texture output
        if 'output_types' in kwargs and not 'texture' in kwargs['output_types']:
            return super().forward(input_fn, field_fn, ray_start, ray_dir, samples, *args, **kwargs)

        viewsN = kwargs['view'].shape[-1]
        pixelsPerView = int(kwargs['hits'].shape[-1] / viewsN)
        voxelsN = samples['sampled_point_voxel_idx'].shape[-1]
        assert kwargs['extrinsics_pl'].shape[0] == 1, 'Multiple shapes are not supported yet'
        # pts.shape: 1 x viewsN x 4 x 1
        # kwargs['extrinsics_pl'].shape: 1 x viewsN x 4 x 4
        pts = torch.Tensor([0, 0, 0, 1]).to(self.args.device_id)[None, :].expand(viewsN, -1)[None, :, :, None]
        plXYZ = torch.matmul(kwargs['extrinsics_pl'], pts)  # 1 x viewsN x 4 x 1

        plXYZExpanded = torch.repeat_interleave(plXYZ[0, :, :, 0], pixelsPerView, 0)[:, None, :3].expand(-1, voxelsN, -1)
        # plCYZExpanded.shape: viewsN * pixelsPerView x voxelsN x 3
        # light_intensity = self.light_intensity
        # if self.predict_l and self.training:
        #     light_intensity = GradMultiply.apply(light_intensity, 1000.0)
        samples.update({'point_light_xyz': plXYZExpanded[None, ...][kwargs['hits']],
                        'point_light_intensity': (self.light_intensity * self.light_intensity_scale).expand(samples['sampled_point_distance'].shape),
                        'point_light_radius': self.light_radius.expand(samples['sampled_point_distance'].shape)
                        })

        results = super().forward(input_fn, field_fn, ray_start, ray_dir, samples, *args, **kwargs)
        return results

    def forward_chunk(self, input_fn, field_fn, ray_start, ray_dir, samples, encoder_states,
            gt_depths=None, output_types=['sigma', 'texture'], global_weights=None, **kwargs):
        return super().forward_chunk(input_fn, field_fn, ray_start, ray_dir, samples, encoder_states,
            gt_depths, output_types, global_weights, **kwargs)

    def forward_once(
        self, input_fn, field_fn, ray_start, ray_dir, samples, encoder_states,
        early_stop=None, output_types=['sigma', 'texture'], **kwargs):
        outputs, _evals = super().forward_once(input_fn, field_fn, ray_start, ray_dir, samples, encoder_states,
                                                early_stop, output_types)
        if outputs is not None:
            if 'texture' in output_types:
                outputs['light_radius'] = self.light_radius.expand(outputs['texture'].size()[:-1])
        return outputs, _evals


@register_renderer('light_iva_volume_renderer')
class LightIVAVolumeRenderer(LightVolumeRenderer):
    def __init__(self, args):
        super().__init__(args)
        self.voxel_sigma = getattr(args, "voxel_sigma", 0.8)

    @staticmethod
    def add_args(parser):
        super(LightIVAVolumeRenderer, LightIVAVolumeRenderer).add_args(parser)
        parser.add_argument('--voxel-sigma', type=float, default=0.8,
                            help='voxel sigma value to be used for voxel approximation')

    def forward_chunk(self, input_fn, field_fn, ray_start, ray_dir, samples, encoder_states,
            gt_depths=None, output_types=['sigma', 'texture'], global_weights=None, **kwargs):
        return super().forward_chunk(input_fn, field_fn, ray_start, ray_dir, samples, encoder_states,
            gt_depths, output_types, global_weights, **kwargs)

    def forward_once(
            self, input_fn, field_fn, ray_start, ray_dir, samples, encoder_states,
            early_stop=None, output_types=['sigma', 'texture'], **kwargs):
        # No need to light overhead if only sigma field requested
        if not 'texture' in output_types:
            return super().forward_once(input_fn, field_fn, ray_start, ray_dir, samples, encoder_states,
                                                   early_stop, output_types)

        outputs, _evals = super().forward_once(input_fn, field_fn, ray_start, ray_dir, samples, encoder_states,
                                               early_stop, output_types)

        ######### <LIGHT_RAYS>
        if outputs is not None:
            # need to intersect light rays with the octree here first
            sampled_xyz = ray(ray_start.unsqueeze(1), ray_dir.unsqueeze(1), samples['sampled_point_depth'].unsqueeze(2))

            point_light_xyz = samples['point_light_xyz']
            # light_dirs = point_light_xyz - sampled_xyz
            # light_start = sampled_xyz
            light_dirs = sampled_xyz - point_light_xyz
            light_start = point_light_xyz
            # [shapes x views x rays x xyz]
            light_start, light_dirs, light_intersection_outputs, light_hits = \
                input_fn.light_ray_intersect(light_start.unsqueeze(0),
                                             light_dirs.unsqueeze(0), encoder_states)

            # [views*rays x sample_points x light_ray_voxel_intersections]
            light_intersection_outputs = {
                name: outs.reshape(*sampled_xyz.size()[:-1], -1) for name, outs in light_intersection_outputs.items()}
            # light_start, light_dirs = light_start.reshape(*sampled_xyz.size()), light_dirs.reshape(*sampled_xyz.size())
            # light_hits = light_hits.reshape(*sampled_xyz.size()[:2])

            light_min_depth = light_intersection_outputs['min_depth']
            light_max_depth = light_intersection_outputs['max_depth']
            # light_voxel_idx = light_intersection_outputs['intersected_voxel_idx']

            transmittances = torch.exp(-torch.sum((light_max_depth - light_min_depth) * self.voxel_sigma, axis=-1))

            # light_mask = light_voxel_idx.ne(-1)
            # transmittances = torch.zeros(sampled_xyz.size()[:2]).to(sampled_xyz.device)
            # for i, ray_hits in enumerate(light_hits):
            #     for j, hit in enumerate(ray_hits):
            #         if not hit: continue
            #         mask = light_mask[0, 0, :]
            #         min_d, max_d = light_min_depth[i, j, mask], \
            #                                 light_max_depth[i, j, mask]
            #         # transmittances[i, j] = torch.sum((max_d - min_d) / voxel_size_norm * self.voxel_sigma)
            #         transmittances[i, j] = torch.exp(-torch.sum((max_d - min_d) * self.voxel_sigma))
            # outputs['light_transmittance'] = transmittances

            ######### </LIGHT_RAYS>


            outputs['light_transmittance'] = transmittances

        return outputs, _evals



@register_renderer('light_nrf_volume_renderer')
class LightNRFVolumeRenderer(LightVolumeRenderer):
    def forward_once(
            self, input_fn, field_fn, ray_start, ray_dir, samples, encoder_states,
            early_stop=None, output_types=['sigma', 'texture'], **kwargs):
        outputs, _evals = super().forward_once(input_fn, field_fn, ray_start, ray_dir, samples, encoder_states,
                                               early_stop, output_types)
        if outputs is not None:
            outputs['light_transmittance'] = torch.Tensor([])
        return outputs, _evals



@register_renderer('light_bruteforce_volume_renderer')
class LightBFVolumeRenderer(LightVolumeRenderer):
    def __init__(self, args):
        super().__init__(args)

    def forward_chunk(self, input_fn, field_fn, ray_start, ray_dir, samples, encoder_states,
            gt_depths=None, output_types=['sigma', 'texture'], global_weights=None, **kwargs):
        return super().forward_chunk(input_fn, field_fn, ray_start, ray_dir, samples, encoder_states,
            gt_depths, output_types, global_weights, **kwargs)

    # intersects light rays with octree
    def intersect(self, light_start, light_dirs, encoder, encoder_states, **kwargs):
        # kwargs['hits'] - hits for view rays
        V, S, _ = light_dirs.size()
        light_start, light_dirs, light_intersection_outputs, light_hits = \
            encoder.light_ray_intersect(light_start.unsqueeze(0),
                                         light_dirs.unsqueeze(0), encoder_states)

        # if self.reader.no_sampling and self.training:  # sample points after ray-voxel intersection
        #     uv, size = kwargs['uv'], kwargs['size']
        #     mask = hits.reshape(*uv.size()[:2], uv.size(-1))
        #
        #     # sample rays based on voxel intersections
        #     sampled_uv, sampled_masks = self.reader.sample_pixels(
        #         uv, size, mask=mask, return_mask=True)
        #     sampled_masks = sampled_masks.reshape(uv.size(0), -1).bool()
        #     hits, sampled_masks = hits[sampled_masks].reshape(S, -1), sampled_masks.unsqueeze(-1)
        #     intersection_outputs = {name: outs[sampled_masks.expand_as(outs)].reshape(S, -1, outs.size(-1))
        #                             for name, outs in intersection_outputs.items()}
        #     ray_start = ray_start[sampled_masks.expand_as(ray_start)].reshape(S, -1, 3)
        #     ray_dir = ray_dir[sampled_masks.expand_as(ray_dir)].reshape(S, -1, 3)
        #
        # else:
        #     sampled_uv = None

        min_depth = light_intersection_outputs['min_depth'].reshape(V, S, -1)
        max_depth = light_intersection_outputs['max_depth'].reshape(V, S, -1)
        pts_idx = light_intersection_outputs['intersected_voxel_idx'].reshape(V, S, -1)
        dists = (max_depth - min_depth)  #.masked_fill(pts_idx.eq(-1), 1.0)
        light_intersection_outputs['probs'] = (dists / dists.sum(dim=-1, keepdim=True))\
                                                .masked_fill(pts_idx.eq(-1), 0.0)\
                                                .reshape(*light_intersection_outputs['min_depth'].size())
        dists = dists.masked_fill(pts_idx.eq(-1), 0.0)
        light_intersection_outputs['steps'] = (dists.sum(-1) / encoder.step_size) \
                                                .reshape(*light_intersection_outputs['min_depth'].size()[:-1])
        # if getattr(self.args, "fixed_num_samples", 0) > 0:
        #     intersection_outputs['steps'] = intersection_outputs['min_depth'].new_ones(
        #         *intersection_outputs['min_depth'].size()[:-1], 1) * self.args.fixed_num_samples
        # else:
        #     intersection_outputs['steps'] = dists.sum(-1) / self.encoder.step_size
        return light_start, light_dirs, light_intersection_outputs, light_hits#, sampled_uv



    def forward_once(
            self, input_fn, field_fn, ray_start, ray_dir, samples, encoder_states,
            early_stop=None, output_types=['sigma', 'texture'], **kwargs):
        ######### <LIGHT_RAYS>
        if not 'texture' in output_types:
            return super().forward_once(input_fn, field_fn, ray_start, ray_dir, samples, encoder_states,
                                                   early_stop, output_types)

        # need to intersect light rays with the octree here first
        sampled_xyz = ray(ray_start.unsqueeze(1), ray_dir.unsqueeze(1), samples['sampled_point_depth'].unsqueeze(2))
        point_light_xyz = samples['point_light_xyz']
        # light_dirs = point_light_xyz - sampled_xyz
        # light_start = sampled_xyz
        light_dirs = sampled_xyz - point_light_xyz
        light_start = point_light_xyz

        # [shapes x views x rays x xyz]
        light_start, light_dirs, light_intersection_outputs, light_hits = \
            self.intersect(light_start, light_dirs, input_fn, encoder_states, **kwargs)

        light_start, light_dirs = light_start.reshape(-1, light_start.size(-1)),\
                                  light_dirs.reshape(-1, light_dirs.size(-1))

        # If light ray hits voxels
        if light_hits.sum() > 0:
            # TODO: better squeeze intersections using hits mask, not implemented yet
            # # Squeezing light outputs to hits only
            # lintersection_outputs_squeezed = {
            #     name: outs[light_hits] for name, outs in light_intersection_outputs.items()}
            # # kwargs.update({'light_hits': light_hits})
            # lstart_squeezed, ldirs_squeezed = light_start.unsqueeze(0)[light_hits], light_dirs.unsqueeze(0)[light_hits]

            # [views*rays*view_samples x lray_voxel_intersections]
            light_intersection_outputs = {
                name: outs.reshape(np.prod(sampled_xyz.size()[:-1]), -1) for name, outs in light_intersection_outputs.items()}
            light_intersection_outputs['steps'] = light_intersection_outputs['steps'].squeeze()

            # sample points and use middle point approximation
            # TODO: different GPUs might sample the same way! have to use 'with with_torch_seed(self.unique_seed):'
            light_samples = input_fn.ray_sample(light_intersection_outputs)
            # light_samples_squeezed = input_fn.ray_sample(lintersection_outputs_squeezed)

            # # [views*rays x view_samples x light_samples]
            # light_samples = {
            #     name: outs.reshape(*sampled_xyz.size()[:-1], -1) for name, outs in light_samples.items()}
            # # [views*rays*view_samples x light_samples]
            # light_samples = {
            #     name: outs.reshape(np.prod(sampled_xyz.size()[:-1]), -1) for name, outs in light_samples.items()}

            # Evaluate model on light
            lresults = self.forward(input_fn, field_fn, light_start, light_dirs, light_samples,
                                    encoder_states, early_stop=None, output_types=['sigma'], **kwargs)
            # lresults = self.forward(input_fn, field_fn, lstart_squeezed, ldirs_squeezed, light_samples_squeezed,
            #                         encoder_states, early_stop=None, output_types=['sigma'], **kwargs)


            # # Unsqueeze results
            # light_intersection_outputs = {
            #     name: outs[light_hits] for name, outs in lintersection_outputs_squeezed.items()}

            lfe = lresults['fe']
            # shifted_lfe = torch.cat([lfe.new_zeros(lfe.size(0), 1), lfe[:, :-1]], dim=-1)  # shift one step
            # tau_j = torch.exp(-torch.cumsum(shifted_lfe.float(), dim=-1))
            # tau = torch.prod(tau_j, dim=-1).reshape(*sampled_xyz.size()[:-1])

            tau = torch.exp(-torch.sum(lfe, dim=-1)).reshape(*sampled_xyz.size()[:-1])
            # tau = torch.exp(-torch.sum(lfe * light_intersection_outputs['steps'].unsqueeze(-1), dim=-1))\
            #     .reshape(*sampled_xyz.size()[:-1])
        else:
            tau = torch.ones(*sampled_xyz.size()[:-1], device=sampled_xyz.device)
        ######### </LIGHT_RAYS>

        outputs, _evals = super().forward_once(input_fn, field_fn, ray_start, ray_dir, samples, encoder_states,
                                               early_stop, output_types)

        outputs['light_transmittance'] = tau

        return outputs, _evals



@register_renderer('surface_volume_rendering')
class SurfaceVolumeRenderer(VolumeRenderer):

    def forward_chunk(
        self, input_fn, field_fn, ray_start, ray_dir, samples, encoder_states,
        gt_depths=None, output_types=['sigma', 'texture'], global_weights=None,
        ):
        results = super().forward_chunk(
            input_fn, field_fn, ray_start, ray_dir, samples, encoder_states,
            output_types=['sigma', 'normal'])
        
        # render at the "intersection"
        n_probs = results['probs'].clamp(min=1e-6).masked_fill(samples['sampled_point_voxel_idx'].eq(-1), 0)
        n_depth = (samples['sampled_point_depth'] * n_probs).sum(-1, keepdim=True) / n_probs.sum(-1, keepdim=True).clamp(min=1e-6)
        n_bound = samples['sampled_point_depth'] + samples['sampled_point_distance'] / 2
        n_vidxs = ((n_depth - n_bound) >= 0).sum(-1, keepdim=True)
        n_vidxs = samples['sampled_point_voxel_idx'].gather(1, n_vidxs)

        new_samples = {
            'sampled_point_depth': n_depth,
            'sampled_point_distance': torch.ones_like(n_depth) * 1e-3,  # dummy distance. not useful.
            'sampled_point_voxel_idx': n_vidxs,
        }
        new_results, _ = self.forward_once(input_fn, field_fn, ray_start, ray_dir, new_samples, encoder_states)
        results['colors'] = new_results['texture'].squeeze(1) * (1 - results['missed'][:, None])
        results['normal'] = new_results['normal'].squeeze(1)
        results['eikonal-term'] = torch.cat([results['eikonal-term'], (results['normal'].norm(p=2, dim=-1) - 1) ** 2], 0)
        return results