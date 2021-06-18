# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

logger = logging.getLogger(__name__)

import cv2, math, time
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models import (
	register_model,
	register_model_architecture
)
from fairseq.utils import item
from fairnr.data.geometry import compute_normal_map, fill_in
from fairnr.models.nsvf import NSVFModel


@register_model('mlnrf')
class MLNRFModel(NSVFModel):
	READER = 'image_reader'
	ENCODER = 'sparsevoxel_light_encoder'
	FIELD = 'radiance_light_field'
	RAYMARCHER = 'light_volume_renderer'

	@classmethod
	def add_args(cls, parser):
		super().add_args(parser)

	def intersecting(self, ray_start, ray_dir, encoder_states, **kwargs):
		return super().intersecting(ray_start, ray_dir, encoder_states, **kwargs)

	def raymarching(self, ray_start, ray_dir, intersection_outputs, encoder_states, fine=False, **kwargs):
		return super().raymarching(ray_start, ray_dir, intersection_outputs, encoder_states, fine, **kwargs)
		return None

	def prepare_hierarchical_sampling(self, intersection_outputs, samples, all_results):
		return super().prepare_hierarchical_sampling(intersection_outputs, samples, all_results)

	def postprocessing(self, ray_start, ray_dir, all_results, hits, sizes):
		return super().postprocessing(ray_start, ray_dir, all_results, hits, sizes)

	def add_other_logs(self, all_results):
		return super().add_other_logs(all_results)

	def _visualize(self, images, sample, output, state, **kwargs):
		return super()._visualize(images, sample, output, state, **kwargs)

	@torch.no_grad()
	def prune_voxels(self, th=0.5, train_stats=False):
		super().prune_voxels(th, train_stats)

	@torch.no_grad()
	def split_voxels(self):
		super().split_voxels()

	@torch.no_grad()
	def reduce_stepsize(self):
		super().reduce_stepsize()

	def clean_caches(self, reset=False):
		super().clean_caches()


@register_model_architecture("mlnrf", "mlnrf_base")
def base_architecture(args):
	# parameter needs to be changed
	args.voxel_size = getattr(args, "voxel_size", None)
	args.max_hits = getattr(args, "max_hits", 60)
	args.raymarching_stepsize = getattr(args, "raymarching_stepsize", 0.01)
	args.raymarching_stepsize_ratio = getattr(args, "raymarching_stepsize_ratio", 0.0)

	# encoder default parameter
	args.voxel_embed_dim = getattr(args, "voxel_embed_dim", 32)
	args.voxel_path = getattr(args, "voxel_path", None)
	args.initial_boundingbox = getattr(args, "initial_boundingbox", None)

	# field
	args.inputs_to_density = getattr(args, "inputs_to_density", "emb:6:32")
	args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256,ray:4,light:4,lightd:0:1")
	args.feature_embed_dim = getattr(args, "feature_embed_dim", 256)
	args.density_embed_dim = getattr(args, "density_embed_dim", 128)
	args.texture_embed_dim = getattr(args, "texture_embed_dim", 256)

	# API Update: fix the number of layers
	args.feature_layers = getattr(args, "feature_layers", 1)
	args.texture_layers = getattr(args, "texture_layers", 3)

	args.background_stop_gradient = getattr(args, "background_stop_gradient", False)
	args.background_depth = getattr(args, "background_depth", 5.0)

	# raymarcher
	args.discrete_regularization = getattr(args, "discrete_regularization", False)
	args.deterministic_step = getattr(args, "deterministic_step", False)
	args.raymarching_tolerance = getattr(args, "raymarching_tolerance", 0)
	args.use_octree = getattr(args, "use_octree", False)

	# reader
	args.pixel_per_view = getattr(args, "pixel_per_view", 2048)
	args.sampling_on_mask = getattr(args, "sampling_on_mask", 0.0)
	args.sampling_at_center = getattr(args, "sampling_at_center", 1.0)
	args.sampling_on_bbox = getattr(args, "sampling_on_bbox", False)
	args.sampling_patch_size = getattr(args, "sampling_patch_size", 1)
	args.sampling_skipping_size = getattr(args, "sampling_skipping_size", 1)

	# others
	args.chunk_size = getattr(args, "chunk_size", 64)
	args.valid_chunk_size = getattr(args, "valid_chunk_size", 64)