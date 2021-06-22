# Microfacet BRDF implementation
# reimplemented based on https://github.com/google/nerfactor/blob/main/brdf/microfacet/microfacet.py

import numpy as np
import torch

class Microfacet:
	"""As described in:
		Microfacet Models for Refraction through Rough Surfaces [EGSR '07]
	"""
	def __init__(self, default_rough=0.3, lambert_only=False, f0=0.91):
		self.default_rough = default_rough
		self.lambert_only = lambert_only
		self.f0 = f0

	def __call__(self, pts2l, pts2c, normal, albedo=None, rough=None):
		"""All in the world coordinates.
		Too low roughness is OK in the forward pass, but may be numerically
		unstable in the backward pass
		pts2l: NxLx3
		pts2c: Nx3
		normal: Nx3
		albedo: Nx3
		rough: Nx1
		"""
		if albedo is None:
			albedo = torch.ones((pts2c.shape[0], 3), dtype=torch.float32)
		if rough is None:
			rough = self.default_rough * torch.ones((pts2c.shape[0], 1), dtype=torch.float32)
		# Normalize directions and normals
		pts2l = safe_l2_normalize(pts2l, axis=2)
		pts2c = safe_l2_normalize(pts2c, axis=1)
		normal = safe_l2_normalize(normal, axis=1)
		# Glossy
		h = pts2l + pts2c[:, None, :]  # NxLx3
		h = safe_l2_normalize(h, axis=2)
		f = self._get_f(pts2l, h)  # NxL
		alpha = rough ** 2
		d = self._get_d(h, normal, alpha=alpha)  # NxL
		g = self._get_g(pts2c, h, normal, alpha=alpha)  # NxL
		l_dot_n = torch.einsum('ijk,ik->ij', pts2l, normal)
		v_dot_n = torch.einsum('ij,ij->i', pts2c, normal)
		denom = 4 * torch.abs(l_dot_n) * torch.abs(v_dot_n)[:, None]
		microfacet = divide_no_nan(f * g * d, denom)  # NxL
		brdf_glossy = torch.tile(microfacet[:, :, None], (1, 1, 3))  # NxLx3
		# Diffuse
		lambert = albedo / np.pi  # Nx3
		brdf_diffuse = torch.broadcast_to(
			lambert[:, None, :], brdf_glossy.shape)  # NxLx3
		# Mix two shaders
		if self.lambert_only:
			brdf = brdf_diffuse
		else:
			brdf = brdf_glossy + brdf_diffuse  # TODO: energy conservation?
		return brdf # NxLx3

	@staticmethod
	def _get_g(v, m, n, alpha=0.1):
		"""Geometric function (GGX).
		"""
		cos_theta_v = torch.einsum('ij,ij->i', n, v)
		cos_theta = torch.einsum('ijk,ik->ij', m, v)
		denom = cos_theta_v[:, None]
		div = divide_no_nan(cos_theta, denom)
		chi = torch.where(div > 0, 1., 0.)
		cos_theta_v_sq = torch.square(cos_theta_v)
		cos_theta_v_sq = torch.clamp(cos_theta_v_sq, 0., 1.)
		denom = cos_theta_v_sq
		tan_theta_v_sq = divide_no_nan(1 - cos_theta_v_sq, denom)
		tan_theta_v_sq = torch.clamp(tan_theta_v_sq, 0., np.inf)
		denom = 1 + torch.sqrt(1 + alpha ** 2 * tan_theta_v_sq[:, None])
		g = divide_no_nan(chi * 2, denom)
		return g  # (n_pts, n_lights)

	@staticmethod
	def _get_d(m, n, alpha=0.1):
		"""Microfacet distribution (GGX).
		"""
		cos_theta_m = torch.einsum('ijk,ik->ij', m, n)
		chi = torch.where(cos_theta_m > 0, 1., 0.)
		cos_theta_m_sq = torch.square(cos_theta_m)
		denom = cos_theta_m_sq
		tan_theta_m_sq = divide_no_nan(1 - cos_theta_m_sq, denom)
		denom = np.pi * torch.square(cos_theta_m_sq) * torch.square(
			alpha ** 2 + tan_theta_m_sq)
		d = divide_no_nan(alpha ** 2 * chi, denom)
		return d # (n_pts, n_lights)

	def _get_f(self, l, m):
		"""Fresnel (Schlick's approximation).
		"""
		cos_theta = torch.einsum('ijk,ijk->ij', l, m)
		f = self.f0 + (1 - self.f0) * (1 - cos_theta) ** 5
		return f  # (n_pts, n_lights)


def safe_l2_normalize(x, axis=None, eps=1e-6):
	return x / torch.norm(x, p=2, dim=axis, epsilon=eps, keepdim=True) + eps

def divide_no_nan(a, b):
	return torch.nan_to_num(torch.div(a, b), nan=0, posinf=0, neginf=0)