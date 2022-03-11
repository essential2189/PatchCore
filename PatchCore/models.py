from typing import Tuple
from tqdm import tqdm

import torch
from torch import tensor
from torch.utils.data import DataLoader
import timm
import torch.nn as nn

import numpy as np
from sklearn.metrics import roc_auc_score

from utils import GaussianBlur, get_coreset_idx_randomp, get_tqdm_params
import matplotlib.pyplot as plt
# import seaborn as sns
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class PatchCore(torch.nn.Module):
	def __init__(
			self,
			f_coreset: float = None, # fraction the number of training samples
			coreset_eps: float = None, # sparse projection parameter
			backbone_name: str = None,
			out_indices: Tuple = None,
			):

		super().__init__()
		print('patchcore:', backbone_name)
		print('out_indices:', out_indices)

		self.f_coreset = f_coreset
		self.coreset_eps = coreset_eps
		self.average = torch.nn.AvgPool2d(3, stride=1)
		self.blur = GaussianBlur(4)
		self.n_reweight = 3
		self.patch_lib = []
		self.resize = None
		self.train_list = []
		self.test_list = []

		self.feature_extractor = timm.create_model(
			backbone_name,
			out_indices=out_indices,
			features_only=True,
			pretrained=True,
		)

		for param in self.feature_extractor.parameters():
			param.requires_grad = False
		self.feature_extractor.eval()

		self.backbone_name = backbone_name # for results metadata
		self.out_indices = out_indices

		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.feature_extractor = self.feature_extractor.to(self.device)

	def __call__(self, x: tensor):
		with torch.no_grad():
			feature_maps = self.feature_extractor(x.to(self.device))
		feature_maps = [fmap.to("cpu") for fmap in feature_maps]

		return feature_maps

	def fit(self, train_dl):
		for sample, _ in tqdm(train_dl, **get_tqdm_params()):
			feature_maps = self(sample)

			if self.resize is None:
				largest_fmap_size = feature_maps[0].shape[-2:]
				print(largest_fmap_size)
				self.resize = torch.nn.AdaptiveAvgPool2d(largest_fmap_size)
			resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]
			patch = torch.cat(resized_maps, 1)
			patch = patch.reshape(patch.shape[1], -1).T

			self.patch_lib.append(patch)

		self.patch_lib = torch.cat(self.patch_lib, 0)

		if self.f_coreset < 1:
			self.coreset_idx = get_coreset_idx_randomp(
				self.patch_lib,
				n=int(self.f_coreset * self.patch_lib.shape[0]),
				eps=self.coreset_eps,
			)
			self.patch_lib = self.patch_lib[self.coreset_idx]

	def predict(self, sample):
		feature_maps = self(sample)
		resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]
		patch = torch.cat(resized_maps, 1)
		patch = patch.reshape(patch.shape[1], -1).T

		dist = torch.cdist(patch, self.patch_lib)

		min_val, min_idx = torch.min(dist, dim=1)
		s_idx = torch.argmax(min_val)
		s_star = torch.max(min_val)

		# reweighting
		m_test = patch[s_idx].unsqueeze(0) # anomalous patch
		m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0) # closest neighbour
		w_dist = torch.cdist(m_star, self.patch_lib) # find knn to m_star pt.1
		_, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False) # pt.2
		# equation 7 from the paper
		m_star_knn = torch.linalg.norm(m_test-self.patch_lib[nn_idx[0,1:]], dim=1)
		# Softmax normalization trick as in transformers.
		# As the patch vectors grow larger, their norm might differ a lot.
		# exp(norm) can give infinities.
		D = torch.sqrt(torch.tensor(patch.shape[1]))
		w = 1-(torch.exp(s_star/D)/(torch.sum(torch.exp(m_star_knn/D))))
		s = w*s_star

		return s

	def evaluate(self, test_dl: DataLoader) -> Tuple[float, float]:
		"""Calls predict step for each test sample."""
		image_preds = []
		image_labels = []

		for sample, label in tqdm(test_dl, **get_tqdm_params()):
			z_score = self.predict(sample)

			image_preds.append(z_score.numpy())
			image_labels.append(label)

		image_preds = np.stack(image_preds)

		image_rocauc = roc_auc_score(image_labels, image_preds)

		return image_rocauc


	def get_parameters(self):
		return {
			"backbone_name": self.backbone_name,
			"out_indices": self.out_indices,
			"f_coreset": self.f_coreset,
			"n_reweight": self.n_reweight,
		}
