import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import numpy as np
import torchvision.models as models

class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SAM_fusion(nn.Module):
  def __init__(self, dim_pv, kernel_size=3, padding=1, reduction=4):
      super(SAM_fusion, self).__init__()
      self.pv_local_conv_1 = nn.Sequential(
          nn.Conv2d(4, dim_pv // reduction, 3, padding=1, bias=False),
          nn.BatchNorm2d(dim_pv // reduction)
      )
      self.pv_local_conv_2 = nn.Sequential(
          nn.Conv2d(4, dim_pv//reduction, 5, padding=2, bias=False),
          nn.BatchNorm2d(dim_pv//reduction)
      )
      self.pv_global_conv = nn.Sequential(
          nn.Conv2d(4, dim_pv // reduction, 7, padding=3, bias=False),
          nn.BatchNorm2d(dim_pv // reduction)
      )
      self.pv_gap = _AsppPooling(4, dim_pv//reduction, nn.BatchNorm2d, norm_kwargs=None)
      self.pv_fuse = nn.Sequential(
          nn.Conv2d(4*dim_pv//reduction, dim_pv, kernel_size, padding=padding, bias=False),
          nn.BatchNorm2d(dim_pv),
          nn.Sigmoid()
      )

      self.art_local_conv_1 = nn.Sequential(
          nn.Conv2d(4, dim_pv // reduction, 3, padding=1, bias=False),
          nn.BatchNorm2d(dim_pv // reduction)
      )
      self.art_local_conv_2 = nn.Sequential(
          nn.Conv2d(4, dim_pv // reduction, 5, padding=2, bias=False),
          nn.BatchNorm2d(dim_pv // reduction)
      )
      self.art_global_conv = nn.Sequential(
          nn.Conv2d(4, dim_pv // reduction, 7, padding=3, bias=False),
          nn.BatchNorm2d(dim_pv // reduction)
      )
      self.art_gap = _AsppPooling(4, dim_pv // reduction, nn.BatchNorm2d, norm_kwargs=None)
      self.art_fuse = nn.Sequential(
          nn.Conv2d(4*dim_pv//reduction, dim_pv, kernel_size, padding=padding, bias=False),
          nn.BatchNorm2d(dim_pv),
          nn.Sigmoid()
      )
      self.softmax = nn.Softmax(dim=1)

  def forward(self, pv, art):
      pv_avg = torch.mean(pv, dim=1, keepdim=True)
      pv_max, _ = torch.max(pv, dim=1, keepdim=True)
      art_avg = torch.mean(art, dim=1, keepdim=True)
      art_max, _ = torch.max(art, dim=1, keepdim=True)
      feature_concat = torch.cat((pv_avg, pv_max, art_avg, art_max), dim=1)

      pv_weight = torch.cat((self.pv_local_conv_1(feature_concat), self.pv_local_conv_2(feature_concat),
                             self.pv_global_conv(feature_concat), self.pv_gap(feature_concat)), dim=1)
      pv_weight = self.pv_fuse(pv_weight).unsqueeze(1)

      art_weight = torch.cat((self.art_local_conv_1(feature_concat), self.art_local_conv_2(feature_concat),
                              self.art_global_conv(feature_concat), self.art_gap(feature_concat)), dim=1)
      art_weight = self.art_fuse(art_weight).unsqueeze(1)

      weights = self.softmax(torch.cat((pv_weight, art_weight), dim=1))
      pv_weight, art_weight = weights[:, 0:1, :, :, :].squeeze(1), weights[:, 1:2, :, :, :].squeeze(1)

      aggregated_feature = pv.mul(pv_weight)+art.mul(art_weight)
      modulated_pv = (pv+aggregated_feature)/2
      modulated_art = (art+aggregated_feature)/2

      return modulated_pv, modulated_art, aggregated_feature


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnext50_32x4d(pretrained=True)
        self.block_1 = nn.Sequential(*list(self.base.children())[:4])
        self.block_2 = nn.Sequential(*list(self.base.children())[4:5])
        self.block_3 = nn.Sequential(*list(self.base.children())[5:6])
        self.block_4 = nn.Sequential(*list(self.base.children())[6:7])
        self.block_5 = nn.Sequential(*list(self.base.children())[7:8])

        self.SAM_1 = SAM_fusion(64 * 4)
        self.SAM_2 = SAM_fusion(128 * 4)
        self.SAM_3 = SAM_fusion(256 * 4)
        self.SAM_4 = SAM_fusion(512 * 4)

    def forward(self, x_pv, x_art):
        pv_scale_1 = self.block_1(x_pv)
        art_scale_1 = self.block_1(x_art)

        pv_scale_2 = self.block_2(pv_scale_1)
        art_scale_2 = self.block_2(art_scale_1)
        pv_scale_2, art_scale_2, aggr_scale_2 = self.SAM_1(pv_scale_2, art_scale_2)

        pv_scale_3 = self.block_3(pv_scale_2)
        art_scale_3 = self.block_3(art_scale_2)
        pv_scale_3, art_scale_3, aggr_scale_3 = self.SAM_2(pv_scale_3, art_scale_3)

        pv_scale_4 = self.block_4(pv_scale_3)
        art_scale_4 = self.block_4(art_scale_3)
        pv_scale_4, art_scale_4, aggr_scale_4 = self.SAM_3(pv_scale_4, art_scale_4)

        pv_scale_5 = self.block_5(pv_scale_4)
        art_scale_5 = self.block_5(art_scale_4)
        pv_scale_5, art_scale_5, aggr_scale_5 = self.SAM_4(pv_scale_5, art_scale_5)

        return aggr_scale_2, aggr_scale_3,\
               aggr_scale_4, aggr_scale_5


class URIM(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.lc_conv_1 = nn.Conv2d(in_channels=258, out_channels=128, kernel_size=3, padding=1)
        self.bn_relu_1 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.lc_conv_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn_relu_2 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        self.lc_conv_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn_relu_3 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.lc_conv_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn_relu_4 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1, padding=0)

        # self.fuse = _ConvBNReLU(4, 2, 1, padding=0)
        self.map_update_conv = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, initial_maps, seg_features):
        score = self.softmax(initial_maps)
        score_top, _ = score.topk(k=2, dim=1)
        uncertainty = score_top[:, 0] / (score_top[:, 1] + 1e-8)  # bs, h, w
        uncertainty = torch.exp(1 - uncertainty).unsqueeze(1)  # bs, 1, h, w
        confidence_map = 1-uncertainty

        seg_features = torch.cat([score, seg_features], dim=1)
        r = self.lc_conv_1(seg_features * confidence_map) / (F.avg_pool2d(confidence_map, 3, 1, padding=1) * 9)
        r = self.bn_relu_1(r)

        confidence_map = self.map_update_conv(confidence_map)
        r = self.lc_conv_2(r * confidence_map) / (F.avg_pool2d(confidence_map, 3, 1, padding=1) * 9)
        r = self.bn_relu_2(r)

        confidence_map = self.map_update_conv(confidence_map)
        r = self.lc_conv_3(r * confidence_map) / (F.avg_pool2d(confidence_map, 3, 1, padding=1) * 9)
        r = self.bn_relu_3(r)

        confidence_map = self.map_update_conv(confidence_map)
        r = self.lc_conv_4(r * confidence_map) / (F.avg_pool2d(confidence_map, 3, 1, padding=1) * 9)
        r = self.bn_relu_4(r)

        # r = torch.cat([r, seg_features], dim=1)
        r = self.classifier(r)
        return r, initial_maps

class Seg_head(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.aspp = _ASPP(512*4, [6, 12, 18], norm_layer=nn.BatchNorm2d, norm_kwargs=None)
        # self.conv_block_1 = _ConvBNReLU(128*block.expansion, 256, 3, padding=1)
        self.conv_5 = _ConvBNReLU(256, 256, 3, padding=1)
        self.conv_4 = _ConvBNReLU(256*4, 256, 3, padding=1)
        self.conv_3 = _ConvBNReLU(128 * 4, 256, 3, padding=1)
        self.conv_2 = _ConvBNReLU(64 * 4, 256, 3, padding=1)

        self.conv_block = nn.Sequential(
            _ConvBNReLU(256*4, 256, 3, padding=1),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Sequential(
            _ConvBNReLU(256, 256, 3, padding=1),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, aggr_5, aggr_4, aggr_3, aggr_2):
        size = aggr_2.size()[2:]
        aggr_5 = self.aspp(aggr_5)
        aggr_5 = F.interpolate(aggr_5, size, mode='bilinear', align_corners=True)
        aggr_5 = self.conv_5(aggr_5)

        aggr_4 = F.interpolate(aggr_4, size, mode='bilinear', align_corners=True)
        aggr_4 = self.conv_4(aggr_4)

        aggr_3 = F.interpolate(aggr_3, size, mode='bilinear', align_corners=True)
        aggr_3 = self.conv_3(aggr_3)

        aggr_2 = F.interpolate(aggr_2, size, mode='bilinear', align_corners=True)
        aggr_2 = self.conv_2(aggr_2)
        features = self.conv_block(torch.cat([aggr_5, aggr_4, aggr_3, aggr_2], dim=1))
        maps = self.classifier(features)
        return features, maps

class Network(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.encoder = Encoder()
        self.head = Seg_head(num_classes)
        self.urim = URIM()
        # self.map_conv_1 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=5, padding=2)
    def forward(self, x_pv, x_art):
        size = x_pv.size()[2:]
        aggr_2, aggr_3, aggr_4, aggr_5 = self.encoder(x_pv, x_art)

        features, maps = self.head(aggr_5, aggr_4, aggr_3, aggr_2)
        x, initial_seg = self.urim(maps, features)
        final_seg = F.interpolate(x, size, mode='bilinear', align_corners=True)
        initial_seg = F.interpolate(initial_seg, size, mode='bilinear', align_corners=True)
        return final_seg, initial_seg

class _ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate, norm_layer, norm_kwargs):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs, **kwargs):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out

class _ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer, norm_kwargs, **kwargs):
        super(_ASPP, self).__init__()
        out_channels = 256
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer, norm_kwargs)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer, norm_kwargs)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer, norm_kwargs)
        self.b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x