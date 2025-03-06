import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import time


# model.py
# -------------------------------------------------------------------------------------
# This model is implemented based on the following work:
# Hao Zhou, Sunil Hadap, Kalyan Sunkavalli, David W. Jacobs. "Deep Single-Image Portrait
# Relighting." ICCV, 2019.
# Original code: https://github.com/zhhoper/DPR/blob/master/model/defineHourglass_512_gray_skip.py


def conv3X3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# define the network
class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, batchNorm_type=1, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # batchNorm_type 0 means batchnormalization
        #                1 means instance normalization
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.conv1 = conv3X3(inplanes, outplanes, 1)
        self.conv2 = conv3X3(outplanes, outplanes, 1)
        if batchNorm_type == 0:
            self.bn1 = nn.BatchNorm2d(outplanes)
            self.bn2 = nn.BatchNorm2d(outplanes)
        else:
            self.bn1 = nn.InstanceNorm2d(outplanes)
            self.bn2 = nn.InstanceNorm2d(outplanes)

        self.shortcuts = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.inplanes != self.outplanes:
            out += self.shortcuts(x)
        else:
            out += x

        out = F.relu(out)
        return out


class HourglassBlock(nn.Module):
    '''
        define a basic block for hourglass neetwork
            ^-------------------------upper conv-------------------
            |                                                      |
            |                                                      V
        input------>downsample-->low1-->middle-->low2-->upsample-->+-->output
        NOTE about output:
            Since we need the lighting from the inner most layer,
            let's also output the results from middel layer
    '''

    def __init__(self, inplane, mid_plane, middleNet, skipLayer=True):
        super(HourglassBlock, self).__init__()
        # upper branch
        self.skipLayer = True
        self.upper = BasicBlock(inplane, inplane, batchNorm_type=1)

        # lower branch
        self.downSample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upSample = nn.Upsample(scale_factor=2, mode='nearest')
        self.low1 = BasicBlock(inplane, mid_plane)
        self.middle = middleNet
        self.low2 = BasicBlock(mid_plane, inplane, batchNorm_type=1)

    def forward(self, x, ppg, light, count, skip_count):
        # we use count to indicate wich layer we are in
        # max_count indicates the from which layer, we would use skip connections
        out_upper = self.upper(x)
        out_lower = self.downSample(x)
        out_lower = self.low1(out_lower)
        out_lower, out_feat, out_middle = self.middle(out_lower, ppg, light, count + 1, skip_count)
        out_lower = self.low2(out_lower)
        out_lower = self.upSample(out_lower)

        if count >= skip_count and self.skipLayer:
            # withSkip is true, then we use skip layer
            # easy for analysis
            out = out_lower + out_upper
        else:
            out = out_lower
            # out = out_upper
        return out, out_feat, out_middle


class PPGNet(nn.Module):
    def __init__(self, ncInput, ncOutput, ncMiddle):
        super(PPGNet, self).__init__()
        self.ncInput = ncInput
        self.ncOutput = ncOutput
        self.ncMiddle = ncMiddle

        self.conv1 = nn.Conv2d(self.ncInput, self.ncMiddle, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(self.ncMiddle, self.ncMiddle // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(self.ncMiddle // 2, self.ncMiddle // 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.predict_relu1 = nn.PReLU()
        self.predict_FC = nn.Linear(128, 10)  # 입력 크기를 128로 고정

        # target_ppg가 [B,150]일 경우, 이를 [B, 128, 16, 16]으로 변환하기 위한 FC layer
        self.ppg_fc = nn.Linear(150, 128 * 16 * 16)
        # 변환된 target_ppg의 채널 수(여기서는 128)를 innerFeat의 채널 수(ncInput)로 매핑하는 conv layer
        self.ppg_conv = nn.Conv2d(128, self.ncInput, kernel_size=3, stride=1, padding=1, bias=False)
        self.ppg_relu = nn.ReLU()

    def forward(self, innerFeat, target_ppg, target_light, count, skip_count):
        # innerFeat의 첫 ncInput 채널을 추출 (lighting feature)
        x = innerFeat[:, 0:self.ncInput, :, :]

        # 특징 추출 및 압축
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.predict_relu1(x)

        # 전역 특징을 단일 값으로 변환 (rPPG 예측)
        x_flat = torch.flatten(x, 1)  # 배치 차원을 제외한 모든 차원을 flatten
        rppg = self.predict_FC(x_flat)
        rppg = rppg.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1] 형태로 변환

        if target_ppg is None:
            return innerFeat, innerFeat[:, :self.ncInput, :, :], rppg

        remaining = innerFeat[:, self.ncInput:, :, :].clone()
        new_innerFeat = torch.cat([target_ppg, remaining], dim=1)

        return new_innerFeat, innerFeat[:, :self.ncInput, :, :], rppg


class ReconNet(nn.Module):
    def __init__(self, rppg_dim=10, feat_dim=(27, 16, 16)):  # (C, H, W)
        super(ReconNet, self).__init__()

        self.feat_dim = feat_dim  # 최종 복원할 feature map 크기
        self.fc1 = nn.Linear(rppg_dim, 256)
        self.fc2 = nn.Linear(256, np.prod(feat_dim))  # 128x16x16으로 변환

        self.conv = nn.Sequential(
            nn.Conv2d(feat_dim[0], feat_dim[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(feat_dim[0], feat_dim[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, rppg_tensor):
        batch_size, seq_len, _ = rppg_tensor.shape

        # FC 레이어를 통과하여 feature map 크기로 변환
        x = self.fc1(rppg_tensor)
        x = F.relu(x)
        x = self.fc2(x)

        # Reshape하여 feature map 형태로 변환
        x = x.view(batch_size, seq_len, *self.feat_dim)  # [B, S, C, H, W]

        # Conv 레이어를 통해 feature map을 정제
        x = x.view(batch_size * seq_len, *self.feat_dim)  # [B*S, C, H, W]
        x = self.conv(x)
        x = x.view(batch_size, seq_len, *self.feat_dim)  # 다시 [B, S, C, H, W]로 변환

        return x


class HourglassNet(nn.Module):
    '''
       basic idea: low layers are shared, upper layers are different
                   lighting should be estimated from the inner most layer
        NOTE: we split the bottle neck layer into albedo, normal and lighting
    '''

    def __init__(self, baseFilter=16, gray=True):
        super(HourglassNet, self).__init__()

        self.ncLight = 27  # number of channels for input to lighting network
        self.baseFilter = baseFilter

        # number of channles for output of lighting network
        if gray:
            self.ncOutLight = 9  # gray: channel is 1
        else:
            self.ncOutLight = 27  # color: channel is 3

        self.ncPre = self.baseFilter  # number of channels for pre-convolution

        # number of channels
        self.ncHG3 = self.baseFilter
        self.ncHG2 = 2 * self.baseFilter
        self.ncHG1 = 4 * self.baseFilter
        self.ncHG0 = 8 * self.baseFilter + self.ncLight

        self.pre_conv = nn.Conv2d(3, self.ncPre, kernel_size=5, stride=1, padding=2)
        self.pre_bn = nn.BatchNorm2d(self.ncPre)

        self.light = PPGNet(self.ncLight, self.ncOutLight, 128)
        self.HG0 = HourglassBlock(self.ncHG1, self.ncHG0, self.light)
        self.HG1 = HourglassBlock(self.ncHG2, self.ncHG1, self.HG0)
        self.HG2 = HourglassBlock(self.ncHG3, self.ncHG2, self.HG1)
        self.HG3 = HourglassBlock(self.ncPre, self.ncHG3, self.HG2)

        self.conv_1 = nn.Conv2d(self.ncPre, self.ncPre, kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(self.ncPre)
        self.conv_2 = nn.Conv2d(self.ncPre, self.ncPre, kernel_size=1, stride=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.ncPre)
        self.conv_3 = nn.Conv2d(self.ncPre, self.ncPre, kernel_size=1, stride=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.ncPre)

        self.output = nn.Conv2d(self.ncPre, 3, kernel_size=1, stride=1, padding=0)

        # Self-attention과 FFN 레이어 추가
        self.self_attn = nn.MultiheadAttention(150, num_heads=5)
        self.norm1 = nn.LayerNorm(150)

        # FFN 레이어들 수정
        self.fc_1 = nn.Linear(1500, 150, bias=False)
        self.norm2 = nn.LayerNorm(150)
        self.ffn = nn.Sequential(
            nn.Linear(150, 300),
            nn.ReLU(),
            nn.Linear(300, 150)
        )

        # Reconstruction 레이어는 유지
        self.fc_recon1 = nn.Linear(150, 150)
        self.fc_recon2 = nn.Linear(150, 1500)

        # 각 프레임의 latent vector를 rPPG 값으로 변환하는 FFN
        self.rppg_ffn = nn.Sequential(
            nn.Linear(10, 256),  # 입력 크기를 10으로 수정
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 각 프레임당 하나의 rPPG 값 출력
        )

        self.recon_ffn = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

        self.reconstruction_net = ReconNet()

    def forward(self, x, target_ppg=None, target_light=None, skip_count=1, oriImg=None):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Initialize tensors to store outputs
        out_img_tensor = []
        out_feat_tensor = []
        rppg_tensor = torch.zeros(batch_size, seq_len, 10).to(x.device)

        for i in range(seq_len):
            # (B, S, C, H, W) B: batch size, S: sequence length, C: channel, H: height, W: width
            feat = self.pre_conv(x[:, i])
            feat = F.relu(self.pre_bn(feat))

            # get the inner most features
            feat, out_feat, rppg = self.HG3(feat, target_ppg, target_light, 0, skip_count)
            feat = self.conv_1(feat)
            feat = self.conv_2(feat)
            feat = self.conv_3(feat)
            out_img = self.output(feat)

            # for training, we need the original image
            # to supervise the bottle neck layer feature
            out_feat_ori = None
            if not oriImg is None:
                _, out_feat_ori, _ = self.HG3(oriImg, target_light, 0, skip_count)
            # print(rppg.shape)
            # Store outputs in tensors
            rppg_tensor[:, i, :] = rppg[0, :, 0, 0]
            out_img_tensor.append(out_img)
            out_feat_tensor.append(out_feat)

        # Stack tensors along sequence dimension
        out_img_tensor = torch.stack(out_img_tensor, dim=1)
        out_feat_tensor = torch.stack(out_feat_tensor, dim=1)

        out_rppg_tensor = rppg_tensor.view(batch_size, seq_len, -1)
        # print(out_feat_tensor.shape)

        # 각 프레임별로 FFN 적용
        rppg_values = self.rppg_ffn(out_rppg_tensor)  # [batch_size, seq_len, 1]
        rppg_tensor = rppg_values.squeeze(-1)  # [batch_size, seq_len]

        # reconstruction을 위한 처리
        recon_rppg_tensor = self.reconstruction_net(out_rppg_tensor)

        # rppg 신호 정규화
        rppg_tensor = (rppg_tensor - rppg_tensor.min(dim=-1, keepdim=True)[0]) / (
                    rppg_tensor.max(dim=-1, keepdim=True)[0] - rppg_tensor.min(dim=-1, keepdim=True)[0])

        return rppg_tensor, out_feat_tensor, out_img_tensor, out_feat_tensor, recon_rppg_tensor

    def get_rppg_only(self, x):
        """rPPG 신호만 계산하는 함수"""
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        with torch.no_grad():  # encoder는 여전히 freeze
            rppg_tensor = torch.zeros(batch_size, seq_len, 10).to(x.device)

            for i in range(seq_len):
                feat = self.pre_conv(x[:, i])
                feat = F.relu(self.pre_bn(feat))
                feat, out_feat, rppg = self.HG3(feat, None, 0, 1)

                # decoder 부분
                feat = self.conv_1(feat)
                feat = self.conv_2(feat)
                feat = self.conv_3(feat)

                rppg_tensor[:, i, :] = rppg[0, :, 0, 0]

            # rPPG FFN 적용
            out_rppg_tensor = rppg_tensor.view(batch_size, seq_len, -1)
            rppg_values = self.rppg_ffn(out_rppg_tensor)
            rppg_tensor = rppg_values.squeeze(-1)

        return rppg_tensor


class HourglassNetStep2(nn.Module):
    '''
       basic idea: low layers are shared, upper layers are different
                   lighting should be estimated from the inner most layer
        NOTE: we split the bottle neck layer into albedo, normal and lighting
    '''

    def __init__(self, baseFilter=16, gray=True):
        super(HourglassNetStep2, self).__init__()

        self.ncLight = 27  # number of channels for input to lighting network
        self.baseFilter = baseFilter

        # number of channles for output of lighting network
        if gray:
            self.ncOutLight = 9  # gray: channel is 1
        else:
            self.ncOutLight = 27  # color: channel is 3

        self.ncPre = self.baseFilter  # number of channels for pre-convolution

        # number of channels
        self.ncHG3 = self.baseFilter
        self.ncHG2 = 2 * self.baseFilter
        self.ncHG1 = 4 * self.baseFilter
        self.ncHG0 = 8 * self.baseFilter + self.ncLight

        self.pre_conv = nn.Conv2d(3, self.ncPre, kernel_size=5, stride=1, padding=2)
        self.pre_bn = nn.BatchNorm2d(self.ncPre)

        self.light = PPGNet(self.ncLight, self.ncOutLight, 128)
        self.HG0 = HourglassBlock(self.ncHG1, self.ncHG0, self.light)
        self.HG1 = HourglassBlock(self.ncHG2, self.ncHG1, self.HG0)
        self.HG2 = HourglassBlock(self.ncHG3, self.ncHG2, self.HG1)
        self.HG3 = HourglassBlock(self.ncPre, self.ncHG3, self.HG2)

        self.conv_1 = nn.Conv2d(self.ncPre, self.ncPre, kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(self.ncPre)
        self.conv_2 = nn.Conv2d(self.ncPre, self.ncPre, kernel_size=1, stride=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.ncPre)
        self.conv_3 = nn.Conv2d(self.ncPre, self.ncPre, kernel_size=1, stride=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.ncPre)

        self.output = nn.Conv2d(self.ncPre, 3, kernel_size=1, stride=1, padding=0)

        # Self-attention과 FFN 레이어 추가
        self.self_attn = nn.MultiheadAttention(150, num_heads=5)
        self.norm1 = nn.LayerNorm(150)

        # FFN 레이어들 수정
        self.fc_1 = nn.Linear(1500, 150, bias=False)
        self.norm2 = nn.LayerNorm(150)
        self.ffn = nn.Sequential(
            nn.Linear(150, 300),
            nn.ReLU(),
            nn.Linear(300, 150)
        )

        # Reconstruction 레이어는 유지
        self.fc_recon1 = nn.Linear(150, 150)
        self.fc_recon2 = nn.Linear(150, 1500)

        # 각 프레임의 latent vector를 rPPG 값으로 변환하는 FFN
        self.rppg_ffn = nn.Sequential(
            nn.Linear(10, 256),  # 입력 크기를 10으로 수정
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 각 프레임당 하나의 rPPG 값 출력
        )

        self.recon_ffn = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

        self.reconstruction_net = ReconNet()

    def forward(self, x, target_ppg=None, target_light=None, skip_count=1, oriImg=None):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Initialize tensors to store outputs
        out_img_tensor = []
        out_feat_tensor = []
        rppg_tensor = torch.zeros(batch_size, seq_len, 10).to(x.device)
        rppg_recon_tensor = torch.zeros(batch_size, seq_len, 10).to(x.device)

        if target_ppg is not None:
            for i in range(seq_len):
                rppg_recon_tensor[:, i, :] = self.recon_ffn(target_ppg[:, i])

        recon_input = self.reconstruction_net(rppg_recon_tensor)

        for i in range(seq_len):
            # (B, S, C, H, W) B: batch size, S: sequence length, C: channel, H: height, W: width
            feat = self.pre_conv(x[:, i])
            feat = F.relu(self.pre_bn(feat))

            # get the inner most features
            feat, out_feat, rppg = self.HG3(feat, recon_input[:, i], target_light, 0, skip_count)
            feat = self.conv_1(feat)
            feat = self.conv_2(feat)
            feat = self.conv_3(feat)
            out_img = self.output(feat)

            # for training, we need the original image
            # to supervise the bottle neck layer feature
            out_feat_ori = None
            if not oriImg is None:
                _, out_feat_ori, _ = self.HG3(oriImg, target_light, 0, skip_count)
            # print(rppg.shape)
            # Store outputs in tensors
            rppg_tensor[:, i, :] = rppg[0, :, 0, 0]
            out_img_tensor.append(out_img)
            out_feat_tensor.append(out_feat)

        # Stack tensors along sequence dimension
        out_img_tensor = torch.stack(out_img_tensor, dim=1)
        out_feat_tensor = torch.stack(out_feat_tensor, dim=1)

        out_rppg_tensor = rppg_tensor.view(batch_size, seq_len, -1)

        # 각 프레임별로 FFN 적용
        rppg_values = self.rppg_ffn(out_rppg_tensor)  # [batch_size, seq_len, 1]
        rppg_tensor = rppg_values.squeeze(-1)  # [batch_size, seq_len]

        # reconstruction을 위한 처리
        recon_rppg_tensor = self.reconstruction_net(out_rppg_tensor)

        # rppg 신호 정규화
        rppg_tensor = (rppg_tensor - rppg_tensor.min(dim=-1, keepdim=True)[0]) / (
                    rppg_tensor.max(dim=-1, keepdim=True)[0] - rppg_tensor.min(dim=-1, keepdim=True)[0])

        return rppg_tensor, out_feat_tensor, out_img_tensor

    def get_rppg_only(self, x):
        """rPPG 신호만 계산하는 함수"""
        batch_size, seq_len = x.shape[0], x.shape[1]
        rppg_tensor = torch.zeros(batch_size, seq_len, 10).to(x.device)

        for i in range(seq_len):
            feat = self.pre_conv(x[:, i])
            feat = F.relu(self.pre_bn(feat))

            # get the inner most features without target_ppg
            feat, out_feat, rppg = self.HG3(feat, None, None, 0, 0)
            rppg_tensor[:, i, :] = rppg[0, :, 0, 0]

        out_rppg_tensor = rppg_tensor.view(batch_size, seq_len, -1)
        rppg_values = self.rppg_ffn(out_rppg_tensor)
        rppg_tensor = rppg_values.squeeze(-1)

        # 정규화
        rppg_tensor = (rppg_tensor - rppg_tensor.min(dim=-1, keepdim=True)[0]) / (
                    rppg_tensor.max(dim=-1, keepdim=True)[0] - rppg_tensor.min(dim=-1, keepdim=True)[0])

        return rppg_tensor