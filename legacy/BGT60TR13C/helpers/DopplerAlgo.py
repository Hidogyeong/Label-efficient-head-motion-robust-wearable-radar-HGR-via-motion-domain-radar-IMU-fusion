# ==========================================================================
# Copyright (C) 2022 Infineon Technologies AG
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ===========================================================================

import numpy as np
from scipy import signal

from helpers.fft_spectrum import *


class DopplerAlgo:
    """Compute Range-Doppler map"""

    def __init__(self, num_samples: int, num_chirps_per_frame: int, num_ant: int, mti_alpha: float = 0.8):
        """Create Range-Doppler map object

        Parameters:
            - num_samples:          Number of samples in a single chirp
            - num_chirps_per_frame: Number of chirp repetitions within a measurement frame
            - num_ant:              Number of antennas
            - mti_alpha:            Parameter alpha of Moving Target Indicator
        """
        self.num_chirps_per_frame = num_chirps_per_frame

        # 체르프 샘플(거리)에 대한 Blackman-Harris 윈도우 매트릭스 계산
        self.range_window = signal.blackmanharris(num_samples).reshape(1, num_samples)

        # 체르프 수(속도)에 대한 Blackman-Harris 윈도우 매트릭스 계산
        self.doppler_window = signal.blackmanharris(self.num_chirps_per_frame).reshape(1, self.num_chirps_per_frame)

        # 움직이는 타겟 지시기(MTI) 파라미터 설정
        self.mti_alpha = mti_alpha

        # MTI 필터 초기화
        self.mti_history = np.zeros((self.num_chirps_per_frame, num_samples, num_ant))

    def compute_doppler_map(self, data: np.ndarray, i_ant: int):
        """Compute Range-Doppler map for i-th antennas

        Parameter:
            - data:     Raw-data for one antenna (dimension:
                        num_chirps_per_frame x num_samples)
            - i_ant:    RX antenna index
        """
        # Step 1 - 신호의 평균 제거 (평균 제거)
        data = data - np.average(data)

        # Step 2 - MTI 처리로 정적 객체 제거
        data_mti = data - self.mti_history[:, :, i_ant]
        self.mti_history[:, :, i_ant] = data * self.mti_alpha + self.mti_history[:, :, i_ant] * (1 - self.mti_alpha)

        # Step 3 - 프레임에 대한 FFT 스펙트럼 계산
        fft1d = fft_spectrum(data_mti, self.range_window)

        # 도플러 FFT 준비

        # 전치 수행
        # 거리 정보가 y 축에 표시됨
        fft1d = np.transpose(fft1d)

        # Step 4 - 도플러에서 데이터 윈도잉
        fft1d = np.multiply(fft1d, self.doppler_window)

        # 제로 패딩 후 FFT 수행
        zp2 = np.pad(fft1d, ((0, 0), (0, self.num_chirps_per_frame)), "constant")
        fft2d = np.fft.fft(zp2) / self.num_chirps_per_frame

        # 속도 0을 중심으로 결과 재배열
        return np.fft.fftshift(fft2d, (1,))
