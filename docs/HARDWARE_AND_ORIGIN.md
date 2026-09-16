# Hardware and code origin / 장비 및 코드 출처

## User-supplied acquisition description

The user reports using an Infineon **BGT60TR13C** radar with modified vendor examples and a **221e Muse** IMU with vendor recording software. Data were subsequently merged. The exact Muse recording application/version is not supplied. Do not substitute a program name, sampling rate, internal filter, or firmware version from general product literature.

## Official resources checked on 2026-09-16

- Infineon BGT60TR13C product: https://www.infineon.com/part/BGT60TR13C
- Infineon Radar Development Kit: https://www.infineon.com/design-resources/development-tools/sdk/radar-development-kit
- 221e Muse product page supplied by the user, without tracking parameters: https://www.221e.com/lp-muse-wireless-imu

These pages identify the products and vendor tooling. The Infineon RDK describes Python/C/C++/MATLAB interfaces and examples. The Muse page identifies the wireless IMU product and its software options. Neither page proves which settings or software options were used in this experiment.

## What the supplied notebook actually shows

`configs/radar_notebook_observation.json` is extracted from the saved output of `BGT60TR13C/Untitled.ipynb`, cell 0. The source opens `DeviceFmcw()`, reads the existing/default acquisition sequence, and retrieves one frame. It does **not** contain a complete timestamped CSV trial recorder.

Observed saved output: SDK 3.5.0; BGT60TR13C; start/end 58/63 GHz; 2 MSps; 64 ADC samples; RX mask 7; TX mask 1; nested loop repetitions 64; chirp repetition 0.0005911249900236726 s; outer repetition 0.07726884633302689 s; IF gain 23 dB; high/low-pass cutoffs 80/500 kHz; TX power code 31.

This is **one saved notebook observation**, not verified dataset-wide acquisition metadata. The original output also contained a board UUID and a small ADC printout. These are retained only in the private companion archive; the public notebook copy has no outputs.

## Missing acquisition provenance

The ZIP contains helper code and offline processing scripts, but no identified final implementation that repeatedly records raw trials and writes the timestamp filenames consumed by downstream processing. `HMI_DataCap_raw_v1.py` is an offline image-processing branch, not proof of acquisition. Preserve the actual modified recorder and vendor software configuration when recovered.

## Third-party code

Infineon example helpers contain a redistribution notice. Their original headers are unchanged. SDK binaries and the Muse recording program are not included. Verify their respective vendor terms before distributing additional files. This repository does not assign a new license to vendor software.
