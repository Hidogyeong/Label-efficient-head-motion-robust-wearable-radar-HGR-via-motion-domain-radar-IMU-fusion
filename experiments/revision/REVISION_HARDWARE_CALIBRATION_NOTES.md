# Hardware and calibration notes for the revised manuscript / supplementary material

Use this file as a drafting source for the revised manuscript and Supplementary Information.
Do not claim information that is not actually true for the experiment configuration.

## Radar hardware

- Manufacturer: Infineon Technologies
- Radar IC: BGT60TR13C / BGT60RT13C family naming should be checked and unified in the manuscript. The official Infineon product page uses **BGT60TR13C**.
- Official product page: https://www.infineon.com/part/BGT60TR13C
- Important official product-page facts to cite in the manuscript/supplement:
  - XENSIV 60 GHz radar sensor.
  - Frequency range: 58 GHz to 63.5 GHz.
  - 1 Tx antenna and 3 Rx antennas.
  - Antennas in package (AIP), L-shaped array.
  - FoV (HPBW): 90 degrees.
  - The built-in finite-state machine manages FMCW frequency sweeps, data acquisition, and FIFO storage.

## Radar configuration still required from your actual experiment

Reviewer #2 explicitly asks for these values. Fill them from the radar configuration files or acquisition code:

- Board/module name used in the experiment:
- Operating center frequency:
- Bandwidth used in the experiment:
- Chirp slope:
- ADC samples per chirp:
- Chirps per frame:
- Sampling rate:
- Frame rate:
- Range resolution:
- Doppler resolution:
- Angle estimation method:
- Preprocessing from raw radar to range/Doppler/angle CSV:

## IMU hardware

- Manufacturer: 221e
- Device: Muse miniaturized multi-sensor IMU
- Product code: A3b1v03 / A3b2v03
- Important datasheet facts:
  - Inertial data sampling/logging up to 1600 Hz.
  - BLE/USB streaming up to 200 Hz.
  - MPE Motion Processing Engine estimates orientation from accelerometer and gyroscope inputs.
  - Orientation is provided in quaternion form and referred to the earth-gravity reference frame.
  - The MPE library includes offset-calibration logic and internal Kalman-filter tuning.
  - Sensor raw-data ranges include accelerometer ±4/±8/±16/±32 g and gyroscope ±250/±500/±1000/±2000 dps.

## IMU configuration still required from your actual experiment

- Product variant used: A3b1v03 or A3b2v03?
- IMU logging or streaming mode:
- IMU data rate used in the experiment:
- Quaternion output source: MPE/vendor firmware? other?
- Timestamp unit:
- Quaternion convention:
- Whether offsets were calibrated before each session:

## Calibration and synchronization wording

A safe manuscript statement, based on your current information:

> The radar and IMU were mounted on the head-worn prototype. The IMU was aligned in the forward direction of the head-worn device, while the radar was mechanically tilted downward by approximately 45 degrees so that the hand-motion region was within the radar field of view. The tilt angle was measured using a protractor. This mechanical tilt correction is not a full radar-IMU extrinsic calibration; therefore, raw, tilt-only, absolute-quaternion, and relative-first coordinate candidates were empirically evaluated as preprocessing alternatives.

Do not write that rigorous extrinsic calibration was performed unless you have a reproducible procedure and calibration data.

## Synchronization wording template

Fill the bracketed fields:

> Radar and IMU samples were synchronized offline using [timestamp source]. For each radar frame, the nearest IMU quaternion was selected / linearly interpolated from neighboring IMU samples. The merged CSV stores radar range, Doppler, horizontal angle, vertical angle, and the associated IMU quaternion for each time step.
