# CHANGELOG


## v0.9.1 (2025-08-12)

### Bug Fixes

- **chromatic_aberration.py**: Remove img parameter when plot_keypoints_displacement is called
  ([`ff2e6b6`](https://github.com/StijnvLierop/DeepfakeDetection/commit/ff2e6b6e17af4cea1ce4932dba97fbc7b270f080))

### Chores

- Set default logging level to INFO
  ([`b870dbc`](https://github.com/StijnvLierop/DeepfakeDetection/commit/b870dbcb7d99675055c409d26207a6ac6fb2d8f7))

### Refactoring

- Replace print statements with logging module calls
  ([`dbd15fe`](https://github.com/StijnvLierop/DeepfakeDetection/commit/dbd15fee976dfb6dd172f7c0559998c46c68a6bd))


## v0.9.0 (2025-08-11)

### Bug Fixes

- Add results to gitignore
  ([`21f34c4`](https://github.com/StijnvLierop/DeepfakeDetection/commit/21f34c487349a059fac9e5bb8e41ff290fa1caf6))

### Chores

- Revert chromatic_aberration.py back to version in main
  ([`efc0f59`](https://github.com/StijnvLierop/DeepfakeDetection/commit/efc0f59ecdc7ef844b3a41bbc385b3581230c441))

### Features

- Add evaluate.py script to automate running evaluation
  ([`27753bd`](https://github.com/StijnvLierop/DeepfakeDetection/commit/27753bdcb60bbdc05eddd21bfa9e5274bb226a0d))

- Add roc_auc to evaluation.py script
  ([`3be5849`](https://github.com/StijnvLierop/DeepfakeDetection/commit/3be584911a7b856685cb383f9b2bc109ef2df8f8))

### Refactoring

- Add model name to metrics output file
  ([`02f6545`](https://github.com/StijnvLierop/DeepfakeDetection/commit/02f6545621a31cae5cb6a87ecf62f01eabd32bab))


## v0.8.0 (2025-08-11)

### Features

- Add chromatic aberration estimation functions
  ([`af1470f`](https://github.com/StijnvLierop/DeepfakeDetection/commit/af1470f646732a1c9aca0371a70cae5c2f8ddb8f))

- Replace chromatic aberration method with faster method
  ([`303060d`](https://github.com/StijnvLierop/DeepfakeDetection/commit/303060d497f1ab11459569160a0fc160547a214c))

### Refactoring

- Changed input and output of noise_residual functions to numpy array
  ([`051b2d8`](https://github.com/StijnvLierop/DeepfakeDetection/commit/051b2d83ec99009a131c7a0186cf63e3e97a22b9))

- Make denoiser a parameter in channel_noise_imbalance_ratio function
  ([`9b06c8c`](https://github.com/StijnvLierop/DeepfakeDetection/commit/9b06c8cc7dad1458fc93b8532a00198d8c4f3006))

### Testing

- Add unit tests for chromatic aberration estimation
  ([`3383dd1`](https://github.com/StijnvLierop/DeepfakeDetection/commit/3383dd1bf12f145c7ae2955013816f1261296548))

- Add unit tests for noise estimation
  ([`264f6ca`](https://github.com/StijnvLierop/DeepfakeDetection/commit/264f6ca07594b56066f4880926a0acda709a71ed))


## v0.7.0 (2025-08-05)

### Documentation

- **README**: Update python version
  ([`1132e8f`](https://github.com/StijnvLierop/DeepfakeDetection/commit/1132e8f4380b5374d50c7eec41fe2b80fc165c14))

### Features

- Add function to compute autocorrelation of an image
  ([`1026dff`](https://github.com/StijnvLierop/DeepfakeDetection/commit/1026dff44987fb385631e02412a49373e5f85b3f))

### Refactoring

- Add function that takes a set of images and a function and return the mean result of the processed
  images
  ([`fcf71c6`](https://github.com/StijnvLierop/DeepfakeDetection/commit/fcf71c6ab472b95411f9e738945700bdbe2d55a6))

- Merge fft functions into a single function that operates on numpy arrays
  ([`56149f2`](https://github.com/StijnvLierop/DeepfakeDetection/commit/56149f284d5e7181fe12a62fce883d09e89175e2))

### Testing

- Add tests for fft function
  ([`9e58414`](https://github.com/StijnvLierop/DeepfakeDetection/commit/9e5841455211a0a0a57f387d65b7d52bb8f4503a))

- Add unit tests for average_over_images function
  ([`f4c3d5e`](https://github.com/StijnvLierop/DeepfakeDetection/commit/f4c3d5e5505c8331ba3f63fd20a177c1939c8fac))


## v0.6.0 (2025-08-04)

### Chores

- Fft function now works with numpy arrays instead of ImageInstances
  ([`07b4522`](https://github.com/StijnvLierop/DeepfakeDetection/commit/07b4522a56ca036cc3c9dbd0959a8f67db9a78e5))

- Set python version to be more flexible (>=3.10)
  ([`e601a82`](https://github.com/StijnvLierop/DeepfakeDetection/commit/e601a820da37cf1cdeae4bfe3002c94d2cd044f5))

### Documentation

- Add documentation for the prnu_from_images function
  ([`0076116`](https://github.com/StijnvLierop/DeepfakeDetection/commit/007611633d2d299a61bf022d5e7928bf4d7e0c7c))

### Features

- Add centercrop function
  ([`ae5d586`](https://github.com/StijnvLierop/DeepfakeDetection/commit/ae5d5865431f48f2555714dbcaddf3c06fbbcf7a))

- Add function that calculates PRNU pattern from dataset of images
  ([`5d6b61d`](https://github.com/StijnvLierop/DeepfakeDetection/commit/5d6b61d59a05f5d5f360371bd1a8c9e74b71e29a))

### Testing

- Add prnu unit tests
  ([`4d7d585`](https://github.com/StijnvLierop/DeepfakeDetection/commit/4d7d585da139bbfb582fdb33944084ae0d67e856))


## v0.5.0 (2025-07-31)

### Chores

- Moved image normalization to a separate utils function
  ([`29436da`](https://github.com/StijnvLierop/DeepfakeDetection/commit/29436da6918b4b3a8935afb15bfcdfefd1ab481a))

### Features

- Add new functionality to compute PRNU pattern using 2nd order FSTV method
  ([`0b56000`](https://github.com/StijnvLierop/DeepfakeDetection/commit/0b56000bac0cdf5f607ec5e4941b75ccb8e134b3))


## v0.4.2 (2025-07-28)


## v0.4.1 (2025-07-28)

### Bug Fixes

- Account for condition where dataset name is none
  ([`43adb55`](https://github.com/StijnvLierop/DeepfakeDetection/commit/43adb55383cc46a5a82af2fdd43fb71b70780695))

- Problem in GenImage dataset where label of nature images was never output, removed label mapping
  since there is only one real label
  ([`e5849fb`](https://github.com/StijnvLierop/DeepfakeDetection/commit/e5849fb2328164bd00c64e12c477d451285e3890))

### Chores

- Add split name in dataset name after splitting
  ([`5628b89`](https://github.com/StijnvLierop/DeepfakeDetection/commit/5628b897a9f3729c40265623116679b149d7eaaa))


## v0.4.0 (2025-07-28)

### Features

- Add function to sample n instances per class and FilteredDataset that samples dataset given a list
  of indices
  ([`3808a69`](https://github.com/StijnvLierop/DeepfakeDetection/commit/3808a69c70d32629ed5806d2b8787c8071b82329))

### Testing

- Add tests for FilteredDataset and sample_n_per_class function
  ([`80d7001`](https://github.com/StijnvLierop/DeepfakeDetection/commit/80d7001c834019b6cf2912dd1cb2dba23993a1e0))


## v0.3.0 (2025-07-28)


## v0.2.0 (2025-07-25)

### Bug Fixes

- Add missing pyproject.toml
  ([`ce92559`](https://github.com/StijnvLierop/DeepfakeDetection/commit/ce92559bc5f49e859ce0a80c5f24de971a433d8e))

### Features

- Add noise residual filter and fft filter
  ([`747f6f4`](https://github.com/StijnvLierop/DeepfakeDetection/commit/747f6f4db7a2e20bfdb20b7974a3b02b0bb374f0))

- Make label and path in instance optional and introduce new file-based instances
  ([`957190a`](https://github.com/StijnvLierop/DeepfakeDetection/commit/957190a659898db83af8261a6faa486ba2dfd8a5))

### Testing

- Adjusted unit tests with new instance changes
  ([`26fe10c`](https://github.com/StijnvLierop/DeepfakeDetection/commit/26fe10c19460fe9de2bccbd8ab4793e57c2d24bc))


## v0.1.0 (2025-07-11)

### Features

- Add split_file parameter to FileImageDataset and FileImageSequenceDataset
  ([`ce2eb05`](https://github.com/StijnvLierop/DeepfakeDetection/commit/ce2eb0533c2c44f6549150f4ab0c0ac0f24cc498))

### Refactoring

- Implemented __len__() method in Dataset class and removed implementations in all child classes
  ([`37dd50e`](https://github.com/StijnvLierop/DeepfakeDetection/commit/37dd50e011622ca3ffb5fab5de6bc63544e56841))

### Testing

- Add unit tests for split_file feature
  ([`2975164`](https://github.com/StijnvLierop/DeepfakeDetection/commit/2975164fe90c0730307d105cf8166691c3a6b5c4))


## v0.0.1 (2025-07-11)

- Initial Release
