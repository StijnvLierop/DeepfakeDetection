# CHANGELOG


## v0.16.0 (2025-12-07)

### Bug Fixes

- **test**: Add save method to MockInstance class
  ([`1e7c504`](https://github.com/StijnvLierop/DeepfakeDetection/commit/1e7c5048ce4fdcfdbe959dc9508356cff188ec81))

### Refactoring

- Remove dataset saving functionality
  ([`d381be6`](https://github.com/StijnvLierop/DeepfakeDetection/commit/d381be68bd6f70906d5e4872392731b223f326ef))

- Remove dnn-cnn denoising model
  ([`8ba8042`](https://github.com/StijnvLierop/DeepfakeDetection/commit/8ba804266522ef6abf7150100da6ee09650f2b31))

- Remove specific models and configs
  ([`2cfe027`](https://github.com/StijnvLierop/DeepfakeDetection/commit/2cfe0278bdd411347cbb7cf6d750d280ee447779))

- Write_predictions_to_file function now only takes predictions and filepath as parameters
  ([`86dc7dd`](https://github.com/StijnvLierop/DeepfakeDetection/commit/86dc7dda8a2bb32c7a6011c172bf059849e7c1f7))

### Testing

- Remove tests that are not needed anymore
  ([`fd6ea70`](https://github.com/StijnvLierop/DeepfakeDetection/commit/fd6ea704c6766ac9302c0e9a4ca5a1a2816aa757))


## v0.15.1 (2025-12-05)

### Bug Fixes

- Refactor repo_id parameter
  ([`6843488`](https://github.com/StijnvLierop/DeepfakeDetection/commit/68434881eb9a14bef81cd65c6989a76466cc28f5))

### Refactoring

- Huggingfacedataset now also allows initialization with datasets.Dataset object besides only
  dataset id
  ([`539d32e`](https://github.com/StijnvLierop/DeepfakeDetection/commit/539d32eb721ebcc1e4958fff160c2b15b5c8e821))


## v0.15.0 (2025-11-30)

### Bug Fixes

- Cached samples are now loaded with file extension included in path
  ([`145a61e`](https://github.com/StijnvLierop/DeepfakeDetection/commit/145a61ea0d2caf2c982b743247a11a71a56e4691))

### Chores

- Update pdm lockfile after merge conflicts
  ([`b0480aa`](https://github.com/StijnvLierop/DeepfakeDetection/commit/b0480aaa4470d1ef57a91d54554b6eeaff3dca39))


## v0.14.0 (2025-11-30)

### Bug Fixes

- Add fiftyone dependency
  ([`3d03dfa`](https://github.com/StijnvLierop/DeepfakeDetection/commit/3d03dfaad440f1a6e6d3f486aa597e0a19148389))

- Add torch dependency and remove streamlit references
  ([`e1b9f7b`](https://github.com/StijnvLierop/DeepfakeDetection/commit/e1b9f7bbacab34d7ea4c56a869dc3b7ecc81f97f))

### Chores

- Remove unneeded packages
  ([`171a3ff`](https://github.com/StijnvLierop/DeepfakeDetection/commit/171a3ff4349d1624784b45cfd8a36af713c736b2))

### Features

- Add confusion matrix to evaluation
  ([`00ebc18`](https://github.com/StijnvLierop/DeepfakeDetection/commit/00ebc181906e49b04cefdfdd8f9fbd943bf62b1d))

- Add fiftyone conversion function
  ([`d79b3aa`](https://github.com/StijnvLierop/DeepfakeDetection/commit/d79b3aa2c596ee97f9c29365ee409345578125fb))

- Add hugginface dataset class
  ([`760552f`](https://github.com/StijnvLierop/DeepfakeDetection/commit/760552fe2ec1b85c5921cc767ffe18b329e36008))

- Add predictions to fiftyone interface
  ([`4d5cb85`](https://github.com/StijnvLierop/DeepfakeDetection/commit/4d5cb85a2882594bb3152ff3c029c410e99d0c6b))

### Refactoring

- Fix python version in pipeline to be consistent with lockfile
  ([`f5cf7bd`](https://github.com/StijnvLierop/DeepfakeDetection/commit/f5cf7bd4a02197430a7d7cb82e977eb22cbcb618))

- Remove dashboard as a front-end will be developed in a different repository
  ([`6f5b2cb`](https://github.com/StijnvLierop/DeepfakeDetection/commit/6f5b2cb816356c365684f8e09f467252243332e1))


## v0.13.0 (2025-08-20)

### Bug Fixes

- Remove msenv folder
  ([`7cc1994`](https://github.com/StijnvLierop/DeepfakeDetection/commit/7cc19948acecebeb349fe0ace9ddb3da10469ffa))

- **pipeline**: Setup cuda toolkit
  ([`1b5e689`](https://github.com/StijnvLierop/DeepfakeDetection/commit/1b5e68992d42dad7ed99ebf31cab7dcb36d8a30f))

### Chores

- Change cuda version in pipeline
  ([`533b66b`](https://github.com/StijnvLierop/DeepfakeDetection/commit/533b66bdf0a542dcea1dcc696b213df480e1a2d8))

### Features

- Add simple fingerprint model
  ([`1b5473a`](https://github.com/StijnvLierop/DeepfakeDetection/commit/1b5473a7b12e306f910b33234c63acb88379605b))

### Refactoring

- Remove cupy from dependencies
  ([`21d1111`](https://github.com/StijnvLierop/DeepfakeDetection/commit/21d1111a83aabfcbda7dcbd85390e371ac338ecd))


## v0.12.1 (2025-08-20)

### Bug Fixes

- Make arrows in aberration visualization unit scale
  ([`0d471b9`](https://github.com/StijnvLierop/DeepfakeDetection/commit/0d471b9fc14a890e881199a9a12744a0078eedcd))


## v0.12.0 (2025-08-15)

### Features

- Add annotation class that includes labels at multiple levels
  ([`6a17856`](https://github.com/StijnvLierop/DeepfakeDetection/commit/6a178561a19daf96935a0ba3bafdedd51c8634de))

- Make accuracy and roc-auc metrics compatible with labels on different levels
  ([`6345cf8`](https://github.com/StijnvLierop/DeepfakeDetection/commit/6345cf80158c1e3b085b690820770ef60d79e0ea))


## v0.11.0 (2025-08-14)

### Features

- **fft**: Add parameter to allow for optional filtering using hamming window
  ([`b079417`](https://github.com/StijnvLierop/DeepfakeDetection/commit/b0794178f54a188f0ae27b2df4368339ae1deeaa))


## v0.10.0 (2025-08-14)

### Bug Fixes

- Remove plot function in denoise_dncnn function
  ([`6718622`](https://github.com/StijnvLierop/DeepfakeDetection/commit/67186221d4e9c83bd12c65180276a20f737bfba0))

### Features

- Add denoiser from Zhang et al. 2017
  ([`430d6d6`](https://github.com/StijnvLierop/DeepfakeDetection/commit/430d6d6750461354700fbb88ea034b2b9651bc96))


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
