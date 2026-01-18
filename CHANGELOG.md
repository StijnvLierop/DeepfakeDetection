# CHANGELOG


## v0.29.3 (2026-01-18)

### Bug Fixes

- Add preprocessing step to cnndetect to convert RGBA images to RGB
  ([`7c324b9`](https://github.com/StijnvLierop/DeepfakeDetection/commit/7c324b9fb215f52b23115c69becf5bddb80449d4))


## v0.29.2 (2026-01-17)

### Bug Fixes

- Filtered dataset and separated sampling logic
  ([`45705b1`](https://github.com/StijnvLierop/DeepfakeDetection/commit/45705b1ceb281d77afa07523439d74684c637ca1))

- Load data from disk is now directly supported from path
  ([`75bd77b`](https://github.com/StijnvLierop/DeepfakeDetection/commit/75bd77baab80cc09058b06d3497b302fd9d3061e))

- Load_dataset function to correctly load functions
  ([`79f6423`](https://github.com/StijnvLierop/DeepfakeDetection/commit/79f64231a60413182e135dbdb1c1dc1e25b08fdf))

- Module import path
  ([`157d4d6`](https://github.com/StijnvLierop/DeepfakeDetection/commit/157d4d64cee5fbd501e8a29174294b0f3bff51d6))

- Remove unused import from tests
  ([`0efab2b`](https://github.com/StijnvLierop/DeepfakeDetection/commit/0efab2be6b30ce35c1bf25e15718625fde70895f))


## v0.29.1 (2026-01-09)

### Bug Fixes

- Remove obsolete test
  ([`d8d90e3`](https://github.com/StijnvLierop/DeepfakeDetection/commit/d8d90e3272b85372b270c28fdfd990637c341fbf))

### Chores

- Update evaluation.py
  ([`93c1127`](https://github.com/StijnvLierop/DeepfakeDetection/commit/93c11271b6199e73509f426432b3944e9d435b90))


## v0.29.0 (2026-01-09)

### Bug Fixes

- Code formatting
  ([`4c78ca6`](https://github.com/StijnvLierop/DeepfakeDetection/commit/4c78ca608c38623ab6497ec4d99b228c240a9219))

- Imports in evaluate.py
  ([`8a46fa6`](https://github.com/StijnvLierop/DeepfakeDetection/commit/8a46fa6c1fa397c0305c8f31b992e2adf86e9144))

- Test imports
  ([`b1e1616`](https://github.com/StijnvLierop/DeepfakeDetection/commit/b1e161699a434b5c1315ee9f86f4bf34361f97ec))

- Unit tests
  ([`361ec2c`](https://github.com/StijnvLierop/DeepfakeDetection/commit/361ec2cddecdd1227fa7d6aaeb2f10717c7858e8))

- Unit tests
  ([`5cb8205`](https://github.com/StijnvLierop/DeepfakeDetection/commit/5cb8205dfcc31e9c815d16d33d4382e7c05a2f46))

### Chores

- Fix code formatting issues
  ([`621ed59`](https://github.com/StijnvLierop/DeepfakeDetection/commit/621ed592b0bb394a81b757021f0f31a318536f87))

- Run unit tests with verbose output in pipeline
  ([`a07e626`](https://github.com/StijnvLierop/DeepfakeDetection/commit/a07e6260c437c8a01f1f7589e9ae3a1a55ae376b))

### Features

- Add combineddataset
  ([`5efa5ef`](https://github.com/StijnvLierop/DeepfakeDetection/commit/5efa5ef5ec46d3b41bb445ac821df2fc7f24a201))

- Add configuration options to load datasets from yaml file
  ([`5583daf`](https://github.com/StijnvLierop/DeepfakeDetection/commit/5583dafe9b8f920981928ee02a7c7acd34af817b))

- Add wildcard label mapping feature to huggingface dataset
  ([`a86ec75`](https://github.com/StijnvLierop/DeepfakeDetection/commit/a86ec750c679d0dc4afd357cefd50994dbfbff7d))


## v0.28.2 (2026-01-08)

### Bug Fixes

- Imports in freqnet
  ([`11a4dbd`](https://github.com/StijnvLierop/DeepfakeDetection/commit/11a4dbd595697811af338b596a3bfd824bbd02c1))

- Imports in univfd
  ([`ccb2904`](https://github.com/StijnvLierop/DeepfakeDetection/commit/ccb2904859ba32610a43c313e5fbbcc92c22fda9))

- Small import module mistakes in models
  ([`41cbd4d`](https://github.com/StijnvLierop/DeepfakeDetection/commit/41cbd4dbe1a3a00dd3f7597f52e1ce5f883cbe04))


## v0.28.1 (2026-01-08)

### Bug Fixes

- Add diffusers package to dependencies
  ([`7a5860b`](https://github.com/StijnvLierop/DeepfakeDetection/commit/7a5860b4bf64f9913006930ea30805d4db286932))

- Formatting
  ([`abb085e`](https://github.com/StijnvLierop/DeepfakeDetection/commit/abb085eae1e88a6ddf6d34b66ae5173029e28273))

- Package name missing in pyproject.toml dependency
  ([`15fd2ef`](https://github.com/StijnvLierop/DeepfakeDetection/commit/15fd2ef20917e62f59e4ee8f36bc94a404bda920))

- Remove build files
  ([`faa3fd1`](https://github.com/StijnvLierop/DeepfakeDetection/commit/faa3fd10b268385dd8f4236d8a55a87bbac38e7d))

- Ruff formatting
  ([`2683e44`](https://github.com/StijnvLierop/DeepfakeDetection/commit/2683e449292dafc01f33384aa2f34eb2cdd230ac))

- Simplify ruff use in pipeline
  ([`2f1a771`](https://github.com/StijnvLierop/DeepfakeDetection/commit/2f1a771229bd444a3f82d43dd082ac51664b7aaa))

- Tests formatting and fixture imports
  ([`3d39e7c`](https://github.com/StijnvLierop/DeepfakeDetection/commit/3d39e7c9bd716999a8974c7f5767345dea36f856))

### Chores

- Add ruff formatting checks to pipeline
  ([`d3d0add`](https://github.com/StijnvLierop/DeepfakeDetection/commit/d3d0add6f02e6d5246250b3f42edc34b83801efe))

### Refactoring

- Overhaul evaluation script to work with updated models
  ([`58f44b6`](https://github.com/StijnvLierop/DeepfakeDetection/commit/58f44b6d17732c3d7973481088700af125644eaf))


## v0.28.0 (2026-01-02)

### Features

- Add fatformer model
  ([`fb70f0a`](https://github.com/StijnvLierop/DeepfakeDetection/commit/fb70f0a78c669dc2d3fc40d0d4997cbcc5468c99))


## v0.27.0 (2026-01-02)

### Bug Fixes

- **latte**: Fix torch transforms
  ([`0976c0e`](https://github.com/StijnvLierop/DeepfakeDetection/commit/0976c0ea82a8b5fdc94ef586dd8f61e98163ca0c))


## v0.26.0 (2026-01-02)


## v0.25.3 (2026-01-02)

### Bug Fixes

- Loading of weights with different formats in npr model
  ([`f19e305`](https://github.com/StijnvLierop/DeepfakeDetection/commit/f19e305ea8b2abb47bd1b29d27b523ddbb66d7cd))

- **npr**: Set transformations equal to genimage testing parameters in paper
  ([`3fad3e3`](https://github.com/StijnvLierop/DeepfakeDetection/commit/3fad3e3e9f46622ebe970e7a2491e55203db10f3))


## v0.25.2 (2026-01-02)

### Bug Fixes

- **univfd**: Reorder transforms to get same inference results as in paper
  ([`e4d0e30`](https://github.com/StijnvLierop/DeepfakeDetection/commit/e4d0e30a5be90173e1a76d5819c8f92c90ddadcb))

### Chores

- Remove build files
  ([`3bcc0a2`](https://github.com/StijnvLierop/DeepfakeDetection/commit/3bcc0a219282775df2214727ad1a03659f3d2f78))

### Features

- Add freqnet model
  ([`9970c07`](https://github.com/StijnvLierop/DeepfakeDetection/commit/9970c07ae76247f0c66f6ed3cc612522ae0ab2b2))

### Refactoring

- Make univfd trainable
  ([`fa29ecb`](https://github.com/StijnvLierop/DeepfakeDetection/commit/fa29ecb913aa18705196101564cfb29525776fed))


## v0.25.1 (2025-12-27)

### Bug Fixes

- Authenticity label in diffusion dataset
  ([`d41e413`](https://github.com/StijnvLierop/DeepfakeDetection/commit/d41e41318ccdb639dbfa54d71c4a3de4602f3e7f))


## v0.25.0 (2025-12-27)

### Features

- Add diffusiondataset used by univfd
  ([`b80201f`](https://github.com/StijnvLierop/DeepfakeDetection/commit/b80201f725aa2cd71dbfdf349994b1c739c387c0))


## v0.24.2 (2025-12-26)

### Bug Fixes

- Loading old cnndetect model weights
  ([`33dc9e6`](https://github.com/StijnvLierop/DeepfakeDetection/commit/33dc9e69942bf7e20828cb403acf124e28b1861e))


## v0.24.1 (2025-12-24)


## v0.24.0 (2025-12-24)

### Bug Fixes

- Cnndetect transform function now transforms inputs instead of returning the transform function
  ([`881ddda`](https://github.com/StijnvLierop/DeepfakeDetection/commit/881dddacea75e9dbf5af4885e39f187e1b5191f0))

- Genimage dataset is now a mapstyle dataset
  ([`db38efd`](https://github.com/StijnvLierop/DeepfakeDetection/commit/db38efdbe7abf90a162e3659c29fadabb9c18f9d))

- Sigma variable naming in randomgaussianblur augmentation
  ([`0a07380`](https://github.com/StijnvLierop/DeepfakeDetection/commit/0a0738037676fe064899718d789f92478ba1b45d))

### Features

- Add blur and compression to data augmentations
  ([`ebd0dd8`](https://github.com/StijnvLierop/DeepfakeDetection/commit/ebd0dd840464dafc19aa9fe8b0dc9c7c5b1c86e6))

- Add support for huggingface trainer and trainingarguments
  ([`0f9af94`](https://github.com/StijnvLierop/DeepfakeDetection/commit/0f9af94ebc6a9f34c7ce0831b5e6c86cbb5bda19))

### Refactoring

- Change default augmentation params to be same as in cnndetect paper
  ([`1853c3d`](https://github.com/StijnvLierop/DeepfakeDetection/commit/1853c3d6fb92f695914da1d9ef3e34588ad057b1))

- Change transform func in cnndetect
  ([`65370b4`](https://github.com/StijnvLierop/DeepfakeDetection/commit/65370b4ddb2cd4b8ef1df0cfa25be0c5450d6bbe))

- Transform inputs function now returns function instead of transformed inputs
  ([`ab8c545`](https://github.com/StijnvLierop/DeepfakeDetection/commit/ab8c5454d4dd1ecd8e1fa504de1632984c425f7f))


## v0.23.0 (2025-12-21)

### Features

- Add evaluator class to quickly calculate metrics
  ([`40d6203`](https://github.com/StijnvLierop/DeepfakeDetection/commit/40d620394e7483d222ee75fb9a1abec4b216b985))


## v0.22.0 (2025-12-20)

### Features

- Add cnndetect dataset
  ([`b8f4794`](https://github.com/StijnvLierop/DeepfakeDetection/commit/b8f4794095412ab970e5c8db6050a9b7db82b032))


## v0.21.0 (2025-12-20)

### Features

- Add support for pytorch training
  ([`d1654ae`](https://github.com/StijnvLierop/DeepfakeDetection/commit/d1654ae58f35a14ae9c01447e32996c15b40a929))

### Refactoring

- Fix formatting
  ([`04953b9`](https://github.com/StijnvLierop/DeepfakeDetection/commit/04953b96e1949897d8762b87623ecd20a1eb11ac))

### Testing

- Fix unit tests
  ([`1a1407e`](https://github.com/StijnvLierop/DeepfakeDetection/commit/1a1407e951dd0c14183b51439c3f076f1c2523ae))


## v0.20.0 (2025-12-13)

### Features

- Move predict function to model class so models only have to implement predict_batch function
  ([`d44a078`](https://github.com/StijnvLierop/DeepfakeDetection/commit/d44a07804684eb9a38faf99695bc0665b3e2f374))


## v0.19.0 (2025-12-11)

### Features

- Add first version of npr model
  ([`92cc963`](https://github.com/StijnvLierop/DeepfakeDetection/commit/92cc963878b293a6460736fc3eeda486c89e268b))

- Correctly load model weights
  ([`d10b652`](https://github.com/StijnvLierop/DeepfakeDetection/commit/d10b6525fc77374b4457ca2612c3d006fb6afc09))

- First version of latte model
  ([`a903df5`](https://github.com/StijnvLierop/DeepfakeDetection/commit/a903df561ddc8d0600b7356a09e2106a03042954))

### Refactoring

- Add missing dependency and fix module import
  ([`ecd30d3`](https://github.com/StijnvLierop/DeepfakeDetection/commit/ecd30d3cbffbbba7d7e76ff29785616e3d9ed52e))


## v0.18.0 (2025-12-08)

### Features

- Add univfd model
  ([`5952267`](https://github.com/StijnvLierop/DeepfakeDetection/commit/59522672a60766cdd42ecbdf392c904e0881ff9f))


## v0.17.0 (2025-12-08)

### Features

- Add CNNDetect model
  ([`ab9ca2c`](https://github.com/StijnvLierop/DeepfakeDetection/commit/ab9ca2c85245fc3c024e9a68e913a6ba1acb1b2f))

- Add dataset batching and batched model predictions
  ([`5807707`](https://github.com/StijnvLierop/DeepfakeDetection/commit/5807707aab4b1c3b8f555116a5c363f26e27b6b3))

### Refactoring

- Module imports
  ([`3356500`](https://github.com/StijnvLierop/DeepfakeDetection/commit/3356500dc56decc45ab9d54fc47ff8244eba58fb))

- Remove weights from repo
  ([`0e618bb`](https://github.com/StijnvLierop/DeepfakeDetection/commit/0e618bbf07d924bfea3960ff28273d56774d93b9))


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
