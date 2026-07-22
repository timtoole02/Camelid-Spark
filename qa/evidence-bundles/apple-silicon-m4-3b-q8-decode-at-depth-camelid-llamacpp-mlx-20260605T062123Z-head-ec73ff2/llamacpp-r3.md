load_backend: loaded BLAS backend from /opt/homebrew/Cellar/ggml/0.13.1/libexec/libggml-blas.so
ggml_metal_device_init: tensor API disabled for pre-M5 and pre-A19 devices
ggml_metal_library_init: using embedded metal library
ggml_metal_library_init: loaded in 0.107 sec
ggml_metal_rsets_init: creating a residency set collection (keep_alive = 180 s)
ggml_metal_device_init: GPU name:   MTL0 (Apple M4)
ggml_metal_device_init: GPU family: MTLGPUFamilyApple9  (1009)
ggml_metal_device_init: GPU family: MTLGPUFamilyCommon3 (3003)
ggml_metal_device_init: GPU family: MTLGPUFamilyMetal4  (5002)
ggml_metal_device_init: simdgroup reduction   = true
ggml_metal_device_init: simdgroup matrix mul. = true
ggml_metal_device_init: has unified memory    = true
ggml_metal_device_init: has bfloat            = true
ggml_metal_device_init: has tensor            = false
ggml_metal_device_init: use residency sets    = true
ggml_metal_device_init: use shared buffers    = true
ggml_metal_device_init: recommendedMaxWorkingSetSize  = 12713.12 MB
load_backend: loaded MTL backend from /opt/homebrew/Cellar/ggml/0.13.1/libexec/libggml-metal.so
load_backend: loaded CPU backend from /opt/homebrew/Cellar/ggml/0.13.1/libexec/libggml-cpu-apple_m4.so
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| llama 3B Q8_0                  |   3.18 GiB |     3.21 B | BLAS,MTL   |       4 |    tg64 @ d1500 |         25.69 ± 0.24 |
| llama 3B Q8_0                  |   3.18 GiB |     3.21 B | BLAS,MTL   |       4 |    tg64 @ d8012 |         18.97 ± 0.10 |

build: d48a56eff (9430)
