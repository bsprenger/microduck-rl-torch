# Third-party notices

## MicroDuck robot model

The files under `assets/robot/microduck/` are derived from the MicroDuck model distributed with
the upstream Pollen Robotics project. The upstream repository identifies its source code as
Apache-2.0 and its 3D model assets as CC BY-SA-NC. Preserve the asset license and attribution
requirements when redistributing this repository.

## Official policy artifact

The validation tool downloads deployment policies from the public Hugging Face repository
[`pollen-robotics/microduck-policies`](https://huggingface.co/pollen-robotics/microduck-policies)
at runtime. Policy files are not committed to this repository; the resolved revision, manifest,
and digest are stored under the ignored `artifacts/hf/` directory for each local validation run.

