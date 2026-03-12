Place the following files in this directory before building the Docker image.
Download from: https://hailo.ai/developer-zone/software-downloads/
(Free registration required.)

Required files — exact filenames:
  hailo_dataflow_compiler-3.33.0-py3-none-linux_x86_64.whl
  hailort-4.23.0-cp311-cp311-linux_x86_64.whl

Version rationale:
  DFC 3.33 + HailoRT 4.23 (cp311 = Python 3.11) — current release.
  hailo_model_zoo v5.x dropped Hailo-8/8L support; the image uses v2.16.

These files are NOT committed to the repository (they are listed in .gitignore).
