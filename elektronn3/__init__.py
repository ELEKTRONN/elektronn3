__all__ = ["cuda_enabled"]

import torch
cuda_enabled = torch.cuda.is_available()
print("cuda %s." % "enabled" if cuda_enabled else "disabled")