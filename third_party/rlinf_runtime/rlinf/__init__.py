# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

# Temporary, opt-in override to validate checkpoints trained against an older
# cosmos-framework commit (e.g. one still implementing joint_attn_implementation
# "flex", since removed upstream). No-op unless the env var is set.
_cosmos_framework_override = os.environ.get("RLINF_COSMOS_FRAMEWORK_OVERRIDE")
if _cosmos_framework_override:
    sys.path.insert(0, _cosmos_framework_override)
    print(
        f"[RLINF_DEBUG pid={os.getpid()}] override={_cosmos_framework_override!r} "
        f"already_imported={'cosmos_framework' in sys.modules!r} "
        f"cached_file={getattr(sys.modules.get('cosmos_framework'), '__file__', None)!r}",
        flush=True,
    )

from .utils.omega_resolver import omegaconf_register

omegaconf_register()
