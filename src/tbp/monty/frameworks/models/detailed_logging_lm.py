# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.mixins.no_reset_evidence import (
    TheoreticalLimitLMLoggingMixin,
)


class EvidenceGraphLMWithDetailedLogging(
    TheoreticalLimitLMLoggingMixin, EvidenceGraphLM
):
    """EvidenceGraphLM with theoretical limit and pose error logging."""

    pass
