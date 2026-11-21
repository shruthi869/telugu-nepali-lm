"""Compatibility composition of the three deduplication responsibilities."""

from .assembly import AssemblyMixin
from .exact_dedup import ExactDedupMixin
from .near_dedup import NearDedupMixin


class DeduplicationMixin(ExactDedupMixin, NearDedupMixin, AssemblyMixin):
    pass
