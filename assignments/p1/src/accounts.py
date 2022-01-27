from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Usage:
    """Usage information (for a given account, model group, and granularity)."""

    # What period it is (so we know when to reset `used`) - for exmaple, for
    # daily granularity, period might be 2021-12-30
    period: Optional[str] = None

    # How many tokens was used
    used: int = 0

    # How much quota do we have (None means unlimited)
    quota: Optional[int] = None

    def update_period(self, period: str):
        if self.period != period:
            self.period = period
            self.used = 0  # Reset in a new period

    def can_use(self):
        return self.quota is None or self.used < self.quota


@dataclass
class Account:
    """An `Account` provides access to the API."""

    # Unique API key that is used both for authentication and for identification.
    # Like credit card numbers, this is a bit of a shortcut since we're trying
    # to avoid building out a full-blown system.  If an API key needs to be
    # replaced, we can simply change it and keep the other data the same.
    api_key: str

    # What this account is used for (can include the user names)
    description: str = ""

    # Emails associated this account
    emails: List[str] = field(default_factory=list)

    # What groups this account is associated with
    groups: List[str] = field(default_factory=list)

    # Whether this account has admin access (e.g., ability to modify accounts)
    is_admin: bool = False

    # Usage is tracked and limited at different granularities
    # `usages`: model group -> granularity -> Usage
    usages: Dict[str, Dict[str, Usage]] = field(default_factory=dict)
