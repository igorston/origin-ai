"""Optional accounts (ORIGIN_AUTH=password): users, session cookies, the middleware.

Manage users with `python -m origin.auth --help`.
"""

import logging

from origin.auth.users import User, UserError, UserStore
from origin.config import Settings

logger = logging.getLogger(__name__)


def bootstrap_admin(settings: Settings, users: UserStore) -> None:
    """With no users yet, create the first admin from ORIGIN_ADMIN_PASSWORD (if set)."""
    if users.count():
        return
    if not settings.origin_admin_password:
        logger.warning(
            "Authentication is on but there are no users: create one with "
            "`python -m origin.auth add-user NAME --admin`, or set ORIGIN_ADMIN_PASSWORD"
        )
        return
    users.add(settings.origin_admin_user, settings.origin_admin_password, is_admin=True)
    logger.info(
        "Created the first admin, %r (from ORIGIN_ADMIN_PASSWORD)", settings.origin_admin_user
    )


__all__ = ["User", "UserError", "UserStore", "bootstrap_admin"]
