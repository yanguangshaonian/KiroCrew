"""Dashboard token authentication.

HMAC-SHA256 token generation, validation, IP binding, consumption
tracking, and aiohttp middleware for Slack-gated dashboard access.
"""

from __future__ import annotations

import importlib as _importlib
import logging
import sys as _sys
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from aiohttp import web

from kiro_crew import _facade_mirror

# Canonical revocation-generation definitions live in revocation_gen so the
# refresh-token module can consult the counter without recreating the
# token_auth <-> refresh_tokens import cycle (same pattern as token_secret
# below). Re-exported here for backwards compatibility. Both validators read
# the LIVE value via current_revocation_gen() — never an import-time copy —
# so a bump is visible to every subsequent validation in-process.
from kiro_crew.dashboard.revocation_gen import (  # noqa: F401  # re-exports
    _REVOCATION_FILE,
    _load_revocation_gen,
    bump_revocation_gen,
    current_revocation_gen,
    current_revocation_gen_or_none,
)
from kiro_crew.dashboard.tailnet import (
    TailnetTrust,
)
from kiro_crew.dashboard.token_auth_tokens import (
    _ct_eq,
)

# Canonical HMAC-secret definitions live in token_secret to break the import
# cycle between token_auth and refresh_tokens. Re-exported here for backwards
# compatibility — callers elsewhere import these names from token_auth. The
# fork keeps the LAZY _get_secret() (NOT an eager module-level _SECRET =
# _load_or_create_secret()) so that merely importing this module never writes
# token_signing.key into $KIROCREW_HOME (the CLI imports token_auth for every
# kirocrew subcommand; an import-time write would break gateway --seed and
# pollute the home for read-only commands).
from kiro_crew.dashboard.token_secret import (  # noqa: F401  # re-exports
    _SECRET_KEY_FILE,
    _get_secret,
    _load_or_create_secret,
)


def internal_path_matches(path: str, entries: Iterable[str]) -> bool:
    """Return whether *path* is an exact or child path of an internal route."""
    return any(path == entry or path.startswith(entry + "/") for entry in entries)


logger = logging.getLogger("kiro_crew.dashboard.token_auth")


def token_auth_middleware(
    *,
    internal_paths: frozenset[str] = frozenset(),
    mixed_internal_paths: frozenset[str] = frozenset(),
    internal_secret: str = "",
    port: int = 5476,
    local_only: bool = True,
    spa_shell_handler: Callable[..., Any] | None = None,
    tailnet_trust: TailnetTrust | None = None,
) -> Callable[..., Any]:
    """为单用户网页设置本地身份, 保留内部工具连接标记."""

    @web.middleware
    async def middleware(request: web.Request, handler: object) -> web.StreamResponse:
        provided_secret = request.headers.get("X-Internal-Secret", "")
        if provided_secret and internal_secret and _ct_eq(internal_secret, provided_secret):
            request["internal_auth"] = True
            return await handler(request)
        request["user"] = "local-owner"
        request["app"] = ""
        request["is_dashboard_user"] = True
        request["auth_token"] = ""
        request["auth_from_query_token"] = False
        return await handler(request)

    middleware._is_token_auth = True  # type: ignore[attr-defined]  # sentinel for server.py security gate
    return middleware


if TYPE_CHECKING:  # every owner-defined name in _EXPORTS, for type checkers and IDEs
    from kiro_crew.dashboard.token_auth_app_scope import (  # noqa: F401
        _APP_PERMS_PRIME_MARGIN,
        _APP_PERMS_TTL,
        _APP_TOKEN_IMPLICIT_ALLOW,
        RESERVED_APP_PATH_SEGMENTS,
        _api_pattern_matches,
        _app_api_allowlist,
        _app_owns_path,
        _app_perms_cache,
        _app_perms_lock,
        app_token_path_allowed,
    )
    from kiro_crew.dashboard.token_auth_callers import (  # noqa: F401
        KNOWN_INTERNAL_CALLERS,
        MEMBER_CHAT_PRINCIPAL_KEY,
        _bound_session_keys_on_chain,
        _cron_job_exists,
        _cron_job_owner,
        _key_segment,
        _live_session_keys_on_chain,
        _slot_by_linked_key,
        _subagent_caller_record,
        _subagent_owner,
        _subagent_record_exists,
        app_owns_transcript,
        caller_names_a_missing_slot,
        caller_record_is_missing,
        derive_caller_app,
        effective_request_app,
        folder_principal,
        refuse_unattributable_caller,
        request_origin,
    )
    from kiro_crew.dashboard.token_auth_paths import (  # noqa: F401
        _403_HTML,
        _APP_WINDOW_EXCLUDED_PATHS,
        _APPS_SPA_EXCLUDED_RE,
        _SELF_AUTH_WEBHOOK_METHODS,
        AGENT_HOOK_PATH,
        CSRF_EXEMPT_EXACT_METHODS,
        SPA_FALLBACK_EXCLUDED_PREFIXES,
        TEAMS_WEBHOOK_PATH,
        UPDATE_REVALIDATE_PATH,
        _credential_fingerprint,
        is_csrf_exempt,
        register_app_window_paths,
    )
    from kiro_crew.dashboard.token_auth_tokens import (  # noqa: F401
        _REVOKED_NONCES_FILE,
        LINK_WINDOW_SECS,
        MAX_CONCURRENT_NONCES,
        RevokedNonceStore,
        TokenStateManager,
        _b64url_encode,
        _evict_expired,
        _revoked_store_lock,
        _revoked_store_singleton,
        _sign,
        _token_pin_key,
        bind_token_ip,
        check_token_ip,
        extract_claims_from_token,
        extract_numeric_claim,
        extract_prompt_from_token,
        generate_app_secret,
        is_consumed,
        mark_consumed,
        parse_duration,
        proxied_pin_observed,
        revoke_access_cookie,
        revoke_all_sessions,
        token_embed_parent_port,
        try_consume,
        validate_app_secret,
        validate_token,
        warm_auth_singletons,
        write_app_secret,
    )

if TYPE_CHECKING:  # names it serves from outside the family, from their homes
    from kiro_crew import (  # noqa: F401
        platform_compat,
    )
    from kiro_crew.dashboard.boot_id import (  # noqa: F401
        current_boot_id,
    )
    from kiro_crew.dashboard.origin import (  # noqa: F401
        request_is_unix_socket,
    )
    from kiro_crew.executors import (  # noqa: F401
        subprocess_executor,
    )
    from kiro_crew.mcp_gateway.socketsec import (  # noqa: F401
        PeerCredResult,
        check_peer_is_self,
        get_peer_pid,
    )
    from kiro_crew.messaging.link import (  # noqa: F401
        is_channel_session_key,
    )
    from kiro_crew.peer_resolve import (  # noqa: F401
        resolve_peer_tenancy,
    )
    from kiro_crew.runtime_ownership import (  # noqa: F401
        session_keys_bound_to_pid,
    )
    from kiro_crew.session_token_sig import (  # noqa: F401
        verify_session_token,
    )

# This module is the facade of dashboard token authentication: every name is reached through
# it as ``kiro_crew.dashboard.token_auth.<name>``. A read is served by the owner that binds
# the name, and a write -- ``monkeypatch.setattr``, ``mock.patch`` -- reaches every owner
# holding it (:mod:`kiro_crew._facade_mirror`). Owners never import this module. The
# middleware above stays here. New work goes to the owner of its responsibility:
#
#   ``token_auth_paths`` -- requests that skip token auth, and the refusal the rest receive
#   ``token_auth_tokens`` -- minting, validating, pinning and revoking session tokens
#   ``token_auth_app_scope`` -- what an app token may reach
#   ``token_auth_callers`` -- Unix-socket peer verification and request attribution
#: Owner modules, lowest layer first: the first holder of a name answers its reads.
_OWNERS = (
    "kiro_crew.dashboard.token_auth_paths",
    "kiro_crew.dashboard.token_auth_tokens",
    "kiro_crew.dashboard.token_auth_app_scope",
    "kiro_crew.dashboard.token_auth_callers",
)
for _owner in _OWNERS:
    _importlib.import_module(_owner)
#: Name the facade serves but does not bind -> the owner that answers a read of it.
_EXPORTS: dict[str, str] = {}
_EXPORTS.update(_facade_mirror.install(_sys.modules[__name__], _OWNERS, excluded=()).served)
_sys.modules[__name__].__class__ = _facade_mirror.MirroredModule
#: What a star import binds: every public name of the unsplit module's namespace.
__all__ = sorted(n for n in set(globals()) | set(_EXPORTS) if not n.startswith("_"))
