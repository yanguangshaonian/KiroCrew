"""旧 ACP 接口的按需兼容导出."""

from importlib import import_module

_EXPORTS = {
    "AcpClient": "client",
    "AcpConversationBindingMismatch": "client",
    "AcpError": "client",
    "AcpPermissionNeeded": "client",
    "AcpProcessDied": "client",
    "AcpRegistrationRateLimited": "client",
    "AcpTimeoutError": "client",
    "AcpRuntime": "runtime",
    "AcpSessionHandle": "runtime",
    "AcpEvent": "types",
    "AcpPromptStats": "types",
    "JsonRpcMessage": "types",
    "JsonRpcRequest": "types",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"未知 ACP 兼容接口: {name}.")
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value
