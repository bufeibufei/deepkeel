"""Optional MCP adapter SDK implementing the protocol-neutral ToolProvider port."""

from deepkeel.mcp import (
    MCP_PROTOCOL_VERSION,
    MCP_TASKS_EXTENSION,
    SUPPORTED_MCP_PROTOCOL_VERSIONS,
    McpCallResult,
    McpClient,
    McpClientPool,
    McpInputRequest,
    McpNormalizedResult,
    McpProtocolError,
    McpRemoteError,
    McpRemoteTool,
    McpServerSpec,
    McpTask,
    McpTimeoutError,
    McpToolBinding,
    McpToolProvider,
    McpTransportError,
    StdioMcpClient,
    StreamableHttpMcpClient,
)

MCP_SDK_API = (
    "MCP_PROTOCOL_VERSION",
    "MCP_TASKS_EXTENSION",
    "SUPPORTED_MCP_PROTOCOL_VERSIONS",
    "McpCallResult",
    "McpClient",
    "McpClientPool",
    "McpInputRequest",
    "McpNormalizedResult",
    "McpProtocolError",
    "McpRemoteError",
    "McpRemoteTool",
    "McpServerSpec",
    "McpTask",
    "McpTimeoutError",
    "McpToolBinding",
    "McpToolProvider",
    "McpTransportError",
    "StdioMcpClient",
    "StreamableHttpMcpClient",
)

__all__ = list(MCP_SDK_API)
