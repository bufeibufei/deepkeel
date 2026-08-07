"""Optional MCP adapter SDK implementing the protocol-neutral ToolProvider port."""

from deepkeel.mcp import (
    McpCallResult,
    McpClient,
    McpClientPool,
    McpNormalizedResult,
    McpProtocolError,
    McpRemoteTool,
    McpServerSpec,
    McpTimeoutError,
    McpToolBinding,
    McpToolProvider,
    McpTransportError,
    StdioMcpClient,
    StreamableHttpMcpClient,
)

MCP_SDK_API = (
    "McpCallResult",
    "McpClient",
    "McpClientPool",
    "McpNormalizedResult",
    "McpProtocolError",
    "McpRemoteTool",
    "McpServerSpec",
    "McpTimeoutError",
    "McpToolBinding",
    "McpToolProvider",
    "McpTransportError",
    "StdioMcpClient",
    "StreamableHttpMcpClient",
)

__all__ = list(MCP_SDK_API)
