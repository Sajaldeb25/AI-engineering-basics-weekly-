CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a math expression. Use decimals for percentages (18% → 0.18 * x).",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Expression to evaluate, e.g. '4892 * 0.17 - 33'",
                }
            },
            "required": ["expression"],
        },
    },
}

GET_CURRENT_TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get current date and time for a timezone.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": (
                        "IANA timezone or 'local'/'UTC'. "
                        "Examples: Asia/Dhaka, Asia/Kolkata, Europe/Oslo."
                    ),
                }
            },
            "required": [],
        },
    },
}

SEARCH_WEB_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for current information, news, or facts.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {
                    "type": "string",
                    "description": "Number of results as digits, e.g. '3' or '5' (default '3', max '5')",
                },
            },
            "required": ["query"],
        },
    },
}

FILE_READ_TOOL = {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read the content of a file from the sandbox directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file to read (e.g. 'notes.txt')",
                    }
                },
                "required": ["filename"],
            },
        },
    }

FILE_WRITE_TOOL = {
        "type": "function",
        "function": {
            "name": "file_write",
            "description": (
                "Write content to a file in the sandbox directory. Overwrites if exists."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file to write (e.g. 'notes.txt')",
                    },
                    "content": {
                        "type": "string",
                        "description": "The text content to write to the file",
                    },
                },
                "required": ["filename", "content"],
            },
        },
    }

TOOLS = [CALCULATOR_TOOL, GET_CURRENT_TIME_TOOL, SEARCH_WEB_TOOL, FILE_READ_TOOL, FILE_WRITE_TOOL]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}