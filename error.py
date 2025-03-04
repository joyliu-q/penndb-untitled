from enum import Enum
import json
import typing as t
from openai import OpenAI


class PipelineError(Enum):
    """
    High-level categories of errors that can occur in a generic ML/data pipeline.
    Following HTTP-like status code structure:
    - 2xx: Success codes
    - 4xx: Input/Internal errors (can be resolved by input changes or pipeline fixes)
    - 5xx: Output/External errors (issues with external services or output processing)
    """

    # 2xx - Success
    SUCCESS = (200, "Success", "Operation completed successfully")

    # 4xx - Input/Internal Errors
    BAD_REQUEST = (400, "Bad Request", "Invalid input parameters, request, or schema")
    UNAUTHORIZED = (401, "Unauthorized", "Authentication or authorization failure")
    NOT_FOUND = (404, "Not Found", "Requested resource or data not found")
    TIMEOUT = (408, "Timeout", "Operation timed out")
    LIMIT_EXCEEDED = (429, "Limit Exceeded", "Rate limit or resource quota exceeded")

    # 5xx - Output/External Errors
    EXTERNAL_ERROR = (500, "External Error", "Unexpected pipeline failure")
    SERVICE_UNAVAILABLE = (
        503,
        "Service Unavailable",
        "External service or dependency unavailable",
    )
    BAD_RESPONSE = (
        590,
        "Bad Response",
        "Invalid or unexpected output, response, or data format",
    )

    def __init__(self, code, name, description):
        self._code = code
        self._name = name
        self._description = description

    def __int__(self):
        return self._code

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def category(self):
        """Returns the category description based on the error code range"""
        if self._code < 300:
            return "Success"
        elif self._code < 500:
            return "Input/Internal Error - Can be resolved by input changes or pipeline fixes"
        else:
            return "Output/External Error - Issues with external services or output processing"

    @classmethod
    def classify_error(
        cls, openai_client: OpenAI, error_description: str
    ) -> t.Tuple["PipelineError", str]:
        """
        Use LLM to classify the error type and generate a standardized description.
        """
        functions = [
            {
                "name": "classify_error",
                "description": """Classify an error into a predefined category.

            4xx - Input/Internal errors:
            - BAD_REQUEST: Invalid input parameters or malformed request data
            - UNAUTHORIZED: Authentication/authorization failure
            - NOT_FOUND: Resource/data not found
            - TIMEOUT: Operation timed out
            - LIMIT_EXCEEDED: Rate limit/quota exceeded

            5xx - Output/External errors:
            - EXTERNAL_ERROR: Unexpected pipeline failure
            - SERVICE_UNAVAILABLE: External service unavailable
            - BAD_RESPONSE: Output data quality/format issues (malformed data, invalid values, failed validation)
            """,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": [
                                "SUCCESS",
                                "BAD_REQUEST",
                                "UNAUTHORIZED",
                                "NOT_FOUND",
                                "TIMEOUT",
                                "LIMIT_EXCEEDED",
                                "EXTERNAL_ERROR",
                                "SERVICE_UNAVAILABLE",
                                "BAD_RESPONSE",
                            ],
                            "description": "The error category that best matches the description.",
                        }
                    },
                    "required": ["category"],
                },
            }
        ]

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a data error classifier."},
                    {
                        "role": "user",
                        "content": f"Classify this error: {error_description}",
                    },
                ],
                functions=functions,
                function_call={"name": "classify_error"},
            )

            try:
                function_args = json.loads(response.choices[0].message.function_call.arguments)
            except:
                function_args = eval(response.choices[0].message.function_call.arguments)

            category_str = function_args["category"]
            category = getattr(cls, category_str, cls.BAD_REQUEST)
        except Exception as e:
            print(f"Error classifying error: {error_description}")
            print(f"Error: {e}")
            category = cls.BAD_REQUEST

        return category, error_description
