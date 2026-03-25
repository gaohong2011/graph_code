"""Example: Hello World with Graph Code

This is a simple example file that Graph Code can help you work with.
"""


def hello_world():
    """Print a hello world message."""
    print("Hello, World!")


def greet(name: str) -> str:
    """Greet someone by name.

    Args:
        name: The name to greet

    Returns:
        A greeting message
    """
    return f"Hello, {name}!"


if __name__ == "__main__":
    hello_world()
    print(greet("Graph Code"))
