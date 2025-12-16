"""DataTable tool integration for textual-chat.

Provides tools that allow LLMs to query and manipulate Textual DataTable widgets.
"""

from collections.abc import Callable
from typing import Union

from rich.text import Text
from textual.coordinate import Coordinate
from textual.widgets import DataTable

# JSON type for table cell values
JSON = Union[dict[str, "JSON"], list["JSON"], str, int, float, bool, None]


def create_datatable_tools(table: DataTable, name: str = "table") -> dict[str, Callable]:
    """Create tool functions for accessing a DataTable.

    Args:
        table: The DataTable widget to expose
        name: Prefix for tool names (e.g., "sales" -> "sales_get_columns")

    Returns:
        Dictionary mapping tool names to functions.
    """

    def get_columns() -> list[str]:
        """Get the column names of the data table."""
        return [str(col.label) for col in table.columns.values()]

    def get_row_count() -> int:
        """Get the number of rows in the data table."""
        return table.row_count

    def get_all_data() -> list[dict[str, JSON]]:
        """Get all data from the table as a list of dictionaries."""
        columns = [str(col.label) for col in table.columns.values()]
        rows = []
        for row_key in table.rows:
            row_data = {}
            for i, col_key in enumerate(table.columns):
                cell = table.get_cell(row_key, col_key)
                row_data[columns[i]] = cell
            rows.append(row_data)
        return rows

    def get_row(index: int) -> dict[str, JSON]:
        """Get a specific row by index (0-based).

        Args:
            index: The row index (0-based)

        Returns:
            Dictionary with column names as keys and cell values as values.
        """
        columns = [str(col.label) for col in table.columns.values()]
        row_keys = list(table.rows.keys())
        if index < 0 or index >= len(row_keys):
            return {"error": f"Row index {index} out of range (0-{len(row_keys) - 1})"}
        row_key = row_keys[index]
        row_data = {}
        for i, col_key in enumerate(table.columns):
            cell = table.get_cell(row_key, col_key)
            row_data[columns[i]] = cell
        return row_data

    def highlight_cell(row_index: int, column: str, color: str) -> str:
        """Highlight a cell with a color.

        Args:
            row_index: The row index (0-based)
            column: The column name
            color: Color name (red, green, blue, yellow, magenta, cyan, etc.)

        Returns:
            Success or error message.
        """
        columns = [str(col.label) for col in table.columns.values()]
        if column not in columns:
            return f"Column '{column}' not found. Available: {columns}"
        row_keys = list(table.rows.keys())
        if row_index < 0 or row_index >= len(row_keys):
            return f"Row index {row_index} out of range (0-{len(row_keys) - 1})"
        row_key = row_keys[row_index]
        col_key = list(table.columns.keys())[columns.index(column)]
        current_value = table.get_cell(row_key, col_key)
        styled = Text(str(current_value), style=f"bold {color}")
        table.update_cell(row_key, col_key, styled)
        return f"Highlighted cell ({row_index}, {column}) with {color}"

    def highlight_row(row_index: int, color: str) -> str:
        """Highlight an entire row with a color.

        Args:
            row_index: The row index (0-based)
            color: Color name (red, green, blue, yellow, magenta, cyan, etc.)

        Returns:
            Success or error message.
        """
        row_keys = list(table.rows.keys())
        if row_index < 0 or row_index >= len(row_keys):
            return f"Row index {row_index} out of range (0-{len(row_keys) - 1})"
        row_key = row_keys[row_index]
        for col_key in table.columns:
            current_value = table.get_cell(row_key, col_key)
            styled = Text(str(current_value), style=f"bold {color}")
            table.update_cell(row_key, col_key, styled)
        return f"Highlighted row {row_index} with {color}"

    def move_cursor(row_index: int, column: str) -> str:
        """Move the table cursor to a specific cell.

        Args:
            row_index: The row index (0-based)
            column: The column name

        Returns:
            Success or error message.
        """
        columns = [str(col.label) for col in table.columns.values()]
        if column not in columns:
            return f"Column '{column}' not found. Available: {columns}"
        row_keys = list(table.rows.keys())
        if row_index < 0 or row_index >= len(row_keys):
            return f"Row index {row_index} out of range (0-{len(row_keys) - 1})"
        col_index = columns.index(column)
        table.cursor_coordinate = Coordinate(row_index, col_index)
        table.focus()
        return f"Cursor moved to ({row_index}, {column})"

    return {
        f"{name}_get_columns": get_columns,
        f"{name}_get_row_count": get_row_count,
        f"{name}_get_all_data": get_all_data,
        f"{name}_get_row": get_row,
        f"{name}_highlight_cell": highlight_cell,
        f"{name}_highlight_row": highlight_row,
        f"{name}_move_cursor": move_cursor,
    }
