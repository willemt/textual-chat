"""Example app with tabbed interface and Chat overlay."""

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Input, Label, TabbedContent, TabPane

from textual_chat import Chat


class TabbedApp(App):
    """An app with multiple tabs and a chat overlay."""

    CSS = """
    TabbedContent {
        height: 1fr;
    }

    TabPane {
        padding: 1 2;
    }

    #sales {
        height: 1fr;
    }

    #inventory {
        height: 1fr;
    }

    .form-row {
        height: auto;
        margin-bottom: 1;
    }

    .form-row Label {
        width: 15;
    }

    .form-row Input {
        width: 1fr;
    }

    Chat {
        dock: right;
        width: 40%;
        height: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal():
            with TabbedContent():
                with TabPane("Sales", id="tab-sales"):
                    yield DataTable(id="sales")

                with TabPane("Inventory", id="tab-inventory"):
                    yield DataTable(id="inventory")

                with TabPane("Settings", id="tab-settings"):
                    with Vertical():
                        with Horizontal(classes="form-row"):
                            yield Label("Username:")
                            yield Input(id="username", placeholder="Enter username")
                        with Horizontal(classes="form-row"):
                            yield Label("API Key:")
                            yield Input(
                                id="api-key", placeholder="Enter API key", password=True
                            )
                        with Horizontal(classes="form-row"):
                            yield Button("Save", id="save-btn", variant="primary")
                            yield Button("Reset", id="reset-btn")

            yield Chat(show_token_usage=True)

    def on_mount(self) -> None:
        # Populate sales table
        sales = self.query_one("#sales", DataTable)
        sales.add_columns("Product", "Q1", "Q2", "Q3", "Q4", "Total")
        sales.add_rows(
            [
                ("Widgets", 150, 200, 180, 220, 750),
                ("Gadgets", 80, 95, 110, 130, 415),
                ("Gizmos", 200, 180, 190, 210, 780),
                ("Doodads", 50, 60, 55, 70, 235),
            ]
        )

        # Populate inventory table
        inventory = self.query_one("#inventory", DataTable)
        inventory.add_columns("Item", "SKU", "In Stock", "Reorder Point", "Status")
        inventory.add_rows(
            [
                ("Widgets", "WGT-001", 500, 100, "OK"),
                ("Gadgets", "GDG-002", 80, 150, "Low"),
                ("Gizmos", "GZM-003", 300, 50, "OK"),
                ("Doodads", "DOD-004", 45, 75, "Critical"),
                ("Thingamajigs", "THG-005", 200, 100, "OK"),
            ]
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            self.notify("Settings saved!")
        elif event.button.id == "reset-btn":
            self.query_one("#username", Input).value = ""
            self.query_one("#api-key", Input).value = ""
            self.notify("Settings reset!")


if __name__ == "__main__":
    app = TabbedApp()
    app.run()
