"""Helpers for Progressive Web App metadata in the Streamlit portal."""

import streamlit.components.v1 as components


def enable_pwa() -> None:
    """Attach the app manifest to Streamlit's document head."""
    components.html(
        """
        <script>
        const appWindow = window.parent;
        const appDocument = appWindow.document;
        let manifestLink = appDocument.querySelector('link[rel="manifest"]');
        if (!manifestLink) {
            manifestLink = appDocument.createElement("link");
            manifestLink.rel = "manifest";
            appDocument.head.appendChild(manifestLink);
        }
        manifestLink.href = new URL(
            "app/static/manifest.json",
            appWindow.location.href
        ).href;

        let themeColor = appDocument.querySelector('meta[name="theme-color"]');
        if (!themeColor) {
            themeColor = appDocument.createElement("meta");
            themeColor.name = "theme-color";
            appDocument.head.appendChild(themeColor);
        }
        themeColor.content = "#0f172a";
        </script>
        """,
        height=0,
    )
