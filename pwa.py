"""Helpers for Progressive Web App metadata in the Streamlit portal."""

import streamlit.components.v1 as components


def enable_pwa() -> None:
    """Attach the app manifest to Streamlit's document head."""
    components.html(
        """
        <script>
        const appWindow = window.parent;
        const appDocument = appWindow.document;

        function addManifest() {
            if (!appDocument.head) return;

            let manifestLink = appDocument.querySelector(
                'link[rel="manifest"]'
            );

            if (!manifestLink) {
                manifestLink = appDocument.createElement("link");
                manifestLink.rel = "manifest";
                appDocument.head.appendChild(manifestLink);
            }

            manifestLink.href = new URL(
                "/app/static/manifest.json",
                appWindow.location.origin
            ).href;
        }

        addManifest();
        appWindow.setTimeout(addManifest, 500);
        appWindow.setTimeout(addManifest, 1500);
        </script>
        """,
        height=0,
    )
