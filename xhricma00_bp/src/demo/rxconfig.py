# demo/rxconfig.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.

import reflex as rx

config = rx.Config(
    app_name="ui",
    telemetry_enabled=False,
    disable_plugins=[rx.plugins.sitemap.SitemapPlugin],
)
