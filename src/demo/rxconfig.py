import reflex as rx

config = rx.Config(
    app_name="ui",
    telemetry_enabled=False,
    disable_plugins=[rx.plugins.sitemap.SitemapPlugin],
)
