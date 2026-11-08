"""Wrap existing driver fakes in the production Standard storage boundary."""
from kb_pipeline.storage.standard import StandardStorage


def storage_runtime(settings, search_settings, client):
    return settings, search_settings, StandardStorage(settings, vector_client=client if client is not None else object())
