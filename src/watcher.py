"""
watcher.py — Surveillance en temps réel du dossier data/.

Dès qu'un fichier est ajouté, modifié ou supprimé, l'index est automatiquement
resynchronisé via src.ingest.sync().

Un "debounce" (délai d'attente) évite de déclencher l'indexation avant que le
fichier ait fini d'être complètement écrit sur le disque (copie longue,
écriture progressive).
"""

from __future__ import annotations

import threading
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src.config import DATA_DIR, SUPPORTED_EXTENSIONS, WATCH_DEBOUNCE_SECONDS


class _DebouncedSyncHandler(FileSystemEventHandler):
    def __init__(self) -> None:
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def _is_relevant(self, path: str) -> bool:
        return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS

    def _schedule_sync(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(WATCH_DEBOUNCE_SECONDS, self._run_sync)
            self._timer.daemon = True
            self._timer.start()

    def _run_sync(self) -> None:
        from src.ingest import sync  # import local pour éviter les imports circulaires

        try:
            sync()
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  Erreur pendant la synchronisation automatique : {e}")

    def on_created(self, event):
        if not event.is_directory and self._is_relevant(event.src_path):
            self._schedule_sync()

    def on_modified(self, event):
        if not event.is_directory and self._is_relevant(event.src_path):
            self._schedule_sync()

    def on_deleted(self, event):
        if not event.is_directory and self._is_relevant(event.src_path):
            self._schedule_sync()

    def on_moved(self, event):
        if not event.is_directory:
            self._schedule_sync()


def start_watcher() -> Observer:
    """Démarre la surveillance de data/ dans un thread en arrière-plan et
    retourne l'Observer (appelle .stop() puis .join() pour l'arrêter proprement)."""
    handler = _DebouncedSyncHandler()
    observer = Observer()
    observer.schedule(handler, str(DATA_DIR), recursive=True)
    observer.start()
    return observer