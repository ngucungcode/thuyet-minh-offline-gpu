"""Network-bound acquisition adapters.

Only this package and the API layer are allowed to communicate with indexers,
download clients, and subtitle services.  Inference workers do not import it.
"""

from .prowlarr import ProwlarrIndexerGateway
from .qbittorrent import QBittorrentDownloadClient
from .service import AcquisitionService
from .factory import build_acquisition_service
from .subtitles import CompositeSubtitleProvider, FfprobeSubtitleProbe

__all__ = [
    "AcquisitionService",
    "build_acquisition_service",
    "CompositeSubtitleProvider",
    "FfprobeSubtitleProbe",
    "ProwlarrIndexerGateway",
    "QBittorrentDownloadClient",
]
