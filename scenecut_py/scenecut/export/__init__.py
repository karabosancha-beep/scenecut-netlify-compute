"""Export package — convert scene files to NLE interchange formats."""
from .fcpxml import export_fcpxml
from .edl import export_edl
from .premiere import export_premiere
from .csv_export import export_csv
from .srt import export_srt


def export(project: dict, format: str, output, source=None, reel_name="SCENE",
           fps=None, timecode_mode="ndf") -> str:
    """Dispatch to the right exporter."""
    if format == "fcpxml":
        return export_fcpxml(project, output, source=source, fps=fps)
    elif format == "edl":
        return export_edl(project, output, source=source, reel_name=reel_name, fps=fps)
    elif format == "premiere":
        return export_premiere(project, output, source=source, fps=fps)
    elif format == "csv":
        return export_csv(project, output)
    elif format == "srt":
        return export_srt(project, output)
    else:
        from ..util import SceneCutError
        raise SceneCutError(f"Unknown export format: {format}", code=5)
