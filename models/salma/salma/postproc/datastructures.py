from dataclasses import dataclass, field, asdict
from .data_enums import AnnotLevel, AnnotAttr
from collections import defaultdict

@dataclass(frozen=True)
class AttributeSpec:
    json_name: str
    level: AnnotLevel
    head: str
    label_map_key: str

@dataclass
class Detection:
    track_id: int
    bbox: list[float]
    conf: float
    attributes: dict[AnnotAttr, int]

@dataclass
class FrameAnnot:
    frame_id: int
    detections: list[Detection] 
    width: int
    height: int

@dataclass
class VideoInfo:
    file_id: str
    file_path: str
    num_frames: int
    duration_s: float
    fps: float
    attributes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "VideoInfo":
        return cls(
            file_id=data.get("file_id", ""),
            file_path=data.get("file_path", ""),
            num_frames=int(data.get("num_frames", "0")),
            duration_s=float(data.get("duration_s", "-1")),
            fps=float(data.get("fps", "0")),
            attributes=data.get("attributes", {}),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def extract_SCEV(self) -> tuple[str, str, str, str]:
        S, C, E, V = self.file_id.split("_")
        return (S, C, E, V)

@dataclass
class VideoAnnot:
    info: VideoInfo
    frames: list[FrameAnnot] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "VideoAnnot":
        info = VideoInfo.from_dict(data["info"])
        frames = [
            FrameAnnot(
                frame_id=f["frame_id"],
                detections=[
                    Detection(
                        track_id = d["track_id"],
                        bbox = d["bbox"],
                        conf = d.get("conf", 1.0),
                        attributes = d["attributes"],
                    )
                    for d in f["detections"]
                ],
                width=f.get("width", 1920),
                height=f.get("height", 1080),
            )
            for f in data["frames"]
        ]

        return cls(info=info, frames=frames)

    def to_dict(self) -> dict:
        return {
            "info": self.info.to_dict(),
            "frames": [
                {
                    "frame_id": f.frame_id,
                    "width": f.width,
                    "height": f.height,
                    "detections": [asdict(d) for d in f.detections],
                }
                for f in self.frames
            ],
        }

    def compute_track_statistics(self):
        track_conf_sum: dict[int, float] = defaultdict(float)
        track_frame_ids: dict[int, set[int]] = defaultdict(set)

        for frame in self.frames:
            for det in frame.detections:
                track_conf_sum[det.track_id] += det.conf
                track_frame_ids[det.track_id].add(frame.frame_id)

        return track_conf_sum, track_frame_ids

class ObservationStore:
    def __init__(self):
        self.frame = defaultdict(list)
        self.track = defaultdict(lambda: defaultdict(list))
        self.video = defaultdict(list)
        self._seen_frames = set()

    def register_frame(self, frame_id: int):
        self._seen_frames.add(frame_id)
        _ = self.frame[frame_id]  # ensure key exists
