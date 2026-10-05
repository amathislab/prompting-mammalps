from dataclasses import fields
import numpy as np

from .data_enums import AnnotAttr, AnnotLevel
from .datastructures import (Detection, 
                            FrameAnnot, 
                            ObservationStore, 
                            VideoAnnot, 
                            VideoInfo,
                            )
                            
from .constants import ATTRIBUTE_SPECS

def add_obs(store: ObservationStore, attr: AnnotAttr, value, *, track_id: int = None, frame_id: int = None) -> None:
    spec = ATTRIBUTE_SPECS[attr]

    if spec.level == AnnotLevel.VIDEO:
        store.video[attr].append(value)

    elif spec.level == AnnotLevel.TRACK:
        store.track[track_id][attr].append(value)

    elif spec.level == AnnotLevel.FRAME:
        store.frame[frame_id].append(value)

def majority_vote(values: list[object]) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return vals[np.argmax(counts)]

def aggregate_video_attrs(store: ObservationStore, id2label: dict[str, dict[int, str]]) -> dict[str, str]:
    out = {}
    for attr, values in store.video.items():
        idx = majority_vote(values)
        spec = ATTRIBUTE_SPECS[attr]
        out[spec.json_name] = id2label[spec.label_map_key][idx]
    return out

def aggregate_tracks(store: ObservationStore, id2label: dict[str, dict[int, str]]) -> dict[int, dict[str, str]]:
    result = {}

    for track_id, attrs in store.track.items():
        final = {}

        for attr, values in attrs.items():
            idx = majority_vote(values)
            spec = ATTRIBUTE_SPECS[attr]
            final[spec.json_name] = id2label[spec.label_map_key].get(idx, "none")

        if (final[AnnotAttr.SPECIES.value] != "red_deer") and (final[AnnotAttr.SPECIES.value] != "roe_deer"):
            final.pop(AnnotAttr.DEER_AGE.value, None)
            final.pop(AnnotAttr.DEER_SEX.value, None)

        result[track_id] = final

    return result

def materialize_frames(store: ObservationStore, 
                        track_attrs: dict[int, dict[str, str]], 
                        id2label: dict[str, dict[int, str]], 
                        width: int, 
                        height: int,
                        ) -> list[FrameAnnot]:
    frames_out = []
    for frame_id in sorted(store._seen_frames):
        detections_out = []

        for det in store.frame[frame_id]:
            attrs = {}

            for attr, idx in det.attributes.items():
                spec = ATTRIBUTE_SPECS[attr]
                attrs[spec.json_name] = id2label[spec.label_map_key].get(idx, "none")

            attrs.update(track_attrs.get(det.track_id, {}))

            detections_out.append(
                Detection(
                    track_id=det.track_id,
                    bbox=det.bbox,
                    conf=det.conf,
                    attributes=attrs,
                )
            )

        frames_out.append(
            FrameAnnot(
                frame_id=frame_id,
                detections=detections_out,
                width=width,
                height=height,
            )
        )
            

    return frames_out

def make_empty_video_annot(meta: dict[str, object]) -> VideoAnnot:
    return VideoAnnot(
        info=VideoInfo(
            file_id=meta["file_id"],
            file_path=meta["file_path"],
            num_frames=int(meta["num_frames"]),
            duration_s=float(meta["duration_s"]),
            fps=float(meta["fps"]),
        ),
        frames=[],
    )

def from_dict(cls, data: dict):
    field_names = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)
    
