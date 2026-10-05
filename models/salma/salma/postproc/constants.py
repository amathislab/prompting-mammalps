from .data_enums import AnnotAttr, AnnotLevel, LabelMappingKey
from .datastructures import AttributeSpec
from salma.model.salma.head_enum import SalmaHead

ATTRIBUTE_SPECS = {
    AnnotAttr.ACTIVITY: AttributeSpec(json_name=AnnotAttr.ACTIVITY, level=AnnotLevel.FRAME, head=SalmaHead.ACTIVITY, label_map_key=LabelMappingKey.ACTIVITY),
    AnnotAttr.ACTION:   AttributeSpec(json_name=AnnotAttr.ACTION,   level=AnnotLevel.FRAME, head=SalmaHead.ACTIONS,  label_map_key=LabelMappingKey.ACTIONS),
    AnnotAttr.ACTION2:  AttributeSpec(json_name=AnnotAttr.ACTION2,  level=AnnotLevel.FRAME, head=SalmaHead.ACTIONS,  label_map_key=LabelMappingKey.ACTIONS),

    AnnotAttr.SPECIES:  AttributeSpec(json_name=AnnotAttr.SPECIES,  level=AnnotLevel.TRACK, head=SalmaHead.SPECIES,  label_map_key=LabelMappingKey.SPECIES),
    AnnotAttr.DEER_AGE: AttributeSpec(json_name=AnnotAttr.DEER_AGE, level=AnnotLevel.TRACK, head=SalmaHead.DEER_AGE, label_map_key=LabelMappingKey.DEER_AGE),
    AnnotAttr.DEER_SEX: AttributeSpec(json_name=AnnotAttr.DEER_SEX, level=AnnotLevel.TRACK, head=SalmaHead.DEER_SEX, label_map_key=LabelMappingKey.DEER_SEX),

    AnnotAttr.WEATHER:  AttributeSpec(json_name=AnnotAttr.WEATHER,  level=AnnotLevel.VIDEO, head=SalmaHead.WEATHER,  label_map_key=LabelMappingKey.WEATHER),
}
