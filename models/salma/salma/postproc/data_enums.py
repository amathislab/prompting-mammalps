from enum import StrEnum, auto

class AnnotLevel(StrEnum):
    FRAME = auto()
    TRACK = auto()
    VIDEO = auto()

class AnnotAttr(StrEnum):
    # FRAME
    ACTIVITY = "Activity"
    ACTION = "Action"
    ACTION2 = "Action2"

    # TRACK
    SPECIES = "Species"
    DEER_AGE = "Deer_age"
    DEER_SEX = "Deer_adult_sex"

    # VIDEO
    WEATHER = "weather"

class LabelMappingKey(StrEnum):
    # FRAME
    ACTIVITY = "activities"
    ACTIONS = "actions"

    # TRACK
    SPECIES = "species"
    DEER_AGE = "deer_ages"
    DEER_SEX = "deer_adult_sexes"

    # VIDEO
    WEATHER = "weather"
    