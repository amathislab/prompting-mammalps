from enum import StrEnum

class SalmaHead(StrEnum):
    SPECIES = "pred_species"
    ACTIVITY = "pred_activities"
    ACTIONS = "pred_actions"
    DEER_AGE = "pred_dages"
    DEER_SEX = "pred_dsexes"
    BOXES = "pred_boxes"
    WEATHER = "pred_weather"
    IS_ANIMAL = "pred_is_animal"
    QUERIES = "animal_queries"
    