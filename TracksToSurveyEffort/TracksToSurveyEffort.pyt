# -*- coding: utf-8 -*-

import arcpy

TARGET_AGOL_LAYER = r"https://services1.arcgis.com/KNdRU5cN6ENqCTjk/arcgis/rest/services/Survey_Effort_Tracking/FeatureServer/1"

ALLOWED_DISCIPLINES = {
    "",
    "Wetlands",
    "Uplands",
    "Zoology",
    "Other"
}


class Toolbox(object):

    def __init__(self):

        self.label = "Survey Effort Tools"
        self.alias = "surveytools"

        self.tools = [GPXToAGOLTool]


class GPXToAGOLTool(object):

    def __init__(self):

        self.label = "GPX Tracks to AGOL Survey Effort Layer"
        self.description = (
            "Convert GPX tracks to lines and append to AGOL feature service."
        )

        self.canRunInBackground = False

    def getParameterInfo(self):

        param0 = arcpy.Parameter(
            displayName="Input GPX (tracks only - download from OnX)",
            name="input_gpx",
            datatype="DEFile",
            parameterType="Required",
            direction="Input"
        )

        param0.filter.list = ["gpx"]

        param1 = arcpy.Parameter(
            displayName="Discipline",
            name="discipline",
            datatype="GPString",
            parameterType="Optional",
            direction="Input"
        )

        param1.filter.type = "ValueList"

        param1.filter.list = [
            "Wetlands",
            "Uplands",
            "Zoology",
            "Other"
        ]

        param2 = arcpy.Parameter(
            displayName="Checked By",
            name="checked_by",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )

        return [param0, param1, param2]

    def execute(self, parameters, messages):

        input_gpx = parameters[0].valueAsText
        discipline = parameters[1].valueAsText
        checked_by = parameters[2].valueAsText

        process_gpx_to_agol(
            input_gpx=input_gpx,
            discipline=discipline,
            checked_by=checked_by,
            target_layer=TARGET_AGOL_LAYER
        )


def _msg(text):

    arcpy.AddMessage(text)
    print(text)


def _err(text):

    arcpy.AddError(text)
    print(f"ERROR: {text}")


def _normalize_discipline(value):

    if value is None:
        return None

    v = str(value).strip()

    if v in ("", "<Null>"):
        return None

    if v not in ALLOWED_DISCIPLINES:

        raise ValueError(
            f"Invalid Discipline '{v}'. "
            f"Allowed values: Wetlands, Uplands, Zoology, Other."
        )

    return v


def _ensure_text_field(feature_class, field_name, length=255):

    field_names = [f.name for f in arcpy.ListFields(feature_class)]

    if field_name not in field_names:

        arcpy.management.AddField(
            feature_class,
            field_name,
            "TEXT",
            field_length=length
        )

        _msg(f"Added field '{field_name}'.")


def _parse_name_parts(name_value):
    """
    Expected format:
    Tracks mm/dd/yy SurveyName Notes

    Returns:
    (date, survey_name, notes)
    """

    if not name_value:
        return None, None, None

    parts = str(name_value).strip().split(" ", 3)

    if len(parts) < 3:
        return None, None, None

    date_part = parts[1]
    survey_name = parts[2]
    notes = parts[3] if len(parts) > 3 else ""

    return date_part, survey_name, notes


def _update_attributes(lines_fc, discipline, checked_by):

    _ensure_text_field(lines_fc, "Date", 25)
    _ensure_text_field(lines_fc, "SurveyName", 100)
    _ensure_text_field(lines_fc, "Notes", 500)
    _ensure_text_field(lines_fc, "Discipline", 50)
    _ensure_text_field(lines_fc, "CheckedBy", 100)

    fields = [
        "Name",
        "Date",
        "SurveyName",
        "Notes",
        "Discipline",
        "CheckedBy"
    ]

    with arcpy.da.UpdateCursor(lines_fc, fields) as cursor:

        for row in cursor:

            date_part, survey_name, notes = _parse_name_parts(row[0])

            if date_part:
                row[1] = date_part

            if survey_name:
                row[2] = survey_name

            row[3] = notes if notes is not None else ""
            row[4] = discipline
            row[5] = checked_by

            cursor.updateRow(row)

    _msg(
        "Updated Date/SurveyName/Notes/"
        "Discipline/CheckedBy fields."
    )


def process_gpx_to_agol(
    input_gpx,
    discipline,
    checked_by,
    target_layer
):

    if not input_gpx or not arcpy.Exists(input_gpx):

        raise FileNotFoundError(
            f"Input GPX does not exist: {input_gpx}"
        )

    if not checked_by or not str(checked_by).strip():

        raise ValueError("CheckedBy is required.")

    if not target_layer:

        raise ValueError(
            "TARGET_AGOL_LAYER is empty."
        )

    discipline_value = _normalize_discipline(discipline)

    checked_by_value = str(checked_by).strip()

    gpx_points = r"in_memory\gpx_pts"
    track_lines = r"in_memory\track_lines"

    # Remove existing in_memory layers if present
    for fc in [gpx_points, track_lines]:

        if arcpy.Exists(fc):
            arcpy.management.Delete(fc)

    _msg(f"Converting GPX to features: {input_gpx}")

    arcpy.conversion.GPXtoFeatures(
        input_gpx,
        gpx_points
    )

    _msg("Creating line features from GPX points.")

    arcpy.management.PointsToLine(
        Input_Features=gpx_points,
        Output_Feature_Class=track_lines,
        Line_Field="Name",
        Sort_Field="DateTimeS",
        Close_Line="NO_CLOSE"
    )

    _update_attributes(
        track_lines,
        discipline_value,
        checked_by_value
    )

    _msg(f"Appending to target layer: {target_layer}")

    arcpy.management.Append(
        inputs=[track_lines],
        target=target_layer,
        schema_type="NO_TEST"
    )

    _msg("Append complete.")