print("loading imports...")

import arcpy
import os

arcpy.env.overwriteOutput = True

output_gdb = r"C:\Users\chollenb\GIS\PyTools\ParcelOutputs\outputs.gdb"
output_fc = os.path.join(output_gdb, "tia_regrid_intersect")

print("calculating geometry...")
arcpy.management.AddField(output_fc, "area_acres_geo", "DOUBLE")

arcpy.management.CalculateField(
    output_fc,
    "area_acres_geo",
    "!shape.geodesicArea@ACRES!",
    "PYTHON3"
)

stats_table = os.path.join("in_memory", "owner_stats")
print("calculating statistics...")
arcpy.analysis.Statistics(
    output_fc,
    stats_table,
    [["area_acres_geo", "SUM"]],
    ["tia_id", "owner"]
)


arcpy.conversion.TableToExcel(stats_table, r"C:\Users\chollenb\GIS\PyTools\ParcelOutputs\parcel_filter_1a.xlsx")
arcpy.conversion.TableToExcel(output_fc, r"C:\Users\chollenb\GIS\PyTools\ParcelOutputs\parcel_filter_1b.xlsx")