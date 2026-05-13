print("loading imports...")
import arcpy
import os
from datetime import datetime

arcpy.env.overwriteOutput = True

print("initializing...")

county_list = ["Teller", "Mesa", "Montrose", "Gunnison", "Garfield", "Delta", "Chaffee", 
               "Lake", "Pitkin", "Grand", "Jackson", "Larimer", "Park", "Pueblo", 
               "El Paso", "Fremont", "Huerfano", "Las Animas", "Custer"]

regrid_path = r"N:\Research\CNHP\GIS_Data\Colorado_Protected_Data\Regrid_Parcels"

tia_path = "https://services1.arcgis.com/KNdRU5cN6ENqCTjk/arcgis/rest/services/CNHP_TIAs_Master_2026/FeatureServer/0"

# --- Step 1: Make and filter TIA layer ---
tia_layer = arcpy.MakeFeatureLayer_management(tia_path, "tia_layer")

# Build SQL filter
#county_sql = " OR ".join([f"CountyName LIKE '%{c}%'" for c in county_list])
sql = f"BiologicalPriority IN ('1', '2')"
arcpy.SelectLayerByAttribute_management(tia_layer, "NEW_SELECTION", sql)

# --- Step 2: Copy filtered TIA locally (important for performance/reliability) ---
tia_local = r"in_memory\tia_local"
arcpy.CopyFeatures_management(tia_layer, tia_local)

# --- Step 3: Prepare output ---
output_gdb = r"C:\Users\chollenb\GIS\PyTools\ParcelOutputs\outputs.gdb"
if not arcpy.Exists(output_gdb):
    gdb_folder = os.path.dirname(output_gdb)
    gdb_name = os.path.basename(output_gdb)
    arcpy.management.CreateFileGDB(gdb_folder, gdb_name)

output_fc = os.path.join(output_gdb, "tia_regrid_intersect")
# Delete output feature class if it exists
if arcpy.Exists(output_fc):
    arcpy.management.Delete(output_fc)

# Initialize output as None
initialized = False

# --- Step 4: Loop through counties ---
for county in county_list:
    county_path = "co_" + county.lower()
    county_path = county_path.replace(" ", "_")

    parcel_path = rf"{regrid_path}\{county_path}.gdb\{county_path}"

    if county=="Teller":
        parcel_path = rf"C:\Users\chollenb\GIS\PyTools\ParcelsCopy\co_teller.gdb\{county_path}"
    print(f"Processing {county}...")

    # Optional: subset TIA to this county only (faster intersects)
    county_tia = arcpy.MakeFeatureLayer_management(
        tia_local, 
        "county_tia", 
        f"CountyName LIKE '%{county}%'"
    )

    intersect_temp = r"in_memory\intersect_temp"
    parcel_clean = r"in_memory\parcel_clean"

    if arcpy.Exists(parcel_clean):
        arcpy.management.Delete(parcel_clean)
    arcpy.management.CopyFeatures(parcel_path, parcel_clean)
    arcpy.management.RepairGeometry(parcel_clean)

    arcpy.analysis.Intersect(
        [county_tia, parcel_clean],
        intersect_temp,
        join_attributes="ALL"
    )

    # Initialize or append
    if not initialized:
        arcpy.CopyFeatures_management(intersect_temp, output_fc)
        initialized = True
    else:
        arcpy.Append_management(intersect_temp, output_fc, "NO_TEST")


print("Finished processing all counties.")



# for each tia_id in tia_regrid_intersect, dissolve parcels by "owner" field. There will be multiple polygons with same tia_id

# calculate acreage for each owner within TIA = ownerTIA_acres
# divide owner acreage by TIA "Acres" field * 100 = ownerTIA_percent

# filter to ownerTIA_percent > 10 OR ownerTIA_acres > 600