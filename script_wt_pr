import streamlit as st
import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize
import whitebox
import numpy as np
from shapely.geometry import Point
import pandas as pd
from pyproj import CRS
import tempfile
import zipfile
import matplotlib.pyplot as plt
import io
import shutil  # For cleaning up the temp directory

# Initialize WhiteboxTools
wbt = whitebox.WhiteboxTools()

# Function to set WhiteboxTools working directory
def set_working_directory(output_dir):
    wbt.work_dir = output_dir

# Function to calculate watershed area in square kilometers
def calculate_watershed_area(watershed_proj):
    watershed_proj['area_m2'] = watershed_proj['geometry'].area
    total_area = watershed_proj['area_m2'].sum()  # in square meters
    total_area_km2 = total_area / 1e6  # Convert to square kilometers
    return total_area_km2

# Function to check if the watershed and raster bounds overlap
def bounds_overlap(bounds1, bounds2):
    return not (
        bounds1[2] < bounds2[0] or  # maxx1 < minx2
        bounds1[0] > bounds2[2] or  # minx1 > maxx2
        bounds1[3] < bounds2[1] or  # maxy1 < miny2
        bounds1[1] > bounds2[3]     # miny1 > maxy2
    )

# Function to clip DEM to watershed boundary
def clip_dem_to_watershed(dem_file, watershed, output_cropped_dem):
    with rasterio.open(dem_file) as src:
        out_image, out_transform = mask(src, watershed.geometry, crop=True)
        out_meta = src.meta.copy()

    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "crs": src.crs
    })

    with rasterio.open(output_cropped_dem, "w", **out_meta) as dest:
        dest.write(out_image)

# Function to process DEM (breach depressions, flow direction, flow accumulation)
def process_dem(output_cropped_dem, wbt_work_dir, shapefile_base):
    breached_dem = os.path.join(wbt_work_dir, f'{shapefile_base}_breached_dem.tif')
    wbt.breach_depressions(dem=output_cropped_dem, output=breached_dem)

    fdir = os.path.join(wbt.work_dir, f'{shapefile_base}_flow_direction.tif')
    wbt.d8_pointer(dem=breached_dem, output=fdir)

    fac = os.path.join(wbt.work_dir, f'{shapefile_base}_flow_accumulation.tif')
    wbt.d8_flow_accumulation(breached_dem, output=fac, out_type='cells')

    return breached_dem, fdir, fac

# Function to rasterize watershed polygon
def rasterize_watershed(watershed, dem_shape, dem_transform, dem_crs, wbt_work_dir, shapefile_base):
    watershed = watershed.to_crs(dem_crs)
    basins_raster = os.path.join(wbt_work_dir, f'{shapefile_base}_basins.tif')
    geometry = [(geom, 1) for geom in watershed.geometry]

    basin_array = rasterize(
        geometry,
        out_shape=dem_shape,
        transform=dem_transform,
        fill=0,  # Background value
        dtype='int32'
    )

    with rasterio.open(
        basins_raster,
        'w',
        driver='GTiff',
        height=dem_shape[0],
        width=dem_shape[1],
        count=1,
        dtype='int32',
        crs=dem_crs,
        transform=dem_transform,
    ) as dst:
        dst.write(basin_array, 1)

    return basins_raster

# Function to zip shapefile components
def zip_shapefile(shapefile_path):
    shapefile_base = os.path.splitext(shapefile_path)[0]
    extensions = [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".fix", ".sbn", ".sbx", ".ain", ".aih", ".atx", ".ixs", ".mxs"]
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for ext in extensions:
            file_path = shapefile_base + ext
            if os.path.exists(file_path):
                zipf.write(file_path, arcname=os.path.basename(file_path))
    zip_buffer.seek(0)
    return zip_buffer

# Function to calculate the longest flow path
def calculate_longest_flow_path(breached_dem, basins_raster, wbt_work_dir, shapefile_base):
    longest_flowpath_vector = os.path.join(wbt.work_dir, f'{shapefile_base}_longest_flowpath.shp')
    wbt.longest_flowpath(dem=breached_dem, basins=basins_raster, output=longest_flowpath_vector)
    return longest_flowpath_vector

# Function to calculate the length of the longest flow path and save only the longest line
def calculate_flowpath_length(longest_flowpath_vector, dem_crs, projected_crs):
    # Read the shapefile into a GeoDataFrame
    longest_flowpath_gdf = gpd.read_file(longest_flowpath_vector)

    # Set the CRS if it's not defined
    if longest_flowpath_gdf.crs is None:
        longest_flowpath_gdf.crs = dem_crs

    # Reproject to the projected CRS
    longest_flowpath_gdf_proj = longest_flowpath_gdf.to_crs(projected_crs)

    # Calculate the length of each geometry
    longest_flowpath_gdf_proj['length_m'] = longest_flowpath_gdf_proj.geometry.length

    # Find the index of the longest geometry
    idx_longest = longest_flowpath_gdf_proj['length_m'].idxmax()

    # Extract only the longest geometry
    longest_line_gdf = longest_flowpath_gdf_proj.loc[[idx_longest]].copy()

    # Reset index to ensure the shapefile will contain only one line
    longest_line_gdf.reset_index(drop=True, inplace=True)

    # Now overwrite the shapefile with only the longest line (deletes all other geometries)
    longest_line_gdf.to_file(longest_flowpath_vector, driver='ESRI Shapefile')

    # Return the length of the longest flow path and the GeoDataFrame containing only the longest line
    return longest_line_gdf['length_m'].values[0], longest_line_gdf

# Function to divide longest flow path into segments and extract elevation
def extract_elevation_points(longest_line, breached_dem, projected_crs, dem_crs):
    segment_points = [longest_line.interpolate(i / 10, normalized=True) for i in range(11)]

    with rasterio.open(breached_dem) as src:
        if projected_crs != dem_crs:
            segment_points = [gpd.GeoSeries([Point(p.x, p.y)], crs=projected_crs).to_crs(dem_crs).iloc[0] for p in segment_points]

        elevations = []
        for point in segment_points:
            try:
                elevation = list(src.sample([(point.x, point.y)]))[0][0]  # Extract elevation
            except IndexError:
                elevation = np.nan  # Handle points outside the DEM
            elevations.append(elevation)

    points_df = pd.DataFrame({
        'Point': [f'Point {i+1}' for i in range(11)],
        'Longitude': [point.x for point in segment_points],
        'Latitude': [point.y for point in segment_points],
        'Elevation (m)': elevations
    })

    return points_df

# Function to calculate UTM zone based on longitude
def get_utm_zone(lon):
    """Calculate UTM zone based on longitude."""
    if lon < -180 or lon > 180:
        raise ValueError(f"Invalid longitude value: {lon}. Longitude must be between -180 and 180.")
    return int((lon + 180) / 6) + 1

# Function to determine UTM zone and EPSG code based on the centroid
def calculate_utm_zone(watershed):
    watershed_wgs84 = watershed.to_crs(epsg=4326)
    centroid = watershed_wgs84.geometry.centroid.iloc[0]
    centroid_lon = centroid.x
    centroid_lat = centroid.y

    utm_zone = get_utm_zone(centroid_lon)

    if centroid_lat >= 0:
        hemisphere = 'north'
        epsg_code = 32600 + utm_zone
    else:
        hemisphere = 'south'
        epsg_code = 32700 + utm_zone

    target_crs = f'EPSG:{epsg_code}'
    st.write(f"Using UTM Zone {utm_zone} {hemisphere.capitalize()} (EPSG:{epsg_code}) as target CRS.")

    return target_crs

# Function to zip shapefile components
def zip_shapefile(shapefile_path):
    shapefile_base = os.path.splitext(shapefile_path)[0]
    extensions = [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".fix", ".sbn", ".sbx", ".ain", ".aih", ".atx", ".ixs", ".mxs"]
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for ext in extensions:
            file_path = shapefile_base + ext
            if os.path.exists(file_path):
                zipf.write(file_path, arcname=os.path.basename(file_path))
    zip_buffer.seek(0)
    return zip_buffer

# Function to clean up temporary directory (optional)
def cleanup_tempdir(tempdir):
    if os.path.exists(tempdir):
        shutil.rmtree(tempdir)
        


# Streamlit App
st.title("Watershed Analysis App")

st.write("Upload a zipped shapefile of your watershed, and this app will perform watershed analysis.")

# Input for Curve Number
curve_number = st.slider("Enter Curve Number (0-100):", min_value=0, max_value=100, value=50)

# Reset Button (optional)
if st.button("Reset Analysis"):
    if 'tempdir' in st.session_state:
        cleanup_tempdir(st.session_state['tempdir'])
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

uploaded_zip = st.file_uploader("Upload a zipped shapefile", type=["zip"])

if uploaded_zip is not None:
    # Check if a new upload has occurred to reset previous session state
    if 'uploaded_zip' not in st.session_state or st.session_state['uploaded_zip'] != uploaded_zip.name:
        # If a new file is uploaded, reset the session state
        if 'tempdir' in st.session_state:
            cleanup_tempdir(st.session_state['tempdir'])
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state['uploaded_zip'] = uploaded_zip.name

    if 'tempdir' not in st.session_state:
        # Create a persistent temporary directory for the session
        st.session_state['tempdir'] = tempfile.mkdtemp()

    tempdir = st.session_state['tempdir']

    # Save the uploaded zip file to the temp directory
    zip_path = os.path.join(tempdir, "uploaded_shapefile.zip")
    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.getbuffer())

    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(tempdir)

    # Find the shapefile (.shp)
    shapefile_path = None
    for file in os.listdir(tempdir):
        if file.endswith(".shp"):
            shapefile_path = os.path.join(tempdir, file)
            break

    if shapefile_path is None:
        st.error("No .shp file found in the uploaded zip file.")
    else:
        # Proceed with the analysis
        # Read the shapefile
        watershed = gpd.read_file(shapefile_path)

        if watershed.crs is None:
            st.error("The shapefile does not have a CRS defined.")
        else:
            # Determine the projected CRS
            crs = CRS(watershed.crs)

            if crs.is_projected:
                projected_crs = watershed.crs
                st.write(f"Shapefile is already in a projected CRS: {projected_crs}")
            else:
                projected_crs = calculate_utm_zone(watershed)

            # Set the working directory for WhiteboxTools
            set_working_directory(tempdir)

            # Extract the base name of the shapefile without extension, replacing spaces with underscores
            shapefile_base = os.path.splitext(os.path.basename(shapefile_path))[0].replace(" ", "_")

            # Set output paths based on tempdir and base name
            output_dir = tempdir
            output_cropped_dem = os.path.join(output_dir, f'{shapefile_base}_cropped_dem.tif')

            # Project the watershed shapefile
            watershed_proj = watershed.to_crs(projected_crs)

            # Calculate the watershed area
            total_area_km2 = calculate_watershed_area(watershed_proj)
            # st.write(f'Total watershed area: {total_area_km2:.2f} square kilometers')

            # Use the DEM file
            dem_file = r'compressed_dem_turkey_0_0032degree_from_srtm30.tif'
            # dem_file = r'g:\My Drive\ONE_DRIVE_METU\Data\arcfiles\Turkiye_Dem_Srtm_30m_0_001_deg\dem_srtm_0_001_deg.tif'

            if not os.path.exists(dem_file):
                st.error(f"DEM file not found at {dem_file}")
            else:
                # Open the DEM and get its CRS
                with rasterio.open(dem_file) as src:
                    raster_crs = src.crs
                    raster_bounds = src.bounds

                # Reproject the watershed to the DEM CRS
                watershed_dem_crs = watershed_proj.to_crs(raster_crs)

                # Check if the bounds overlap
                watershed_bounds = watershed_dem_crs.total_bounds
                if not bounds_overlap(raster_bounds, watershed_bounds):
                    st.error("The extents of the raster and shapefile do not overlap.")
                else:
                    # Initialize session state to store processing results
                    if 'processing_done' not in st.session_state:
                        st.session_state['processing_done'] = False

                    if not st.session_state['processing_done']:
                        # Clip the DEM to the watershed boundary
                        clip_dem_to_watershed(dem_file, watershed_dem_crs, output_cropped_dem)

                        # Process the DEM (breach depressions, flow direction, flow accumulation)
                        breached_dem, fdir, fac = process_dem(output_cropped_dem, wbt.work_dir, shapefile_base)

                        # Rasterize watershed polygon
                        with rasterio.open(breached_dem) as src:
                            dem_shape = src.shape
                            dem_transform = src.transform
                            dem_crs = src.crs
                        basins_raster = rasterize_watershed(watershed_dem_crs, dem_shape, dem_transform, dem_crs, wbt.work_dir, shapefile_base)

                        # Calculate the longest flow path
                        longest_flowpath_vector = calculate_longest_flow_path(breached_dem, basins_raster, wbt.work_dir, shapefile_base)

                        # Calculate flow path length and overwrite the shapefile with only the longest line
                        longest_flowpath_length, longest_flowpath_gdf_proj = calculate_flowpath_length(longest_flowpath_vector, dem_crs, projected_crs)

                        # Extract elevation points along the longest flow path
                        longest_line = longest_flowpath_gdf_proj.geometry.iloc[0]
                        points_df = extract_elevation_points(longest_line, breached_dem, projected_crs, dem_crs)

                        # Calculate and store general watershed information
                        general_info_df = pd.DataFrame({
                            'Longest Flow Path (m)': [longest_flowpath_length],
                            'Curve Number': [curve_number],
                            'Watershed Area (km²)': [total_area_km2]
                        })


                        # Save the output Excel file with separate sheets
                        output_excel_file = os.path.join(output_dir, f'{shapefile_base}_watershed_analysis_output.xlsx')
                        with pd.ExcelWriter(output_excel_file) as writer:
                            points_df.to_excel(writer, sheet_name='Elevation Points', index=False)
                            general_info_df.to_excel(writer, sheet_name='General Info', index=False)

                        # Read the Elevation Points sheet
                        tc_calculation = pd.read_excel(output_excel_file, sheet_name='Elevation Points')

                        # Add 'Curve_number' to the first row
                        tc_calculation.loc[0, "Curve_number"] = curve_number

                        # Sort by 'Elevation (m)'
                        tc_calculation = tc_calculation.sort_values(by="Elevation (m)", ascending=True).reset_index(drop=True)

                        # Perform calculations
                        for jn in range(len(tc_calculation["Elevation (m)"])):
                            if jn == 0:
                                tc_calculation.loc[jn, "h(m)"] = 0
                                tc_calculation.loc[jn, "(1/h)^0.5"] = 0
                            else:
                                elevation_diff = tc_calculation.loc[jn, "Elevation (m)"] - tc_calculation.loc[jn-1, "Elevation (m)"]
                                tc_calculation.loc[jn, "h(m)"] = elevation_diff
                                if elevation_diff != 0:
                                    tc_calculation.loc[jn, "(1/h)^0.5"] = (general_info_df.loc[0, "Longest Flow Path (m)"] / 10 / elevation_diff) ** 0.5
                                else:
                                    tc_calculation.loc[jn, "(1/h)^0.5"] = 0

                        # Handle division by zero or invalid values
                        tc_calculation["(1/h)^0.5"] = tc_calculation["(1/h)^0.5"].replace([np.inf, -np.inf], 0).fillna(0)

                        # Perform sum and other calculations
                        tc_calculation.loc[0, "Area (km²)"] = general_info_df.loc[0, "Watershed Area (km²)"]
                        tc_calculation.loc[0, "sum_(1/h)^0.5"] = tc_calculation["(1/h)^0.5"].sum()
                        tc_calculation.loc[0, "S"] = (10 / tc_calculation.loc[0, "sum_(1/h)^0.5"]) ** 2
                        tc_calculation.loc[0, "Tc_hours"] = 0.00032 * (general_info_df.loc[0, "Longest Flow Path (m)"] ** 0.77) / (tc_calculation.loc[0, "S"] ** 0.385)
                        
                        # Extract values from general_info_df
                        area_km2 = general_info_df.loc[0, "Watershed Area (km²)"]
                        longest_flow_path = general_info_df.loc[0, "Longest Flow Path (m)"]
                        curve_number_value = general_info_df.loc[0, "Curve Number"]

                        # Extract Tc_hours from tc_calculation
                        tc_hours = tc_calculation.loc[0, "Tc_hours"]

                        # Create a summary DataFrame
                        tc_calculation_2 = pd.DataFrame({
                            "Area (km²)": [area_km2],
                            "Longest Flow Path (m)": [longest_flow_path],
                            "Curve_number": [curve_number_value],
                            "Tc_hours": [tc_hours]
                        })
                        st.write(tc_calculation_2)
                        
                        
                        
                        # Create the map visualization after elevation profile plot
                        st.write("Creating map visualization...")
                        try:
                            # Set figure size
                            fig2, ax2 = plt.subplots(figsize=(15, 12))
                            
                            # Read and plot the cropped DEM
                            with rasterio.open(output_cropped_dem) as src:
                                dem_data = src.read(1)
                                # Get the GCS bounds from the cropped DEM
                                bounds = src.bounds
                                transform = src.transform
                                
                                dem_min = 0  # Minimum elevation value
                                dem_max = 6000  # Maximum elevation value from the cropped DEM
                                
                                # Plot DEM with correct extent
                                im = ax2.imshow(dem_data, cmap='terrain', 
                                               extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                                               aspect='equal', vmin=dem_min, vmax=dem_max)
                                
                                cbar = plt.colorbar(im, ax=ax2, label='Elevation (m)', pad=0.02)
                                cbar.ax.tick_params(labelsize=10)
                                
                                # Convert longest flowpath to GCS if it's not already
                                longest_flowpath_gdf = gpd.read_file(longest_flowpath_vector)
                                if longest_flowpath_gdf.crs != src.crs:
                                    longest_flowpath_gdf = longest_flowpath_gdf.to_crs(src.crs)
                                longest_flowpath_gdf.plot(ax=ax2, color='red', linewidth=3,
                                                        label='Longest Flow Path')
                                
                                # Convert watershed boundary to GCS if it's not already
                                if watershed_dem_crs.crs != src.crs:
                                    watershed_dem_crs = watershed_dem_crs.to_crs(src.crs)
                                watershed_dem_crs.boundary.plot(ax=ax2, color='black', linewidth=2,
                                                              label='Watershed Boundary')
                                
                                # Set the plot extent to match the cropped DEM
                                ax2.set_xlim([bounds.left, bounds.right])
                                ax2.set_ylim([bounds.bottom, bounds.top])
                                
                                # Adjust title and labels
                                ax2.set_title('Watershed DEM with Longest Flow Path', fontsize=14, pad=20)
                                ax2.legend(fontsize=12)
                                ax2.set_xlabel('Longitude', fontsize=12)
                                ax2.set_ylabel('Latitude', fontsize=12)
                                ax2.tick_params(labelsize=10)
                                
                                plt.tight_layout()
                                
                                st.pyplot(fig2)
                                st.write("Map visualization completed")
                                
                        except Exception as e:
                            st.error(f"Error occurred: {str(e)}")
                            st.write("Error details:", traceback.format_exc())  # This will print the full error traceback
                            
                        # Plot the elevation profile
                        # fig, ax = plt.subplots()
                        # distances = np.linspace(0, longest_flowpath_length, len(points_df))
                        # ax.plot(distances, points_df['Elevation (m)'])
                        # ax.set_xlabel('Distance along flow path (m)')
                        # ax.set_ylabel('Elevation (m)')
                        # ax.set_title('Elevation Profile along Longest Flow Path')
                        # st.pyplot(fig)

                        # Store all necessary paths and data in session state
                        st.session_state['processing_done'] = True
                        st.session_state['output_cropped_dem'] = output_cropped_dem
                        st.session_state['breached_dem'] = breached_dem
                        st.session_state['fdir'] = fdir
                        st.session_state['fac'] = fac
                        st.session_state['basins_raster'] = basins_raster
                        st.session_state['longest_flowpath_vector'] = longest_flowpath_vector
                        st.session_state['longest_flowpath_length'] = longest_flowpath_length
                        st.session_state['tc_calculation_2'] = tc_calculation_2
                        st.session_state['tc_calculation'] = tc_calculation
                        st.session_state['output_excel_file'] = output_excel_file
                        st.session_state['points_df'] = points_df
                        st.session_state['general_info_df'] = general_info_df
                        st.session_state['shapefile_base'] = shapefile_base

                    # Now, whether processing was just done or already done, provide download buttons
                    if st.session_state.get('processing_done', False):
                        shapefile_base = st.session_state['shapefile_base']
                        output_dir = tempdir

                        # Download Cropped DEM
                        output_cropped_dem = st.session_state['output_cropped_dem']
                        if os.path.exists(output_cropped_dem):
                            with open(output_cropped_dem, "rb") as f:
                                st.download_button(
                                    label="Download Cropped DEM",
                                    data=f,
                                    file_name=f"{shapefile_base}_cropped_dem.tif"
                                )
                        else:
                            st.warning("Cropped DEM file is not available for download.")

                        # Download Breached DEM
                        breached_dem = st.session_state['breached_dem']
                        if os.path.exists(breached_dem):
                            with open(breached_dem, "rb") as f:
                                st.download_button(
                                    label="Download Breached DEM",
                                    data=f,
                                    file_name=f"{shapefile_base}_breached_dem.tif"
                                )
                        else:
                            st.warning("Breached DEM file is not available for download.")

                        # Download Flow Direction
                        fdir = st.session_state['fdir']
                        if os.path.exists(fdir):
                            with open(fdir, "rb") as f:
                                st.download_button(
                                    label="Download Flow Direction",
                                    data=f,
                                    file_name=f"{shapefile_base}_flow_direction.tif"
                                )
                        else:
                            st.warning("Flow Direction file is not available for download.")

                        # Download Flow Accumulation
                        fac = st.session_state['fac']
                        if os.path.exists(fac):
                            with open(fac, "rb") as f:
                                st.download_button(
                                    label="Download Flow Accumulation",
                                    data=f,
                                    file_name=f"{shapefile_base}_flow_accumulation.tif"
                                )
                        else:
                            st.warning("Flow Accumulation file is not available for download.")

                        # Download Longest Flow Path Shapefile
                        longest_flowpath_vector = st.session_state['longest_flowpath_vector']
                        if os.path.exists(longest_flowpath_vector):
                            zip_buffer = zip_shapefile(longest_flowpath_vector)
                            st.download_button(
                                label="Download Longest Flow Path Shapefile (ZIP)",
                                data=zip_buffer,
                                file_name=f"{shapefile_base}_longest_flowpath.zip",
                                mime="application/zip"
                            )
                        else:
                            st.warning("Longest Flow Path Shapefile is not available for download.")

                        # Download Output Excel File
                        output_excel_file = st.session_state['output_excel_file']
                        if os.path.exists(output_excel_file):
                            with open(output_excel_file, "rb") as f:
                                excel_data = f.read()
                            st.download_button(
                                label="Download Output Excel File",
                                data=excel_data,
                                file_name=f'{shapefile_base}_watershed_analysis_output.xlsx',
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        else:
                            st.warning("Output Excel file is not available for download.")

                        st.success("Analysis complete.")
