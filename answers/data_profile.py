from ydata_profiling import ProfileReport
import xarray as xr



file_path = r'C:\Users\Ankit\Documents\Vedanshi\IISERB-ML-DL-course\PERCDR_0.25deg_2001_2010_precipitation_data.nc'
ds = xr.open_dataset(file_path)
df = ds.to_dataframe().reset_index()

title = "Data Profile"
profile = ProfileReport(df, title=title)
profile.to_file("data_profile.html")
print("Data Profile generated successfully")