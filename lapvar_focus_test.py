#%%
import importlib
import laplacian_variance_focus
importlib.reload(laplacian_variance_focus)
from laplacian_variance_focus import LaplacianVarianceGroupComparison



focus_group_comparison = LaplacianVarianceGroupComparison()


focus_group_comparison.add_image_dir("C:/Users/GregBaker/Box/Tracking Cameras Team/Test Data/Ganymede Beta Development/Focus and Brightness Tests/2023_01_10_gredmann_mtf_analysis_2.1/55_40/isp_off", name="55_40")
focus_group_comparison.add_image_dir("C:/Users/GregBaker/Box/Tracking Cameras Team/Test Data/Ganymede Beta Development/Focus and Brightness Tests/2023_01_10_gredmann_mtf_analysis_2.1/55_35/isp_off", name="55_35")
focus_group_comparison.add_image_dir("C:/Users/GregBaker/Box/Tracking Cameras Team/Test Data/Ganymede Beta Development/Focus and Brightness Tests/2023_01_10_gredmann_mtf_analysis_2.1/50_40/isp_off", name="50_40")

focus_group_comparison.plot_focus_scatter()
focus_group_comparison.plot_focus_kde()
focus_group_comparison.print_focus_stats()


# %%
