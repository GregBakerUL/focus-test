#%%
import importlib
import laplacian_variance_focus
importlib.reload(laplacian_variance_focus)
from laplacian_variance_focus import LaplacianVarianceGroupComparison



focus_group_comparison = LaplacianVarianceGroupComparison()


focus_group_comparison.add_image_dir(r"C:\Users\GregBaker\Box\Reliability (RG)\Ganymede\Images HALT on 50 Ganymedes\Ganymede HALT on 50 devices\22-12-2022 (Pre-test)", name="Pre-HALT")
focus_group_comparison.add_image_dir(r"C:\Users\GregBaker\Box\Reliability (RG)\Ganymede\Images HALT on 50 Ganymedes\Ganymede HALT on 50 devices\07-02-2023 (6th-last read point)", name="Post-HALT")

focus_group_comparison.plot_focus_scatter()
focus_group_comparison.plot_focus_kde()
focus_group_comparison.ouput_focus_scores("./focus_scores.csv")


# %%
