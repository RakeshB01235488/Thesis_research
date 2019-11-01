import skextremes as ske
import matplotlib.pyplot as plt
data = ske.datasets.name_of_the_dataset()
data_array = data.asarray()
target_vals = data.fields.sea_level
model = ske.models.classic.GEV(target_vals, fit_method = 'mle', ci = 0.05, ci_method = 'delta')





ppf at 0.95
19.781991125794754

