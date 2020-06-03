import numpy as np

from scripts.utils.config_utils import config
from scripts.utils.data_utils import Data

import matplotlib.pyplot as plt

data = Data(config.animals_add4)

# train_embed_x = data.embed_X[np.array(data.train_idx)]
# train_y = data.y[np.array(data.train_idx)]
# color_map = plt.get_cmap("tab10")(train_y.astype(int))
# plt.scatter(train_embed_x[:,0], train_embed_x[:,1],
#             c=color_map)
# plt.show()


test_embed_x = data.embed_X[np.array(data.test_idx)]
test_y = data.y[np.array(data.test_idx)]
color_map = plt.get_cmap("tab10")(test_y.astype(int))
plt.scatter(test_embed_x[:,0], test_embed_x[:,1],
            c=color_map)
plt.show()