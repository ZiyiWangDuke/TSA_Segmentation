import numpy as np
min_count = 3
eps = 5
weak_link = 0.9
strong_link=0.98
last_n_cluster = 2
last_geom = []
#plotly
alpha = 1
colors_plt = np.array([[10, 10, 10],
              [252, 115, 41],
              [83, 84, 209],
              [52, 163, 80],
              [109, 109, 109],
              [246, 198, 11],
              [109, 109, 109],
              [246, 198, 11],
              [109, 109, 109],
              [253, 165, 205],
              [188, 232, 57],
              [199, 182, 235],
              [137, 212, 237],
              [249, 240, 12],
              [229, 28, 23],
              [163, 108, 84],
              [232, 220, 171],
              [90, 124, 162],
              [254, 90, 254],
              [126, 67, 82],
              [152, 58, 151]], dtype=np.float64) / 255

colors_plotly = ['rgba(10, 10, 10, {})'.format(alpha),
          'rgba(252, 115, 41, {})'.format(alpha),
          'rgba(83, 84, 209, {})'.format(alpha),
          'rgba(52, 163, 80, {})'.format(alpha),
          'rgba(109, 109, 109, {})'.format(alpha),
          'rgba(246, 198, 11, {})'.format(alpha),
          'rgba(109, 109, 109, {})'.format(alpha),
          'rgba(253, 165, 205, {})'.format(alpha),
          'rgba(188, 232, 57, {})'.format(alpha),
          'rgba(199, 182, 235, {})'.format(alpha),
          'rgba(137, 212, 237, {})'.format(alpha),
          'rgba(249, 240, 12, {})'.format(alpha),
          'rgba(229, 28, 23, {})'.format(alpha),
          'rgba(163, 108, 84, {})'.format(alpha),
          'rgba(232, 220, 171, {})'.format(alpha),
          'rgba(90, 124, 162, {})'.format(alpha),
          'rgba(254, 90, 254, {})'.format(alpha),
          'rgba(126, 67, 82, {})'.format(alpha),
          'rgba(152, 58, 151, {})'.format(alpha)]
