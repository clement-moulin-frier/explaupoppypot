from explauto.sensorimotor_model.nearest_neighbor import NearestNeighbor
from explauto.interest_model.discrete_progress import DiscretizedProgress
from explauto.interest_model.competences import competence_dist

sm_config = {'cls': NearestNeighbor,
             'config': {'sigma_ratio': 1. / 38}
             }

im_config = {'cls': DiscretizedProgress,
             'config': {'x_card': 400,
                        'win_size': 10,
                        'measure': competence_dist
                        }
             }

ag_config = {'n_bfs': 3,
             'sm': sm_config,
             'babbling': 'goal',
             'im': im_config,
             'bfs_half_range': 600.,
             'angle_half_range': 135.,
             's_mins': [-1., -0.7, -0.1],
             's_maxs': [1., 0.7, 0.7]
             }

env_config = {'motors': 'motors',
              'move_duration': 6.,
              't_reset': 1.,
              'm_half_range': 360.,
              's_mins': [-0.5, -0.7, 0.],
              's_maxs': [1., 0.7, 0.7]
              }

expe_config = {'scene': 'poppy-lying_sticky.ttt',
               'bootstrap_config': {'n': 16, 'bootstap_range_div': 48.},
               'n_runs': 20,
               'log_each': 10
               }
