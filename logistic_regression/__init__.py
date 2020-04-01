import os
import logging
from logging import config

# Version
__version__ = '0.1.0'

# Logging configurations
logging_version = 1
logging_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'logs')
logging_filename_base = 'logistic_regression'
package_name = 'logistic_regression'
disable_existing_loggers = False

loggers = {'': {'handlers': ['console', 'info_file_handler', 'error_file_handler'],
                'level': 'DEBUG',
                'propagate': True
                }
           }

handlers = {'console': {'class': 'logging.StreamHandler',
                        'level': 'DEBUG',
                        'formatter': 'simple',
                        'stream': 'ext://sys.stdout'
                        },
            'info_file_handler': {'class': 'logging.handlers.RotatingFileHandler',
                                  'level': 'INFO',
                                  'formatter': 'simple',
                                  'filename': os.path.join(logging_dir, f'{logging_filename_base}_info.log'),
                                  'maxBytes': 10485760,
                                  'backupCount': 20,
                                  'encoding': 'utf8'
                                  },
            'error_file_handler': {'class': 'logging.handlers.RotatingFileHandler',
                                   'level': 'ERROR',
                                   'formatter': 'simple',
                                   'filename': os.path.join(logging_dir, f'{logging_filename_base}_error.log'),
                                   'maxBytes': 10485760,
                                   'backupCount': 20,
                                   'encoding': 'utf8'
                                   }
            }

formatters = {'simple': {'format': '[%(levelname)s] %(name)s: %(message)s | %(asctime)s',
                         'class': 'logging.Formatter'
                         }
              }

logging_config_dict = {'version': logging_version,
                       'disable_existing_loggers': disable_existing_loggers,
                       'handlers': handlers,
                       'formatters': formatters,
                       'loggers': loggers
                       }

logging.config.dictConfig(logging_config_dict)
LOGGER = logging.getLogger(package_name)
