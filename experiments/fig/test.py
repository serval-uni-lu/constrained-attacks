import comet_ml
experiment = comet_ml.OfflineExperiment()
experiment.log_metrics({'accuracy': 0.5, 'loss': 0.001})
experiment.end()