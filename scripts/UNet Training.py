from PredictiveModel.UNetSingleParticleTracker import UNetSingleParticleTracker


for circle_size in [2,3]:
    network = UNetSingleParticleTracker(128,128,circle_size)
    network.fit()
    network.save_as_file()
