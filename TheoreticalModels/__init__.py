from .AnnealedTransientTimeMotion import AnnealedTransientTimeMotion
from .ContinuousTimeRandomWalk import ContinuousTimeRandomWalk
from .FractionalBrownianMotion import FractionalBrownianMotion, FractionalBrownianMotionBrownian, FractionalBrownianMotionSubDiffusive, FractionalBrownianMotionSuperDiffusive
from .ScaledBrownianMotion import ScaledBrownianMotion, ScaledBrownianMotionBrownian, ScaledBrownianMotionSubDiffusive, ScaledBrownianMotionSuperDiffusive
from .LevyWalk import LevyWalk
from .TwoStateObstructedDiffusion import TwoStateObstructedDiffusion
from .TwoStateImmobilizedDiffusion import TwoStateImmobilizedDiffusion

ANDI_MODELS = [
    AnnealedTransientTimeMotion,
    ContinuousTimeRandomWalk,
    FractionalBrownianMotion,
    LevyWalk,
    ScaledBrownianMotion,
]

SUB_DIFFUSIVE_MODELS = [AnnealedTransientTimeMotion, ContinuousTimeRandomWalk, FractionalBrownianMotion, ScaledBrownianMotion]
SUP_DIFFUSIVE_MODELS = [LevyWalk, FractionalBrownianMotion, ScaledBrownianMotion]
BROWNIAN_MODELS = [FractionalBrownianMotion, ScaledBrownianMotion]

ALL_MODELS = ANDI_MODELS + [TwoStateObstructedDiffusion, TwoStateImmobilizedDiffusion]
SBM_MODELS = [ScaledBrownianMotionSubDiffusive, ScaledBrownianMotionBrownian, ScaledBrownianMotionSuperDiffusive]
FBM_MODELS = [FractionalBrownianMotionSubDiffusive, FractionalBrownianMotionBrownian, FractionalBrownianMotionSuperDiffusive]
SUB_MODELS = SBM_MODELS + FBM_MODELS
ALL_SUB_MODELS = SUB_MODELS + [LevyWalk, ContinuousTimeRandomWalk, AnnealedTransientTimeMotion]

STRING_LABEL_TO_MODEL = {model_class: model_class.STRING_LABEL for model_class in ALL_MODELS + SUB_MODELS}
