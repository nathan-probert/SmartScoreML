from config import ENV
from smartscore_ml.custom_model import EnhancedModel

LAMBDA_API_NAME = f"Api-{ENV}"

PATH = "smartscore_ml\\lib"
MODEL_PATH = f"{PATH}\\model.pth"

FEATURES = ["gpg", "hgpg", "five_gpg", "tgpg", "otga", "hppg", "otshga", "home"]
MODEL_STRUCT = EnhancedModel(len(FEATURES))
