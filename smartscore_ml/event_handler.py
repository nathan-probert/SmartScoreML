from decorators import lambda_handler_error_responder
from service import make_predictions


@lambda_handler_error_responder
def handle_make_predictions(event, context):
    players = event.get("players", [])

    entries = make_predictions(players)

    return {"statusCode": 200, "players": entries}
