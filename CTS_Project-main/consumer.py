import pika
import json

# RabbitMQ setup
connection_params = pika.ConnectionParameters('localhost')
queue_name = 'claims_queue'
stop_stream = False

# Required feature columns
REQUIRED_COLUMNS = {
    "Provider",
    "is_inpatient",
    "is_groupcode",
    "ChronicCond_rheumatoidarthritis",
    "Beneficiaries_Count",
    "DeductibleAmtPaid",
    "InscClaimAmtReimbursed",
    "ChronicCond_Alzheimer",
    "ChronicCond_IschemicHeart",
    "Days_Admitted",
    "ChronicCond_stroke"
}

def consume_claims():
    """
    Generator: Consumes claims from RabbitMQ, validates required columns,
    and yields:
      - Valid claims: {"type": "valid", "data": { ... }}
      - Invalid claims: {"type": "invalid", "reason": "...", "data": { ... }}
      - Errors: {"type": "error", "reason": "..."}
    
    Format: Server-Sent Events (SSE)
    """
    global stop_stream
    stop_stream = False

    try:
        # Connect to RabbitMQ
        connection = pika.BlockingConnection(connection_params)
        channel = connection.channel()
        channel.queue_declare(queue=queue_name)

        print(f" [*] Waiting for messages in '{queue_name}'. To exit press CTRL+C")

        # Consume messages
        for method_frame, properties, body in channel.consume(queue=queue_name, inactivity_timeout=1):
            if stop_stream:
                print(" [!] Stream stopped by user.")
                break

            if body is None:
                continue  # timeout from inactivity_timeout, just keep looping

            try:
                claim = json.loads(body)

                # Validate: must be a dictionary
                if not isinstance(claim, dict):
                    yield f'data: {json.dumps({"type": "invalid", "reason": "Not a JSON object", "data": claim})}\n\n'
                    continue

                # Check for missing required columns
                received_keys = set(claim.keys())
                missing_cols = REQUIRED_COLUMNS - received_keys

                if missing_cols:
                    yield f'data: {json.dumps({{"type": "invalid", "reason": "Missing required fields: {list(missing_cols)}", "data": claim}})}\n\n'
                    continue

                # ✅ All good — send valid claim
                yield f'data: {json.dumps({"type": "valid", "data": claim})}\n\n'

            except json.JSONDecodeError:
                yield f'data: {json.dumps({"type": "invalid", "reason": "Invalid JSON format", "data": None})}\n\n'
            except Exception as e:
                yield f'data: {json.dumps({"type": "invalid", "reason": f"Processing error: {str(e)}", "data": None})}\n\n'

        # Clean shutdown
        channel.cancel()
        connection.close()

    except Exception as e:
        error_msg = f"Connection error: {str(e)}"
        print(f" [!] {error_msg}")
        yield f'data: {json.dumps({"type": "error", "reason": "{error_msg}"})}\n\n'


def clear_queue():
    """
    Clears all messages from the queue (useful before starting a new stream).
    """
    try:
        connection = pika.BlockingConnection(connection_params)
        channel = connection.channel()
        channel.queue_declare(queue=queue_name)  # Ensure queue exists
        purged_count = channel.queue_purge(queue=queue_name)
        print(f" [+] Purged {purged_count.method.message_count} messages from '{queue_name}'")
        connection.close()
    except Exception as e:
        print(f" [!] Failed to clear queue: {str(e)}")