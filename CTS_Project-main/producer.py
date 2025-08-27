import pika
import json
import random
import time

# RabbitMQ setup
connection_params = pika.ConnectionParameters('localhost')
queue_name = 'claims_queue'
stop_stream = False

# Track counts for balance (optional logging)
counts = {"non_fraud": 0, "medium": 0, "fraud": 0}

def generate_claim(category):
    """
    Generate a claim biased toward a risk category.
    """
    base_provider = f"PRV{random.randint(51001, 57763)}"

    if category == "non_fraud":
        # Normal, reasonable values
        return {
            "Provider": base_provider,
            "is_inpatient": random.randint(0, 10),
            "is_groupcode": random.randint(1, 200),
            "ChronicCond_rheumatoidarthritis": random.randint(0, 50),
            "Beneficiaries_Count": random.randint(1, 100),
            "DeductibleAmtPaid": round(random.uniform(0, 200), 2),
            "InscClaimAmtReimbursed": round(random.uniform(500, 1500), 2),
            "ChronicCond_Alzheimer": random.randint(0, 100),
            "ChronicCond_IschemicHeart": random.randint(0, 150),
            "Days_Admitted": round(random.uniform(1, 5), 2),
            "ChronicCond_stroke": random.randint(0, 20)
        }

    elif category == "medium":
        # Moderate values, some red flags
        return {
            "Provider": base_provider,
            "is_inpatient": random.randint(10, 100),
            "is_groupcode": random.randint(50, 400),
            "ChronicCond_rheumatoidarthritis": random.randint(50, 300),
            "Beneficiaries_Count": random.randint(100, 500),
            "DeductibleAmtPaid": round(random.uniform(100, 500), 2),
            "InscClaimAmtReimbursed": round(random.uniform(1500, 3000), 2),
            "ChronicCond_Alzheimer": random.randint(100, 500),
            "ChronicCond_IschemicHeart": random.randint(150, 1000),
            "Days_Admitted": round(random.uniform(5, 30), 2),
            "ChronicCond_stroke": random.randint(20, 100)
        }

    elif category == "fraud":
        # Suspicious patterns: high values, long stays, many claims
        return {
            "Provider": base_provider,
            "is_inpatient": random.randint(500, 640),
            "is_groupcode": random.randint(500, 604),
            "ChronicCond_rheumatoidarthritis": random.randint(2000, 2511),
            "Beneficiaries_Count": random.randint(2000, 2857),
            "DeductibleAmtPaid": round(random.uniform(800, 1068), 2),
            "InscClaimAmtReimbursed": round(random.uniform(3500, 5000), 2),
            "ChronicCond_Alzheimer": random.randint(7000, 8240),
            "ChronicCond_IschemicHeart": random.randint(5000, 6074),
            "Days_Admitted": round(random.uniform(100, 200), 2),
            "ChronicCond_stroke": random.randint(700, 810)
        }
    else:
        raise ValueError("Category must be 'non_fraud', 'medium', or 'fraud'")


def run_producer():
    global stop_stream
    connection = pika.BlockingConnection(connection_params)
    channel = connection.channel()
    channel.queue_declare(queue=queue_name)
    channel.queue_purge(queue=queue_name)  # Clear existing messages

    categories = ["non_fraud", "medium", "fraud"]

    print("Producer started. Sending balanced fraud/non-fraud/medium claims...")

    try:
        while not stop_stream:
            # Rotate category to ensure equal distribution
            category = random.choice(categories)  # Use random.choice for randomness
            # Or: category = categories[ (counts["non_fraud"] + counts["medium"] + counts["fraud"]) % 3 ]  # For strict round-robin

            claim = generate_claim(category)
            claim["risk_category"] = category  # Optional: Add label for consumer

            message = json.dumps(claim)
            channel.basic_publish(exchange='', routing_key=queue_name, body=message)

            # Update counter
            counts[category] += 1

            print(f"[{category.upper()}] Sent claim: {claim['InscClaimAmtReimbursed']:.2f} for {claim['Provider']} "
                  f"(Counts: N={counts['non_fraud']}, M={counts['medium']}, F={counts['fraud']})")

            time.sleep(5)  # Adjust delay as needed

    except KeyboardInterrupt:
        print("\nProducer stopped by user.")
    finally:
        print(f"Final counts: {dict(counts)}")
        connection.close()