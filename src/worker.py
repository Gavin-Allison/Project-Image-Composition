import pika
import json
import time
import base64
from io import BytesIO
from PIL import Image


def process_task(data):
    header, encoded = data['image_data'].split(",", 1)
    image_bytes = base64.b64decode(encoded)
    img = Image.open(BytesIO(image_bytes))
    print(f"Processing image of size: {img.size}")
    
    # SIMULATED WORK: Replace this with your new model inference
    time.sleep(5) 
    
    print(f"Done processing!")

def main():
    print("Worker starting up...")
    # Connect to RabbitMQ
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='rabbitmq')
    )
    channel = connection.channel()

    channel.queue_declare(queue='task_queue', durable=True)
    
    # Fair dispatch: don't give more than one message to a worker at a time
    channel.basic_qos(prefetch_count=1)

    def callback(ch, method, properties, body):
        task_data = json.loads(body)
        
        try:
            process_task(task_data)
            # Acknowledge the message was processed successfully
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            print(f" [X] Error: {e}")
            # Optional: nack the message to requeue it
            # ch.basic_nack(delivery_tag=method.delivery_tag)

    channel.basic_consume(queue='task_queue', on_message_callback=callback)

    print('Waiting for messages.')
    channel.start_consuming()

if __name__ == '__main__':
    main()