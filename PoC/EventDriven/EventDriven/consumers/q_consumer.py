import time
import pika
import psycopg2.pool

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()
channel.queue_declare(queue='run')

pool = psycopg2.pool.ThreadedConnectionPool(1, 10, user='postgres', host='localhost', database='postgres')

def execute(sql, commit=True):
    try:
        conn = pool.getconn()
    except psycopg2.Error as e:
        raise e
    with conn.cursor() as cur:
        try:
            cur.execute(sql)
            if commit:
                conn.commit()
        except psycopg2.Error as e:
            raise e
        finally:
            pool.putconn(conn)

def callback(ch, method, properties, body):
    print("received {}".format(body))
    time.sleep(0.03)
    execute('INSERT INTO files(name) VALUES (\'{}\')'.format(body.decode('utf-8')))
    # channel = connection.channel()
    # channel.queue_declare(queue='res')
    # channel.basic_publish(exchange='', routing_key='res', body='DONE!!')


channel.basic_consume(callback, queue='run', no_ack=True)
print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()

